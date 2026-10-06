"""複数 GPU の全パラメータ学習（FSDP、fp32 の重みと Adam 状態、bf16 で計算）。Seiran の B200 向け。

torchrun で起動する:
    torchrun --nproc-per-node 8 -m japersonaplex.train_full \
        --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
        --train data/xxx/npz/train --valid data/xxx/npz/heldout --out runs/full_v1

学習率は PersonaPlex 論文と同じく Temporal Transformer 2e-6、Depformer 4e-6 が既定。
保存物:
    model_step{N}.safetensors  推論用の bf16 重み（upstream の moshi.server / offline がそのまま読める）
    resume_step{N}/            再開用。各 rank が自分のシャードと optimizer 状態を保存する（最新 1 つだけ残す）。
                               再開は同じ GPU 数・同じ設定でのみ可能
FSDP1 API を使う（ローカルの PyTorch 2.4 と Seiran の新しい PyTorch の両方で動かすため）。
"""
import argparse
import functools
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, DistributedSampler

from moshi.models import loaders
from moshi.modules.transformer import StreamingTransformerLayer

from .data import PromptedDialogues, collate, compute_loss


class ForwardTrain(nn.Module):
    """FSDP はルートの forward 経由でないとパラメータを集めないので、forward_train を forward にする。"""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, codes):
        return self.lm.forward_train(codes)


def wrap_fsdp(lm, device: torch.device, checkpoint_temporal: bool = True) -> FSDP:
    if checkpoint_temporal:
        temporal = set(lm.transformer.layers)
        apply_activation_checkpointing(
            lm,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: m in temporal,
        )
    return FSDP(
        ForwardTrain(lm),
        auto_wrap_policy=ModuleWrapPolicy({StreamingTransformerLayer}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def is_depformer_param(name: str) -> bool:
    """Depth Transformer 側か。depformer 本体、depformer_in / depformer_emb / depformer_text_emb、音声の出力層 linears。"""
    return clean_name(name).startswith(("depformer", "linears."))


def make_optimizer(model: FSDP, lr_temporal: float, lr_depformer: float, weight_decay: float):
    temporal, depformer = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (depformer if is_depformer_param(name) else temporal).append(p)
    return torch.optim.AdamW([{"params": temporal, "lr": lr_temporal}, {"params": depformer, "lr": lr_depformer}],
                             weight_decay=weight_decay, betas=(0.9, 0.95))


def clean_name(name: str) -> str:
    """FSDP / checkpoint_wrapper の接頭辞を除き、LMModel の state_dict のキーに戻す。"""
    for junk in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
        name = name.replace(junk, "")
    return name.removeprefix("lm.")


def save_inference_weights(model: FSDP, path: Path, rank: int) -> None:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        state = model.state_dict()
    if rank == 0:
        state = {clean_name(k): v.to(torch.bfloat16).contiguous() for k, v in state.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(path.with_suffix(".tmp")))
        path.with_suffix(".tmp").rename(path)


def save_resume(model: FSDP, optimizer, scheduler, step: int, out: Path, rank: int, position: dict) -> None:
    target = out / f"resume_step{step}"
    if rank == 0:
        target.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        torch.save({"model": model.state_dict(), "optim": FSDP.optim_state_dict(model, optimizer),
                    "world_size": dist.get_world_size(), "cuda_rng": torch.cuda.get_rng_state()},
                   target / f"rank{rank}.pt")
    dist.barrier()
    if rank == 0:
        (target / "trainer.json").write_text(json.dumps({"step": step, "scheduler": scheduler.state_dict(), **position}))
        for old in out.glob("resume_step*"):
            if old != target:
                shutil.rmtree(old)
    dist.barrier()


def latest_resume(out: Path) -> Path | None:
    done = [p for p in out.glob("resume_step*") if (p / "trainer.json").exists()]
    return max(done, key=lambda p: int(p.name.removeprefix("resume_step"))) if done else None


def load_resume(model: FSDP, optimizer, scheduler, path: Path) -> dict:
    state = torch.load(path / f"rank{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    if state["world_size"] != dist.get_world_size():
        raise RuntimeError(f"resume needs world size {state['world_size']}, got {dist.get_world_size()}")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, state["optim"]))
    torch.cuda.set_rng_state(state["cuda_rng"])
    trainer = json.loads((path / "trainer.json").read_text())
    scheduler.load_state_dict(trainer["scheduler"])
    return trainer


def reduce_mean(values: dict, device) -> dict:
    keys = sorted(values)
    t = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return {k: (t[i] / dist.get_world_size()).item() for i, k in enumerate(keys)}


def run_loss(model, lm, codes, loss_mask, args):
    return compute_loss(lm, model(codes), codes, loss_mask, args.text_pad_weight, args.acoustic_weight,
                        args.user_weight)


def evaluate(model, lm, loader, args, device) -> dict:
    model.eval()
    sums, n = {}, 0
    with torch.no_grad():
        for codes, loss_mask in loader:
            parts = run_loss(model, lm, codes.to(device), loss_mask.to(device), args)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
    model.train()
    return reduce_mean({k: v / max(1, n) for k, v in sums.items()}, device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--extra-valid", action="append", default=[], metavar="NAME=DIR",
                        help="追加の検証セット。metrics には valid_NAME で記録する（train.py と同じ）")
    parser.add_argument("--extra-valid-limit", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2, help="GPU あたり")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr-temporal", type=float, default=2e-6)
    parser.add_argument("--lr-depformer", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--text-pad-weight", type=float, default=0.3)
    parser.add_argument("--acoustic-weight", type=float, default=0.02)
    parser.add_argument("--user-weight", type=float, default=1.0,
                        help="user 音声ストリームの損失の重み（PersonaPlex と同じく既定で学習する）")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int, help="このステップで保存して終了する（Slurm の時間制限内でジョブをつなぐ用）。"
                                                    "学習率スケジュールは --steps に従う")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def train(args, lm) -> None:
    """lm は CPU 上の fp32 LMModel（全 rank で同じ重み）。"""
    rank, local_rank = dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    dist.barrier()

    model = wrap_fsdp(lm, device, checkpoint_temporal=not args.no_activation_checkpointing)
    optimizer = make_optimizer(model, args.lr_temporal, args.lr_depformer, args.weight_decay)
    schedule = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    collate_fn = functools.partial(collate, pad_id=lm.zero_token_id)
    train_set = PromptedDialogues(args.train, args.max_frames)
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn,
                              drop_last=True, num_workers=2)
    valid_sets = {"valid": PromptedDialogues(args.valid, args.max_frames)}
    for spec in args.extra_valid:
        name, directory = spec.split("=", 1)
        valid_sets[f"valid_{name}"] = PromptedDialogues(directory, args.max_frames)
        valid_sets[f"valid_{name}"].files = valid_sets[f"valid_{name}"].files[: args.extra_valid_limit]
    valid_loaders = {k: DataLoader(s, batch_size=args.batch_size, collate_fn=collate_fn,
                                   sampler=DistributedSampler(s, shuffle=False)) for k, s in valid_sets.items()}
    evaluate_all = lambda: {k: evaluate(model, lm, loader, args, device) for k, loader in valid_loaders.items()}

    log = open(out / "metrics.jsonl", "a") if rank == 0 else None

    def record(row):
        if rank == 0:
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)

    # データの読み出し位置（エポックとエポック内で消費したバッチ数）も保存し、再開時に同じ順序から続ける
    start_step, epoch, consumed = 0, 0, 0
    resume_path = latest_resume(out) if args.resume else None
    if resume_path is not None:
        trainer = load_resume(model, optimizer, scheduler, resume_path)
        start_step, epoch, consumed = trainer["step"], trainer["epoch"], trainer["consumed"]
        record({"step": start_step, "resumed_from": str(resume_path)})
    else:
        record({"step": 0, **evaluate_all()})

    sampler.set_epoch(epoch)
    batches = itertools.islice(iter(train_loader), consumed, None)
    started = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        running = {}
        for micro in range(args.grad_accum):
            try:
                codes, loss_mask = next(batches)
            except StopIteration:
                epoch, consumed = epoch + 1, 0
                sampler.set_epoch(epoch)
                batches = iter(train_loader)
                codes, loss_mask = next(batches)
            consumed += 1
            codes, loss_mask = codes.to(device), loss_mask.to(device)
            sync = micro == args.grad_accum - 1
            with (model.no_sync() if not sync else _nullcontext()):
                parts = run_loss(model, lm, codes, loss_mask, args)
                (parts["total"] / args.grad_accum).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum
        running = reduce_mean(running, device)
        if not math.isfinite(running["total"]):
            raise RuntimeError(f"non-finite loss at step {step}: {running}")
        grad_norm = model.clip_grad_norm_(1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == start_step + 1:
            record({"step": step, "train": running, "grad_norm": grad_norm, "lr": scheduler.get_last_lr(),
                    "elapsed_s": round(time.time() - started),
                    "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30})
        if step % args.eval_every == 0 or step == args.steps:
            record({"step": step, **evaluate_all()})
        stopping = step == args.stop_at
        if step % args.save_every == 0 or step == args.steps or stopping:
            save_inference_weights(model, out / f"model_step{step}.safetensors", rank)
            save_resume(model, optimizer, scheduler, step, out, rank, {"epoch": epoch, "consumed": consumed})
        if stopping:
            record({"step": step, "stopped": "--stop-at"})
            break


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main():
    args = parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    lm = loaders.get_moshi_lm(args.moshi_weight, device="cpu", dtype=torch.float32)
    try:
        train(args, lm)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
