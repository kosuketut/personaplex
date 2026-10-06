"""単一 GPU の LoRA 学習ランナー。

例:
    NO_TORCH_COMPILE=1 python -m japersonaplex.train \
        --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
        --train data/pilot/npz/train --valid data/pilot/npz/heldout --out runs/pilot_v1
"""
import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

from moshi.models import loaders

from .data import PromptedDialogues, collate, compute_loss
from .lora import add_lora, lora_state_dict, save_lora


def save_state(out: Path, step: int, lm, optimizer, scheduler) -> None:
    """再開用の状態。LoRA の重みと optimizer / scheduler / 乱数状態。古いものは消す。"""
    path = out / f"state_step{step}.pt"
    torch.save({"step": step, "lora": lora_state_dict(lm), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    for old in out.glob("state_step*.pt"):
        if old != path:
            old.unlink()


def load_state(path: Path, lm, optimizer, scheduler, device) -> int:
    """save_state の逆。再開するステップ番号を返す。データの読み出し位置は復元しない（シャッフルし直す）。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = lm.load_state_dict({k: v.to(device) for k, v in state["lora"].items()}, strict=False)
    assert not unexpected, unexpected
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def latest_state(out: Path) -> Path | None:
    states = sorted(out.glob("state_step*.pt"), key=lambda p: int(p.stem.removeprefix("state_step")))
    return states[-1] if states else None


def evaluate(lm, loader, args, device) -> dict:
    sums, n = {}, 0
    with torch.no_grad(), parametrize.cached():
        for codes, loss_mask in loader:
            codes, loss_mask = codes.to(device), loss_mask.to(device)
            parts = compute_loss(lm, lm.forward_train(codes), codes, loss_mask, args.text_pad_weight,
                                 args.acoustic_weight, args.user_weight)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
    return {k: v / n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--extra-valid", action="append", default=[], metavar="NAME=DIR",
                        help="追加の検証セット（評価専用の名前・業種・声など）。metrics には valid_NAME で記録する。繰り返し指定できる")
    parser.add_argument("--extra-valid-limit", type=int, default=32, help="追加の検証セットは先頭からこの本数だけ使う")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=2048, help="PersonaPlex と同じ 163.84 秒")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--no-lora-depformer", action="store_true")
    parser.add_argument("--text-pad-weight", type=float, default=0.3)
    parser.add_argument("--acoustic-weight", type=float, default=0.02)
    parser.add_argument("--user-weight", type=float, default=0.0,
                        help="user 音声ストリームの損失の重み。0 なら agent 側だけ学習する")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="--out にある最新の state_step*.pt から再開する（同じ引数で起動すること）。"
                             "データの順序は復元しない（シャッフルし直す）")
    parser.add_argument("--stop-at", type=int, help="このステップで保存して終了する。学習率スケジュールは --steps に従う")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), lora_depformer=not args.no_lora_depformer)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    params = add_lora(lm, args.lora_rank, args.lora_alpha, depformer=not args.no_lora_depformer)
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.1f}M params", flush=True)

    collate_fn = partial(collate, pad_id=lm.zero_token_id)
    train_loader = DataLoader(PromptedDialogues(args.train, args.max_frames), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=2)
    valid_loaders = {"valid": DataLoader(PromptedDialogues(args.valid, args.max_frames), batch_size=args.batch_size,
                                         collate_fn=collate_fn)}
    for spec in args.extra_valid:
        name, directory = spec.split("=", 1)
        dataset = PromptedDialogues(directory, args.max_frames)
        dataset.files = dataset.files[: args.extra_valid_limit]
        valid_loaders[f"valid_{name}"] = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    evaluate_all = lambda: {k: evaluate(lm, loader, args, args.device) for k, loader in valid_loaders.items()}
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    schedule = lambda step: min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    log = open(out / "metrics.jsonl", "a")

    def record(row):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)

    start_step = 0
    state_path = latest_state(out) if args.resume else None
    if state_path is not None:
        start_step = load_state(state_path, lm, optimizer, scheduler, args.device)
        record({"step": start_step, "resumed_from": str(state_path)})
    elif args.resume:
        raise FileNotFoundError(f"--resume: no state_step*.pt under {out}")
    else:
        record({"step": 0, **evaluate_all()})
    batches = iter(train_loader)
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        running = {}
        for _ in range(args.grad_accum):
            try:
                codes, loss_mask = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                codes, loss_mask = next(batches)
            codes, loss_mask = codes.to(args.device), loss_mask.to(args.device)
            with parametrize.cached():
                parts = compute_loss(lm, lm.forward_train(codes), codes, loss_mask, args.text_pad_weight,
                                     args.acoustic_weight, args.user_weight)
                (parts["total"] / args.grad_accum).backward()
            if not torch.isfinite(parts["total"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            record({"step": step, "train": running, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - started), "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
        if step % args.eval_every == 0 or step == args.steps:
            record({"step": step, **evaluate_all()})
        if step % args.save_every == 0 or step == args.steps or step == args.stop_at:
            save_lora(lm, out / f"lora_step{step}.safetensors", config)
            save_state(out, step, lm, optimizer, scheduler)
        if step == args.stop_at:
            record({"step": step, "stopped": "--stop-at"})
            break


if __name__ == "__main__":
    main()
