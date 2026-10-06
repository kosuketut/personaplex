"""train_full（FSDP）の動作確認。torchrun で 2 GPU 以上で実行する（pytest では実行しない）:

    NO_TORCH_COMPILE=1 torchrun --nproc-per-node 2 tests/dist_smoke.py

小さい LMModel と乱数データで、6 step 設定で 3 step 目に止めて保存 → 再開して 6 step まで、を行い、
推論用重みが LMModel に strict で読み込めること、再開後に step が続くことを確かめる。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from japersonaplex.train_full import parse_args, train  # noqa: E402
from tests.test_sequence import tiny_lm  # noqa: E402


def make_data(root: Path, n: int, seed: int):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        T = int(rng.integers(40, 80))
        codes = np.concatenate([rng.integers(0, 32000, (1, T)), rng.integers(0, 2048, (16, T))]).astype(np.int32)
        np.savez(root / f"{i:03d}.npz", codes=codes, loss_start=10)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    holder = [tempfile.mkdtemp() if rank == 0 else None]
    dist.broadcast_object_list(holder)
    tmp = Path(holder[0])
    if rank == 0:
        make_data(tmp / "train", 16, 0)
        make_data(tmp / "valid", 4, 1)
    dist.barrier()
    common = ["--moshi-weight", "unused", "--train", str(tmp / "train"), "--valid", str(tmp / "valid"),
              "--extra-valid", f"again={tmp / 'valid'}", "--extra-valid-limit", "2",
              "--out", str(tmp / "run"), "--batch-size", "2", "--grad-accum", "2", "--save-every", "2",
              "--eval-every", "2", "--lr-temporal", "1e-3", "--lr-depformer", "1e-3", "--warmup", "1"]

    torch.manual_seed(0)
    train(parse_args(common + ["--steps", "6", "--stop-at", "3"]), tiny_lm().float())
    torch.manual_seed(0)
    train(parse_args(common + ["--steps", "6", "--resume"]), tiny_lm().float())
    # 比較用: 中断なしで 6 step
    straight = [c if c != str(tmp / "run") else str(tmp / "straight") for c in common]
    torch.manual_seed(0)
    train(parse_args(straight + ["--steps", "6"]), tiny_lm().float())

    if rank == 0:
        rows = [json.loads(line) for line in (tmp / "run" / "metrics.jsonl").read_text().splitlines()]
        assert any(r.get("resumed_from", "").endswith("resume_step3") for r in rows), rows
        assert [p.name for p in (tmp / "run").glob("resume_step*")] == ["resume_step6"]
        reference = tiny_lm()
        weights = load_file(str(tmp / "run" / "model_step6.safetensors"))
        reference.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
        before = load_file(str(tmp / "run" / "model_step3.safetensors"))
        changed = sum(not torch.equal(before[k], weights[k]) for k in weights)
        losses = [r["train"]["total"] for r in rows if "train" in r]
        reference_run = load_file(str(tmp / "straight" / "model_step6.safetensors"))
        # 保存重みは bf16 なので、比較は「bf16 で 1 ulp を超えて違う要素」の割合で見る
        differing, total, worst = 0, 0, 0.0
        for k in weights:
            a, b = weights[k].float(), reference_run[k].float()
            ulp = torch.finfo(torch.bfloat16).eps * a.abs().clamp_min(1e-3)
            differing += int(((a - b).abs() > ulp * 1.01).sum())
            total += a.numel()
            worst = max(worst, float(((a - b).abs() / ulp).max()))
        print(f"resumed vs uninterrupted: {differing}/{total} elements differ by > 1 bf16 ulp (worst {worst:.1f} ulp)")
        assert differing / total < 1e-3, "resumed run diverged from uninterrupted run"
        print(f"OK: {len(weights)} tensors, {changed} changed between step 3 and 6, train losses {losses}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
