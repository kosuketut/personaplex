"""LoRA をベース重みに焼き込み、upstream の moshi.server / moshi.offline がそのまま読める重みを書き出す。

例:
    python -m japersonaplex.merge --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
        --lora runs/pilot_v2/lora_step200.safetensors --out checkpoints/pilot_v2_step200/model.safetensors
"""
import argparse
from pathlib import Path

from safetensors.torch import save_file

from moshi.models import loaders

from .lora import load_lora, merge_lora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    expected = set(lm.state_dict())
    load_lora(lm, args.lora)
    merge_lora(lm)
    state = {k: v.detach().cpu().contiguous() for k, v in lm.state_dict().items()}
    assert set(state) == expected, sorted(set(state) ^ expected)[:5]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, args.out)
    print(f"wrote {args.out} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
