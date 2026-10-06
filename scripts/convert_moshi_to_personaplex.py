"""Moshi 形式 (depformer 8 codebook) の重みを PersonaPlex 形式 (dep_q=16) に変換する。

loaders.get_moshi_lm と同じ規則（self_attn の拡張、0..7 -> 8..15 のコピー）を適用し、
コピー元が存在しないテンソル（depformer_emb.7）はゼロで初期化する。
depformer_emb.7 は agent の最終コードブックのトークンをユーザ側最初の depformer ステップへ
渡す埋め込みで、元の Moshi には対応物が無い。depformer_emb.6 のコピーで初期化する手もある。

例:
    python scripts/convert_moshi_to_personaplex.py \
        --src ~/.cache/huggingface/hub/models--llm-jp--llm-jp-moshi-v1/snapshots/<rev>/model.safetensors \
        --dst checkpoints/llm-jp-moshi-v1-pp16/model.safetensors
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from moshi.models import loaders
from moshi.models.lm import LMModel

COPY_GROUPS = ["gating", "linears", "depformer_in", "depformer_emb"]


def convert(src: str, dst: str) -> None:
    lm_kwargs = dict(loaders._lm_kwargs)
    lm_kwargs["dep_q"] = 16
    target = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs).state_dict()
    state = load_file(src, device="cpu")

    expanded, copied, zeroed = 0, 0, []
    for name, tensor in list(state.items()):
        if name in target and tensor.shape != target[name].shape:
            assert "depformer" in name and "self_attn" in name, (name, tensor.shape, target[name].shape)
            state[name] = torch.cat([tensor, tensor], dim=0)
            expanded += 1
    for name, ref in target.items():
        if name in state:
            continue
        for old, new in zip(range(8), range(8, 16)):
            for group in COPY_GROUPS:
                needle = f"{group}.{new}."
                src_name = name.replace(needle, f"{group}.{old}.")
                if needle in name and src_name in state:
                    state[name] = state[src_name].clone()
                    copied += 1
                    break
            if name in state:
                break
        if name not in state:
            state[name] = torch.zeros(ref.shape, dtype=torch.bfloat16)
            zeroed.append(name)

    unexpected = sorted(set(state) - set(target))
    assert not unexpected, unexpected
    for name, ref in target.items():
        assert state[name].shape == ref.shape, (name, state[name].shape, ref.shape)
        state[name] = state[name].to(torch.bfloat16)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, dst)
    print(f"expanded={expanded} copied={copied} zero-initialized={zeroed}")
    print(f"wrote {dst} ({len(state)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)
