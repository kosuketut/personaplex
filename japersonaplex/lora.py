"""LoRA。凍結した bf16 の重みに、fp32 で持つ低ランク差分を足す。

対象は Temporal Transformer（と任意で Depformer）の self-attention の `in_proj_weight` と
`out_proj.weight`。`torch.nn.utils.parametrize` を使うので upstream のモジュールは書き換えない。
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.utils import parametrize

from moshi.modules.transformer import StreamingMultiheadAttention


class LoRA(nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_dim, device=weight.device, dtype=torch.float32) / rank**0.5)
        self.B = nn.Parameter(torch.zeros(out_dim, rank, device=weight.device, dtype=torch.float32))
        self.scale = alpha / rank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.B @ self.A * self.scale).to(weight.dtype)


def _targets(lm, depformer: bool):
    roots = [lm.transformer] + ([lm.depformer] if depformer else [])
    for root in roots:
        for module in root.modules():
            if isinstance(module, StreamingMultiheadAttention):
                yield module, "in_proj_weight"
                yield module.out_proj, "weight"


def add_lora(lm, rank: int, alpha: float, depformer: bool = True) -> list[nn.Parameter]:
    """全パラメータを凍結して LoRA を付け、学習対象のパラメータを返す。"""
    for p in lm.parameters():
        p.requires_grad = False
    params = []
    for module, name in _targets(lm, depformer):
        lora = LoRA(getattr(module, name), rank, alpha)
        parametrize.register_parametrization(module, name, lora, unsafe=True)
        params += [lora.A, lora.B]
    return params


def lora_state_dict(lm) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in lm.state_dict().items() if k.endswith((".0.A", ".0.B"))}


def save_lora(lm, path: str | Path, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(lm), str(path))
    path.with_suffix(".json").write_text(json.dumps(config, ensure_ascii=False, indent=2))


def load_lora(lm, path: str | Path) -> dict:
    """保存済みアダプタを付ける。config を返す。"""
    path = Path(path)
    config = json.loads(path.with_suffix(".json").read_text())
    add_lora(lm, config["lora_rank"], config["lora_alpha"], config["lora_depformer"])
    state = load_file(str(path))
    missing, unexpected = lm.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if k.endswith((".0.A", ".0.B"))]
    return config


def merge_lora(lm) -> None:
    """LoRA を重みに焼き込み、parametrization を外す（推論用）。"""
    for root in (lm.transformer, lm.depformer):
        for module in root.modules():
            if parametrize.is_parametrized(module):
                for name in list(module.parametrizations.keys()):
                    parametrize.remove_parametrizations(module, name, leave_parametrized=True)
