"""学習の再開用状態の保存・読込が LoRA 重みと optimizer を元どおりに戻すことを確認する。"""
import math

import torch

from japersonaplex.lora import add_lora, lora_state_dict
from japersonaplex.train import latest_state, load_state, save_state
from tests.test_sequence import tiny_lm


def make(lm):
    params = add_lora(lm, rank=4, alpha=4.0)
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / 10)))
    return params, optimizer, scheduler


def test_save_and_resume(tmp_path):
    lm = tiny_lm()
    params, optimizer, scheduler = make(lm)
    for step in (1, 2):
        sum((p ** 2).sum() for p in params).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        save_state(tmp_path, step, lm, optimizer, scheduler)
    assert [p.name for p in tmp_path.glob("state_step*.pt")] == ["state_step2.pt"]

    resumed = tiny_lm()
    _, optimizer2, scheduler2 = make(resumed)
    assert load_state(latest_state(tmp_path), resumed, optimizer2, scheduler2, "cpu") == 2
    for k, v in lora_state_dict(lm).items():
        assert torch.equal(v, lora_state_dict(resumed)[k]), k
    assert scheduler2.last_epoch == scheduler.last_epoch
    assert optimizer2.state_dict()["state"][0]["step"] == optimizer.state_dict()["state"][0]["step"]


def test_depformer_param_groups():
    """全パラメータ学習で Depformer 側の学習率を使うパラメータの振り分け。"""
    from japersonaplex.train_full import is_depformer_param

    names = [n for n, _ in tiny_lm().named_parameters()]
    depformer = {n for n in names if is_depformer_param("_fsdp_wrapped_module.lm." + n)}
    assert {n for n in names if n.startswith("linears.")} <= depformer
    assert {n for n in names if n.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb"))} <= depformer
    assert not {n for n in names if n.startswith(("transformer.", "emb.", "text_emb", "text_linear", "out_norm"))} & depformer
    assert is_depformer_param("_fsdp_wrapped_module.lm.depformer.layers._fsdp_wrapped_module.0.self_attn.in_proj_weight")
