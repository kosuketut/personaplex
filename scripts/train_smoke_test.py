"""学習の実現可能性チェック: forward_train が 17 ストリーム (text + agent 8 + user 8) で動くか、
コードブック別の loss、逆伝播時のピーク VRAM を測る。重みは更新しない。

例:
    NO_TORCH_COMPILE=1 python scripts/train_smoke_test.py \
        --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
        --agent-wav outputs/baseline/jp_wavvoice.wav --user-wav assets/test_ja/input_ja.wav
"""
import argparse

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.models.lm import load_audio


def encode(mimi, path, device):
    wav = torch.from_numpy(load_audio(path, mimi.sample_rate)).to(device)[None, :1]
    frame = int(mimi.sample_rate / mimi.frame_rate)
    wav = wav[..., : wav.shape[-1] // frame * frame]
    with torch.no_grad():
        return mimi.encode(wav)  # [1, 8, T]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight")
    parser.add_argument("--agent-wav", required=True)
    parser.add_argument("--user-wav", required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--trainable", choices=["all", "depformer+text", "none"], default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    mimi = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), args.device)
    weight = args.moshi_weight or hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(weight, device=args.device)

    agent = encode(mimi, args.agent_wav, args.device)[..., : args.frames]
    user = encode(mimi, args.user_wav, args.device)[..., : args.frames]
    T = min(agent.shape[-1], user.shape[-1])
    # テキストは時刻アライン済み書き起こしが無いので全フレーム PAD とする（text loss は参考値）
    text = torch.full((1, 1, T), lm.text_padding_token_id, device=args.device, dtype=torch.long)
    codes = torch.cat([text, agent[..., :T], user[..., :T]], dim=1)
    print("codes", tuple(codes.shape), "num_codebooks", lm.num_codebooks, "dep_q", lm.dep_q)

    for name, p in lm.named_parameters():
        p.requires_grad = args.trainable == "all" or (
            args.trainable == "depformer+text" and (name.startswith("depformer") or "text" in name or name.startswith("linears"))
        )
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e9:.2f}B / {sum(p.numel() for p in lm.parameters()) / 1e9:.2f}B")

    torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(n_train > 0):
        out = lm.forward_train(codes)
        target = codes[:, lm.audio_offset: lm.audio_offset + lm.dep_q]
        losses = []
        for k in range(lm.dep_q):
            mask = out.mask[:, k]
            losses.append(F.cross_entropy(out.logits[:, k][mask].float(), target[:, k][mask]))
        text_mask = out.text_mask[:, 0]
        text_loss = F.cross_entropy(out.text_logits[:, 0][text_mask].float(), codes[:, 0][text_mask])
        loss = torch.stack(losses).mean() + text_loss
        if n_train > 0:
            loss.backward()
    fmt = lambda xs: " ".join(f"{x.item():.2f}" for x in xs)
    print("agent audio loss (cb0..7):", fmt(losses[:8]))
    print("user  audio loss (cb0..7):", fmt(losses[8:]))
    print(f"text(pad) loss: {text_loss.item():.3f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB at T={T} frames ({T / 12.5:.0f}s), batch 1")


if __name__ == "__main__":
    main()
