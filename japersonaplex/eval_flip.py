"""プロンプト差し替え（flip）評価。

同じ user 音声に対し、事実 1 点だけが違うプロンプト A/B で生成し、agent のテキスト出力に
各プロンプトの期待値（例: 「十九時」）が現れるかを見る。生成は upstream の LMGen をそのまま使う。

    pass        : A で expect[0]、B で expect[1] が出て、相手側の期待値は出ない
    a_ok / b_ok : 片側ずつの正答

例:
    NO_TORCH_COMPILE=1 python -m japersonaplex.eval_flip --flip data/pilot/flip.jsonl \
        --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
        --tokenizer checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model \
        --lora runs/pilot_v1/lora_step600.safetensors --out runs/pilot_v1/flip_step600
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

from moshi.models import LMGen, loaders
from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio
from moshi.offline import seed_all, warmup

from .lora import load_lora, merge_lora
from .scoring import mentions
from .sequence import wrap_with_system_tags

SPECIAL = {0: "", 3: ""}

def generate(lm_gen, mimi, other_mimi, tokenizer, voice_prompt: str, text_prompt: str, user_audio: str):
    lm_gen.load_voice_prompt(voice_prompt)
    lm_gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(text_prompt))
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    user = load_audio(user_audio, mimi.sample_rate)
    pcm, text = [], []
    for encoded in encode_from_sphn(mimi, _iterate_audio(user, sample_interval_size=lm_gen._frame_size, pad=True)):
        tokens = lm_gen.step(encoded)
        if tokens is None:
            continue
        pcm.append(other_mimi.decode(tokens[:, 1:9]).cpu().numpy()[0, 0])
        token = tokens[0, 0, 0].item()
        text.append(SPECIAL.get(token, tokenizer.id_to_piece(token).replace("▁", "")))
    return np.concatenate(pcm), "".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip", required=True)
    parser.add_argument("--moshi-weight", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mimi-weight")
    parser.add_argument("--lora")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    mimi_weight = args.mimi_weight or hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi, other_mimi = loaders.get_mimi(mimi_weight, args.device), loaders.get_mimi(mimi_weight, args.device)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    if args.lora:
        load_lora(lm, args.lora)
        merge_lora(lm)
    lm.eval()
    lm_gen = LMGen(lm, device=args.device, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.flip).parent
    cases = [json.loads(line) for line in Path(args.flip).read_text().splitlines() if line.strip()][: args.limit]
    rows = []
    with torch.no_grad():
        warmup(mimi, other_mimi, lm_gen, args.device, lm_gen._frame_size)
        for case in cases:
            texts = []
            for side, prompt in zip("AB", case["prompts"]):
                seed_all(args.seed)
                pcm, text = generate(lm_gen, mimi, other_mimi, tokenizer, str(root / case["voice_prompt"]),
                                     prompt, str(root / case["user_audio"]))
                sphn.write_wav(str(out / f"{case['id']}_{side}.wav"), pcm, mimi.sample_rate)
                texts.append(text)
            a_ok, b_ok = mentions(case["expect"][0], texts[0]), mentions(case["expect"][1], texts[1])
            crossed = mentions(case["expect"][1], texts[0]) or mentions(case["expect"][0], texts[1])
            # 固有名詞: プロンプトに書かれた店名・担当者名を A/B 両方の出力で言えているか
            names = re.search("あなたは(.+?)という.+?名前は(.+?)です", case["prompts"][0])
            shop_ok, staff_ok = (all(name in text for text in texts) for name in names.groups()) if names else (None, None)
            rows.append({"id": case["id"], "kind": case["kind"], "question": case["question"],
                         "expect": case["expect"], "text": texts, "a_ok": a_ok, "b_ok": b_ok,
                         "pass": a_ok and b_ok and not crossed, "shop_ok": shop_ok, "staff_ok": staff_ok})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = {"n": len(rows), "pass": sum(r["pass"] for r in rows),
               "side_correct": sum(r["a_ok"] + r["b_ok"] for r in rows), "sides": 2 * len(rows),
               "shop_ok": sum(bool(r["shop_ok"]) for r in rows), "staff_ok": sum(bool(r["staff_ok"]) for r in rows)}
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
