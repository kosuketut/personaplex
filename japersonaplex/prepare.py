"""マニフェスト（jsonl）から学習用 npz を作る。

マニフェスト 1 行:
    {"id": ..., "audio": ステレオ wav（左 = agent、右 = user）, "words": 時刻付き単語 json,
     "text_prompt": 役割文, "voice_prompt": 声プロンプト wav}
words は [{"speaker": "A"|"B", "word": ..., "start": 秒, "end": 秒}, ...]。A が agent。
音声は推論時と同じくストリーミングの Mimi で 1 フレームずつ符号化する。

例:
    python -m japersonaplex.prepare --manifest data/pilot/train.jsonl --out data/pilot/npz/train \
        --tokenizer checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.models.lm import _iterate_audio, encode_from_sphn, load_audio, normalize_audio

from .sequence import align_text, build_example, build_prefix, wrap_with_system_tags

VOICE_PROMPT_LUFS = -24.0


def encode_mono(mimi, pcm: np.ndarray) -> np.ndarray:
    """pcm [1, T] -> codes [8, F]。"""
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    mimi.reset_streaming()
    frames = list(encode_from_sphn(mimi, _iterate_audio(pcm, sample_interval_size=frame_size, pad=True)))
    return torch.cat(frames, dim=-1)[0].cpu().numpy()


def encode_voice_prompt(mimi, path: str) -> np.ndarray:
    """LMGen.load_voice_prompt と同じ前処理（-24 LUFS 正規化）で符号化する。"""
    raw = normalize_audio(load_audio(path, mimi.sample_rate), mimi.sample_rate, VOICE_PROMPT_LUFS)
    return encode_mono(mimi, raw[None, :] if raw.ndim == 1 else raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mimi-weight")
    parser.add_argument("--silence-frames", type=int, default=6, help="moshi.server / offline と同じ 0.5 秒")
    parser.add_argument("--kanji-out", help="数字を漢数字（words の spoken）にしたテキストの npz もここに書く。"
                                            "音声の符号は共通（data_spec §5.4 の数字の表記の A/B 用）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    mimi_weight = args.mimi_weight or hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, args.device)
    mimi.streaming_forever(1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.kanji_out:
        Path(args.kanji_out).mkdir(parents=True, exist_ok=True)
    root = Path(args.manifest).parent

    voice_cache: dict[str, np.ndarray] = {}
    lines = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    with torch.no_grad():
        for i, item in enumerate(lines):
            voice_path = str(root / item["voice_prompt"])
            if voice_path not in voice_cache:
                voice_cache[voice_path] = encode_voice_prompt(mimi, voice_path)
            prompt_ids = tokenizer.encode(wrap_with_system_tags(item["text_prompt"]))
            prefix = build_prefix(voice_cache[voice_path], prompt_ids, args.silence_frames)

            pcm = load_audio(str(root / item["audio"]), mimi.sample_rate)
            assert pcm.shape[0] == 2, f"{item['audio']}: stereo (left=agent, right=user) expected"
            agent, user = encode_mono(mimi, pcm[0:1]), encode_mono(mimi, pcm[1:2])
            words = [w for w in json.loads((root / item["words"]).read_text()) if w["speaker"] == "A"]
            variants = [(out_dir, words)]
            if args.kanji_out:
                variants.append((Path(args.kanji_out), [dict(w, word=w.get("spoken", w["word"])) for w in words]))
            for directory, ws in variants:
                example = build_example(prefix, align_text(ws, tokenizer, agent.shape[1]), agent, user)
                np.savez(directory / f"{item['id']}.npz", codes=example.codes.astype(np.int32),
                         loss_start=example.loss_start)
            if (i + 1) % 50 == 0 or i + 1 == len(lines):
                print(f"{i + 1}/{len(lines)}", flush=True)


if __name__ == "__main__":
    main()
