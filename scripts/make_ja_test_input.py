"""日本語ベースライン用のテスト入力 wav を pyopenjtalk で合成し、
テキストトークン数/秒（Moshi のフレームレート 12.5 Hz との比較）を出す。"""
import glob
import json
import os

import numpy as np
import pyopenjtalk
import sentencepiece as spm
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
FRAME_RATE = 12.5
TOTAL_SEC = 40.0
# (開始秒, 発話)
UTTERANCES = [
    (1.0, "こんにちは。ちょっと相談してもいいですか。"),
    (10.0, "最近、夜なかなか眠れなくて困っているんです。何かいい方法はありますか。"),
    (26.0, "なるほど。寝る前にスマホを見るのは、やっぱりよくないんでしょうか。"),
]

HUB = os.path.expanduser("~/.cache/huggingface/hub")
TOKENIZERS = {
    "personaplex(en)": glob.glob(f"{HUB}/models--nvidia--personaplex-7b-v1/snapshots/*/tokenizer_spm_32k_3.model")[0],
    "llm-jp-moshi(ja)": glob.glob(f"{HUB}/models--llm-jp--llm-jp-moshi-v1/snapshots/*/tokenizer_spm_32k_3.model")[0],
}


def main():
    out = np.zeros(int(TOTAL_SEC * SR), dtype=np.float32)
    tokenizers = {k: spm.SentencePieceProcessor(v) for k, v in TOKENIZERS.items()}
    rows = []
    for start, text in UTTERANCES:
        wav, sr = pyopenjtalk.tts(text)
        wav = resample_poly(wav / 32768.0, SR, sr).astype(np.float32)
        begin = int(start * SR)
        out[begin:begin + len(wav)] += wav[: len(out) - begin]
        dur = len(wav) / SR
        row = {"start": start, "dur": round(dur, 2), "text": text, "mora_per_s": round(len(pyopenjtalk.g2p(text, kana=True)) / dur, 1)}
        for name, tok in tokenizers.items():
            n = len(tok.encode(text))
            row[name] = {"tokens": n, "tokens_per_s": round(n / dur, 1), "fits": n / dur <= FRAME_RATE}
        rows.append(row)
    out *= 0.5 / max(1e-6, np.abs(out).max())
    sf.write("assets/test_ja/input_ja.wav", out, SR, subtype="PCM_16")
    with open("assets/test_ja/input_ja.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
