"""flip 評価の出力音声を Whisper で書き起こし、テキスト出力と同じ基準で採点する。

moshi 用の環境とは依存が衝突しうるので別環境で実行する:
    uv run --no-project --python 3.10 --with faster-whisper python scripts/asr_flip.py runs/pilot_v2/flip_step200
"""
import json
import sys
from pathlib import Path

try:  # faster-whisper（ctranslate2）が venv の cuDNN・cuBLAS を見つけられるよう、torch を先に読み込む（align.py と同じ）
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
from faster_whisper import WhisperModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from japersonaplex.scoring import mentions  # noqa: E402


def main(directory: str, model_name: str = "medium"):
    directory = Path(directory)
    results = json.loads((directory / "results.json").read_text())
    model = WhisperModel(model_name, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
    rows = []
    for row in results["rows"]:
        asr = []
        for side in "AB":
            segments, _ = model.transcribe(str(directory / f"{row['id']}_{side}.wav"), language="ja", beam_size=5)
            asr.append("".join(s.text for s in segments))
        a_ok, b_ok = mentions(row["expect"][0], asr[0]), mentions(row["expect"][1], asr[1])
        crossed = mentions(row["expect"][1], asr[0]) or mentions(row["expect"][0], asr[1])
        rows.append({"id": row["id"], "kind": row["kind"], "expect": row["expect"], "asr": asr,
                     "a_ok": a_ok, "b_ok": b_ok, "pass": a_ok and b_ok and not crossed})
    summary = {"n": len(rows), "pass": sum(r["pass"] for r in rows),
               "side_correct": sum(r["a_ok"] + r["b_ok"] for r in rows), "sides": 2 * len(rows)}
    (directory / "results_asr.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:])
