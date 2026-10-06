#!/usr/bin/env bash
# 台本 jsonl から学習用 npz までを通す: TTS -> アライメント -> 組み立て（品質検査に落ちた発話は作り直す）-> prepare。
#
#   scripts/datagen.sh <out_dir> <scripts.jsonl> [gpu]
#
# gpu を省くと CUDA_VISIBLE_DEVICES をそのまま使う（Slurm が割り当てた GPU）。
# 環境: data/envs/tts（Qwen3-TTS）、data/envs/align（アライナと whisper）、.venv（組み立てと prepare）。
# 環境の python は PY_TTS、PY_ALIGN、PY_MAIN で差し替えられる（Seiran では data/envs/main を使う）。
# ASR_CHECK=0 で whisper の CER 検査を省く（速いが、内容の誤りは CTC の unexplained_speech_s でしか見ない）。
# VOICE_SPLIT=heldout で評価専用の声だけを使う（H-voice）。MIMI_WEIGHT で Mimi の重みのファイルを指定できる
# （既定は prepare.py の既定。Seiran では llm-jp-moshi-v1 に同梱のものを使う）。
# NPZ_KANJI=1 で、数字を漢数字にしたテキストの npz も $OUT/npz_kanji に作る（数字の表記の A/B 用。音声は共通）。
set -euo pipefail
OUT=$1; SCRIPTS=$2; GPU=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TOK=${TOK:-$ROOT/checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model}
VOICES=${VOICES:-$ROOT/data/voicebank/refs_v0/manifest.jsonl}
VOICE_SPLIT=${VOICE_SPLIT:-train}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
TTS_BATCH=${TTS_BATCH:-64}
TOKEN_BUDGET=${TOKEN_BUDGET:-6000}
PY_TTS=${PY_TTS:-$ROOT/data/envs/tts/bin/python}
PY_ALIGN=${PY_ALIGN:-$ROOT/data/envs/align/bin/python}
PY_MAIN=${PY_MAIN:-$ROOT/.venv/bin/python}
ASR=$([ "${ASR_CHECK:-1}" = 1 ] && echo --asr || true)
export PYTHONPATH=$ROOT NO_TORCH_COMPILE=1
if [ -n "$GPU" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
cd "$ROOT"
mkdir -p "$OUT"
t() { local s=$(date +%s); "$@"; echo "$(date -Is) $3 ... $(( $(date +%s) - s ))s" >> "$OUT/timing.log"; }
TTS=("$PY_TTS" -m japersonaplex.datagen.tts --scripts "$SCRIPTS" --voices "$VOICES" --out "$OUT" --voice-split "$VOICE_SPLIT")

if [ ! -s "$OUT/tts/utts.jsonl" ]; then
  t "${TTS[@]}" --batch "$TTS_BATCH" --token-budget "$TOKEN_BUDGET"
fi
# 声プロンプトが 3.5〜5.5 秒に入らなかった声があれば、文を替えて作り足す
t "${TTS[@]}" --fill-voices
# 相槌の音声（声と役割ごと、§3.3）を作り、読みが合うものだけ残す
if [ ! -s "$OUT/tts/backchannels_all.jsonl" ]; then
  t "${TTS[@]}" --backchannels --batch $(( TTS_BATCH * 2 )) --token-budget $(( TOKEN_BUDGET * 4 / 3 ))
  t "$PY_ALIGN" -m japersonaplex.datagen.align --out "$OUT" --check-backchannels
fi
for round in $(seq 1 "$MAX_ATTEMPTS"); do
  t "$PY_ALIGN" -m japersonaplex.datagen.align --out "$OUT" $ASR
  t "$PY_MAIN" -m japersonaplex.datagen.assemble --out "$OUT" --tokenizer "$TOK" --max-attempts "$MAX_ATTEMPTS"
  [ -s "$OUT/redo.txt" ] || break
  t "${TTS[@]}" --batch "$TTS_BATCH" --token-budget "$TOKEN_BUDGET" --redo "$OUT/redo.txt"
done
rm -rf "$OUT/npz" "$OUT/npz_kanji"  # 前回の実行で作った、今回は捨てた対話の npz を残さない
t "$PY_MAIN" -m japersonaplex.prepare --manifest "$OUT/train.jsonl" --out "$OUT/npz" --tokenizer "$TOK" \
  ${MIMI_WEIGHT:+--mimi-weight "$MIMI_WEIGHT"} $([ "${NPZ_KANJI:-0}" = 1 ] && echo --kanji-out "$OUT/npz_kanji" || true)
head -c 400 "$OUT/qc.json"; echo
