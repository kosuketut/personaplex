#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-mkflipf
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# flip40 v2 と同じ問で、質問の形だけを変えたセットを作る（項目名を言わない言い換え、事情を添えた質問）。
#   sbatch slurm/make_flip_forms.sbatch
# 1. 台本 LLM が質問の文を書く（japersonaplex.datagen.flip_questions）→ ${SRC_FLIP%/*}_questions.jsonl
# 2. 形ごとに user の声で TTS し、元の最初の user 発話の位置につなぐ（make_flip --from --questions）→ ${OUT_PREFIX}_<形>
# 3. 質問の音声を whisper で確かめる（make_flip --check）
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
SRC="${SRC:-data/datagen/pilot40}"
SRC_FLIP="${SRC_FLIP:-${SRC}/flip40v2}"
QUESTIONS="${QUESTIONS:-${SRC_FLIP}_questions.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-${SRC}/flip40v3}"
FORMS="${FORMS:-paraphrase situational}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv。コンテナの CUDA 向けの変数を外す
echo "SRC=${SRC} SRC_FLIP=${SRC_FLIP} QUESTIONS=${QUESTIONS} OUT_PREFIX=${OUT_PREFIX} FORMS=${FORMS}"
for form in ${FORMS}; do
  test ! -e "${OUT_PREFIX}_${form}/flip.jsonl"  # 評価セットは作り直さない。作り直すなら手で消す
done

SETS="--set h_name='${SRC}/h_name' --set h_domain='${SRC}/h_domain' --set h_voice='${SRC}/h_voice' --set valid='${SRC}/main:valid.jsonl'"
run_in_container "
if [[ ! -s '${QUESTIONS}' ]]; then
  HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 data/envs/llm/bin/python -m japersonaplex.datagen.flip_questions \
    --flip '${SRC_FLIP}/flip.jsonl' --out '${QUESTIONS}'
fi
for form in ${FORMS}; do
  data/envs/tts/bin/python -m japersonaplex.datagen.make_flip --from '${SRC_FLIP}' --questions '${QUESTIONS}' --form \${form} \
    --out '${OUT_PREFIX}_'\${form} ${SETS}
  data/envs/align/bin/python -m japersonaplex.datagen.make_flip --check --out '${OUT_PREFIX}_'\${form}
done
"
