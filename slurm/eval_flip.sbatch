#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-flip
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# LoRA のチェックポイントを flip 評価する配列ジョブ。配列の 1 タスクが EVAL_STEPS の 1 つを受け持つ（0 は LoRA なし）。
#   RUN_NAME=pilot40_lora_v1 EVAL_STEPS="0 400 800 1200" sbatch --array=0-3 slurm/eval_flip.sbatch
# FULL=1 なら全パラメータ学習の重み（outputs/<RUN_NAME>/model_step<N>.safetensors）を評価する。
# FLIPS の各 flip.jsonl について、テキスト出力の採点（eval_flip）と出力音声の ASR 採点（asr_flip、GPU）をする。
# 出力: outputs/<RUN_NAME>/eval_<flip.jsonl のあるディレクトリ名>_step<N><OUT_SUFFIX>/。results_asr.json があれば飛ばす。
# EVAL_ARGS は eval_flip に足す引数（例: 評価のばらつきを見るなら EVAL_ARGS="--seed 1" OUT_SUFFIX=_seed1）。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN_NAME="${RUN_NAME:-pilot40_lora_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
BASE_DIR="${BASE_DIR:-checkpoints/llm-jp-moshi-v1-pp16}"
EVAL_STEPS="${EVAL_STEPS:-0}"
FLIPS="${FLIPS:-data/pilot/flip.jsonl data/datagen/pilot40/flip40/flip.jsonl}"
EVAL_ARGS="${EVAL_ARGS:-}"
OUT_SUFFIX="${OUT_SUFFIX:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs "${OUTPUT_DIR}"
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv。コンテナの CUDA 向けの変数を外す

steps=(${EVAL_STEPS})
STEP="${steps[${SLURM_ARRAY_TASK_ID:-0}]}"
WEIGHT="${BASE_DIR}/model.safetensors"
LORA_ARG=""
if [[ "${STEP}" != 0 && "${FULL:-0}" == 1 ]]; then  # 全パラメータ学習（train_full.py）の重み
  WEIGHT="${OUTPUT_DIR}/model_step${STEP}.safetensors"
  test -f "${WEIGHT}"
elif [[ "${STEP}" != 0 ]]; then
  LORA_ARG="--lora ${OUTPUT_DIR}/lora_step${STEP}.safetensors"
  test -f "${OUTPUT_DIR}/lora_step${STEP}.safetensors"
fi
echo "RUN_NAME=${RUN_NAME} OUTPUT_DIR=${OUTPUT_DIR} STEP=${STEP} WEIGHT=${WEIGHT} LORA_ARG=${LORA_ARG} FLIPS=${FLIPS} EVAL_ARGS=${EVAL_ARGS} OUT_SUFFIX=${OUT_SUFFIX}"

for flip in ${FLIPS}; do
  out="${OUTPUT_DIR}/eval_$(basename "$(dirname "${flip}")")_step${STEP}${OUT_SUFFIX}"
  if [[ -f "${out}/results_asr.json" ]]; then
    echo "skip ${out}: done"
    continue
  fi
  echo "== ${flip} -> ${out}"
  run_in_container "
\${PY_MAIN} -m japersonaplex.eval_flip --flip '${flip}' \
  --moshi-weight '${WEIGHT}' --tokenizer '${BASE_DIR}/tokenizer_spm_32k_3.model' \
  --mimi-weight '${BASE_DIR}/tokenizer-e351c8d8-checkpoint125.safetensors' ${LORA_ARG} --out '${out}' ${EVAL_ARGS}
data/envs/align/bin/python scripts/asr_flip.py '${out}'
"
done
