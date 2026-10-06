#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# 単一 GPU の LoRA 学習と、その後の flip 評価（出力音声の ASR 採点まで）。
#   RUN_NAME=pilot40_lora_v1 STEPS=800 sbatch slurm/train_lora.sbatch
# 環境は scripts/seiran_setup_envs.sh の data/envs/main（学習・flip）と data/envs/align（ASR 採点の faster-whisper）。
# 学習データの既定は data/datagen/pilot40/main（QA は入れない。2026-10-02 決定）。EXTRA_VALID の評価専用セット
# （名前・業種・声）の損失も valid_<名前> として metrics.jsonl に記録する。
# 同じ RUN_NAME で投げ直すと --resume で続きから進む。FLIP の flip.jsonl があれば、FLIP_STEP の LoRA で評価する。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN_NAME="${RUN_NAME:-pilot40_lora_v1}"
DATA="${DATA:-data/datagen/pilot40}"
TRAIN_DIR="${TRAIN_DIR:-${DATA}/main/npz_train}"
VALID_DIR="${VALID_DIR:-${DATA}/main/npz_valid}"
EXTRA_VALID="${EXTRA_VALID:-h_name=${DATA}/h_name/npz_train h_domain=${DATA}/h_domain/npz_train h_voice=${DATA}/h_voice/npz_train}"
BASE_DIR="${BASE_DIR:-checkpoints/llm-jp-moshi-v1-pp16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
STEPS="${STEPS:-800}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
FLIP="${FLIP:-data/pilot/flip.jsonl}"
FLIP_STEP="${FLIP_STEP:-${STEPS}}"
FLIP_LIMIT="${FLIP_LIMIT:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs "${OUTPUT_DIR}"
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv。コンテナの CUDA 向けの変数を外す

RESUME_FLAG=""
if compgen -G "${OUTPUT_DIR}/state_step*.pt" >/dev/null; then
  RESUME_FLAG="--resume"
fi
EXTRA_ARGS=""
for spec in ${EXTRA_VALID}; do
  EXTRA_ARGS+=" --extra-valid ${spec}"
done
FLIP_OUT="${OUTPUT_DIR}/flip_step${FLIP_STEP}"

echo "RUN_NAME=${RUN_NAME} OUTPUT_DIR=${OUTPUT_DIR} STEPS=${STEPS} RESUME_FLAG=${RESUME_FLAG}"
echo "TRAIN_DIR=${TRAIN_DIR} VALID_DIR=${VALID_DIR} EXTRA_VALID=${EXTRA_VALID}"
echo "BASE_DIR=${BASE_DIR} TRAIN_ARGS=${TRAIN_ARGS} FLIP=${FLIP} FLIP_STEP=${FLIP_STEP} FLIP_LIMIT=${FLIP_LIMIT}"

run_in_container "
\${PY_MAIN} -m japersonaplex.train --moshi-weight '${BASE_DIR}/model.safetensors' \
  --train '${TRAIN_DIR}' --valid '${VALID_DIR}' ${EXTRA_ARGS} --out '${OUTPUT_DIR}' --steps ${STEPS} \
  ${RESUME_FLAG} ${TRAIN_ARGS}
if [[ -f '${FLIP}' ]]; then
  \${PY_MAIN} -m japersonaplex.eval_flip --flip '${FLIP}' \
    --moshi-weight '${BASE_DIR}/model.safetensors' --tokenizer '${BASE_DIR}/tokenizer_spm_32k_3.model' \
    --mimi-weight '${BASE_DIR}/tokenizer-e351c8d8-checkpoint125.safetensors' \
    --lora '${OUTPUT_DIR}/lora_step${FLIP_STEP}.safetensors' --out '${FLIP_OUT}' ${FLIP_LIMIT:+--limit ${FLIP_LIMIT}}
  data/envs/align/bin/python scripts/asr_flip.py '${FLIP_OUT}'
else
  echo 'skip flip: ${FLIP} not found'
fi
"
