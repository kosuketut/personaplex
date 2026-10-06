#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-full
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G

# 1 ノード 8 GPU の全パラメータ学習（FSDP、japersonaplex/train_full.py）。環境は data/envs/main（torch 2.8.0+cu128）。
#   RUN_NAME=full_v1 STEPS=2000 sbatch slurm/train_full.sbatch
# 学習データの既定は data/datagen/pilot40v2/main（算用数字のテキスト）。EXTRA_VALID の評価専用セットの損失も記録する。
# 時間切れに備え、同じ RUN_NAME で再投入すれば --resume で続きから再開する（同じ GPU 数のときだけ）。
# メモリ: 各 rank が fp32 の重み（約 34 GB）を CPU に読んでから FSDP で分けるので、8 rank で 300 GB 近く使う。
# 評価: FULL=1 RUN_NAME=full_v1 EVAL_STEPS="..." sbatch --array=... slurm/eval_flip.sbatch
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN_NAME="${RUN_NAME:-full_v1}"
DATA="${DATA:-data/datagen/pilot40v2}"
TRAIN_DIR="${TRAIN_DIR:-${DATA}/main/npz_train}"
VALID_DIR="${VALID_DIR:-${DATA}/main/npz_valid}"
EXTRA_VALID="${EXTRA_VALID:-h_name=${DATA}/h_name/npz_train h_domain=${DATA}/h_domain/npz_train h_voice=${DATA}/h_voice/npz_train}"
BASE_DIR="${BASE_DIR:-checkpoints/llm-jp-moshi-v1-pp16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
STEPS="${STEPS:-2000}"
NPROC="${NPROC:-8}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs "${OUTPUT_DIR}"
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv。コンテナの CUDA 向けの変数を外す

RESUME_FLAG=""
if compgen -G "${OUTPUT_DIR}/resume_step*/trainer.json" >/dev/null; then
  RESUME_FLAG="--resume"
fi
EXTRA_ARGS=""
for spec in ${EXTRA_VALID}; do
  EXTRA_ARGS+=" --extra-valid ${spec}"
done

echo "RUN_NAME=${RUN_NAME} OUTPUT_DIR=${OUTPUT_DIR} NPROC=${NPROC} STEPS=${STEPS} RESUME_FLAG=${RESUME_FLAG}"
echo "TRAIN_DIR=${TRAIN_DIR} VALID_DIR=${VALID_DIR} EXTRA_VALID=${EXTRA_VALID}"
echo "BASE_DIR=${BASE_DIR} TRAIN_ARGS=${TRAIN_ARGS}"

run_in_container "
\${PY_MAIN} -m torch.distributed.run --standalone --nproc-per-node ${NPROC} -m japersonaplex.train_full \
  --moshi-weight '${BASE_DIR}/model.safetensors' \
  --train '${TRAIN_DIR}' --valid '${VALID_DIR}' ${EXTRA_ARGS} --out '${OUTPUT_DIR}' \
  --steps ${STEPS} ${RESUME_FLAG} ${TRAIN_ARGS}
"
