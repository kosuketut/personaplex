#!/bin/bash
# datagen_*.sbatch から source する共通部分。PROJECT_DIR と CONTAINER_IMAGE は呼び出し側で決めておく。
# 環境は scripts/seiran_setup_envs.sh がプロジェクトの下に作った uv の venv（data/envs/*）。コンテナは OS の
# ライブラリと GPU ドライバ（--nv）のためだけに使い、python は venv のものを使う。
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-}"
export TMPDIR="${TMPDIR:-${PROJECT_DIR}/tmp}"
mkdir -p "${TMPDIR}" "${CACHE_DIR}"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "Container image not found: ${CONTAINER_IMAGE}" >&2
  echo "Override with: CONTAINER_IMAGE=/path/to/image.sif sbatch ..." >&2
  exit 1
fi
if [[ -z "${CONTAINER_ENGINE}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_ENGINE="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_ENGINE="singularity"
  else
    echo "Neither apptainer nor singularity found on PATH" >&2
    exit 1
  fi
fi

# コンテナ（NGC PyTorch 25.10 系）は TRITON_*_PATH、CUDA_HOME、LD_LIBRARY_PATH をコンテナの CUDA 13 と torch 2.9 に
# 向けている。venv の torch 2.8+cu128 と Triton 3.4 がそれを拾うと壊れる（Triton が CUDA 13 の ptxas の版を読めず
# vLLM の起動で止まった。ジョブ 8536）。venv は自前の CUDA ライブラリと ptxas を持つので、ドライバ（--nv で
# /.singularity.d/libs に入る）以外は外す。
ENV_RESET='unset TRITON_PTXAS_PATH TRITON_CUOBJDUMP_PATH TRITON_NVDISASM_PATH TRITON_CUDACRT_PATH TRITON_CUDART_PATH'
ENV_RESET+=' TRITON_CUPTI_PATH CUDA_HOME TORCH_CUDA_ARCH_LIST PIP_CONSTRAINT; export LD_LIBRARY_PATH=/.singularity.d/libs'

# 使い方: run_in_container '<bash のコマンド>'
run_in_container() {
  "${CONTAINER_ENGINE}" exec --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${CACHE_DIR}:${CACHE_DIR}" \
    --bind "${TMPDIR}:${TMPDIR}" \
    --env "HF_HOME=${CACHE_DIR}/huggingface" \
    --env "UV_PYTHON_INSTALL_DIR=${PROJECT_DIR}/.uv/python" \
    --env "TMPDIR=${TMPDIR}" \
    --env "PYTHONPATH=${PROJECT_DIR}" \
    --env "NO_TORCH_COMPILE=1" \
    --env "PY_MAIN=${PROJECT_DIR}/data/envs/main/bin/python" \
    --env "MIMI_WEIGHT=${PROJECT_DIR}/checkpoints/llm-jp-moshi-v1-pp16/tokenizer-e351c8d8-checkpoint125.safetensors" \
    --pwd "${PROJECT_DIR}" \
    "${CONTAINER_IMAGE}" \
    bash -lc "set -euo pipefail; ${ENV_RESET}; $1"
}

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "CONTAINER_ENGINE=${CONTAINER_ENGINE}"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-} TASK=${SLURM_ARRAY_TASK_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
