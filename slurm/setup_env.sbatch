#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-setup
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# 環境構築と検証: SIF の PyTorch を使う venv を作り、依存と moshi を入れ、ベースモデルを取得・変換し、
# テストと学習スモークテストを実行する。SIF の PyTorch を置き換えないよう moshi は --no-deps で入れる
# （upstream は torch<2.5 を指定しているが B200 には新しい PyTorch が必要）。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-seiran}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs "${DATA_DIR}" "${CKPT_DIR}" "${CACHE_DIR}"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "Container image not found: ${CONTAINER_IMAGE}" >&2
  echo "Override with: CONTAINER_IMAGE=/path/to/image.sif sbatch $0" >&2
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

run_in_container() {
  "${CONTAINER_ENGINE}" exec --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${CKPT_DIR}:${CKPT_DIR}" \
    --bind "${CACHE_DIR}:${CACHE_DIR}" \
    --env "HF_HOME=${CACHE_DIR}/huggingface" \
    --pwd "${PROJECT_DIR}" \
    "${CONTAINER_IMAGE}" \
    bash -lc "$1"
}

BASE_REPO="${BASE_REPO:-llm-jp/llm-jp-moshi-v1}"
BASE_DIR="${CKPT_DIR}/llm-jp-moshi-v1-pp16"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "CONTAINER_ENGINE=${CONTAINER_ENGINE}"
echo "BASE_REPO=${BASE_REPO}"

run_in_container "
set -euo pipefail
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
[[ -d '${VENV_DIR}' ]] || python -m venv --system-site-packages '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
python -m pip install -r slurm/requirements-seiran.txt
python -m pip install --no-deps -e moshi/
python -m pytest -q tests
mkdir -p '${BASE_DIR}'
python - <<'PY'
import shutil
from huggingface_hub import hf_hub_download
for name in ('model.safetensors', 'tokenizer_spm_32k_3.model', 'tokenizer-e351c8d8-checkpoint125.safetensors'):
    path = hf_hub_download('${BASE_REPO}', name)
    if name != 'model.safetensors':
        shutil.copy(path, '${BASE_DIR}/' + name)
    else:
        print(path, file=open('${BASE_DIR}/source_path.txt', 'w'))
PY
[[ -f '${BASE_DIR}/model.safetensors' ]] || python scripts/convert_moshi_to_personaplex.py \
  --src \"\$(cat '${BASE_DIR}/source_path.txt')\" --dst '${BASE_DIR}/model.safetensors'
"
