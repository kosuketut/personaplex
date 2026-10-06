#!/usr/bin/env bash
# Seiran 上にデータ生成用の環境とモデルを用意する。インターネットが要るので、ログインノードで実行する
# （計算ノードから huggingface.co / pypi.org に出られるかは未確認）。GPU は使わない。
#
#   cd /home/projects/aitc/yano/japersonaplex && bash scripts/seiran_setup_envs.sh
#
# 作るもの（すべてプロジェクトの下。uv の Python とキャッシュも含める。コンテナの中からも同じパスで使える）:
#   data/envs/tts    Qwen3-TTS（torch 2.8.0+cu128、qwen_tts はリポジトリの写し data/envs/src/Qwen3-TTS から）
#   data/envs/align  アライナと whisper（torch 2.8.0+cu128、qwen-asr、faster-whisper）
#   data/envs/llm    台本 LLM（vLLM 0.11.0、transformers 4.57.1）
#   data/envs/main   組み立て・prepare・学習（torch 2.8.0+cu128、moshi を --no-deps で）
#   .cache/huggingface  モデル。checkpoints/llm-jp-moshi-v1-pp16 にベースモデルを変換して置く
# mercury では torch 2.4.1 の .venv を使っているが、B200（sm_100）には CUDA 12.8 版の torch 2.7 以降が要る。
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export UV_CACHE_DIR="${PROJECT_DIR}/.uv/cache" UV_PYTHON_INSTALL_DIR="${PROJECT_DIR}/.uv/python"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export PATH="${PROJECT_DIR}/.uv/bin:${PATH}"
CU128=https://download.pytorch.org/whl/cu128
echo "PROJECT_DIR=${PROJECT_DIR} HF_HOME=${HF_HOME}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${PROJECT_DIR}/.uv/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

mkenv() {  # mkenv <name> <python> <packages...>
  local d="data/envs/$1" py=$2; shift 2
  if [[ ! -x "$d/bin/python" ]]; then
    uv venv "$d" --python "$py" -q
  fi
  uv pip install -q -p "$d/bin/python" torch==2.8.0 torchaudio==2.8.0 --index-url "${CU128}"
  if [[ $# -gt 0 ]]; then
    uv pip install -q -p "$d/bin/python" "$@" --index-strategy unsafe-best-match --extra-index-url "${CU128}"
  fi
  "$d/bin/python" -c "import pyopenjtalk; pyopenjtalk.g2p('あ')" 2>/dev/null || true  # 辞書を取得しておく
}

set -x
mkenv tts 3.11 ./data/envs/src/Qwen3-TTS transformers==4.57.3 pyloudnorm pyopenjtalk soundfile scipy
mkenv align 3.11 qwen-asr pyopenjtalk soundfile scipy pillow faster-whisper nvidia-cublas-cu12 \
  "nvidia-cudnn-cu12==9.10.2.21"
if [[ ! -x data/envs/llm/bin/python ]]; then uv venv data/envs/llm --python 3.11 -q; fi
VIRTUAL_ENV="${PROJECT_DIR}/data/envs/llm" uv pip install -q vllm==0.11.0 --extra-index-url "${CU128}" \
  --index-strategy unsafe-best-match
VIRTUAL_ENV="${PROJECT_DIR}/data/envs/llm" uv pip install -q transformers==4.57.1 pyopenjtalk sentencepiece
data/envs/llm/bin/python -c "import pyopenjtalk; pyopenjtalk.g2p('あ')"
mkenv main 3.10 -r slurm/requirements-seiran.txt
uv pip install -q -p data/envs/main/bin/python --no-deps -e moshi/
set +x

# モデル（台本 LLM、TTS、アライナ、whisper、ひらがな CTC、ベースモデル）
data/envs/main/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
             "Qwen/Qwen3-ForcedAligner-0.6B", "Systran/faster-whisper-medium",
             "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "vumichien/wav2vec2-large-xlsr-japanese-hiragana"]:
    print(repo, snapshot_download(repo), flush=True)
PY
BASE_DIR=checkpoints/llm-jp-moshi-v1-pp16
mkdir -p "${BASE_DIR}"
if [[ ! -f "${BASE_DIR}/model.safetensors" ]]; then
  data/envs/main/bin/python - <<PY
import shutil
from huggingface_hub import hf_hub_download
for name in ("tokenizer_spm_32k_3.model", "tokenizer-e351c8d8-checkpoint125.safetensors"):
    shutil.copy(hf_hub_download("llm-jp/llm-jp-moshi-v1", name), "${BASE_DIR}/" + name)
print(hf_hub_download("llm-jp/llm-jp-moshi-v1", "model.safetensors"), file=open("${BASE_DIR}/source_path.txt", "w"))
PY
  data/envs/main/bin/python scripts/convert_moshi_to_personaplex.py --src "$(cat "${BASE_DIR}/source_path.txt")" \
    --dst "${BASE_DIR}/model.safetensors"
fi
# 名前の辞書（IPAdic）
mkdir -p data/dict/ipadic
for f in Noun.name.csv Noun.place.csv COPYING; do
  [[ -s "data/dict/ipadic/$f" ]] || curl -sfL -o "data/dict/ipadic/$f" "https://raw.githubusercontent.com/taku910/mecab/master/mecab-ipadic/$f"
done
# GPU を使わないテスト
PYTHONPATH=. data/envs/main/bin/python -m pytest -q tests/test_datagen_units.py tests/test_datagen_assemble.py
du -sh data/envs .uv "${HF_HOME}" "${BASE_DIR}"
echo "setup done"
