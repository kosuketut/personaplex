#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-dgcheck
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# データ生成の動作確認（B200 で全工程が動くか）。scripts/seiran_setup_envs.sh の後に投入する。
# 0) 計算ノードから外に出られるか 1) 各環境の torch が GPU を認識するか 2) 台本 LLM で 6 本
# 3) その 6 本で TTS から npz まで 4) まとめと統計
#   sbatch slurm/datagen_check.sbatch
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
OUT="data/datagen/check"  # 毎回消して作り直すので固定（変数で別の場所を渡せないようにしてある）

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv（GPU を使う）
echo "OUT=${OUT}"
for url in https://huggingface.co https://pypi.org/simple/; do  # 遮断されていても止めない（結果を見るだけ）
  echo "outbound ${url}: $(curl -s -m 10 -o /dev/null -w '%{http_code}' "${url}" || echo blocked)"
done

run_in_container "
test -s checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model
for env in tts align llm main; do
  echo \"== \${env}\"
  data/envs/\${env}/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
done
rm -rf data/datagen/check && mkdir -p data/datagen/check
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 data/envs/llm/bin/python -m japersonaplex.datagen.scripts_llm \
  --n 6 --scenario mix --seed 9 --out '${OUT}/scripts.jsonl' --only-passing --gpu-mem 0.5
PY_MAIN=\${PY_MAIN} TTS_BATCH=64 NPZ_KANJI=1 bash scripts/datagen.sh '${OUT}/shard_0' '${OUT}/scripts.jsonl'
\${PY_MAIN} -m japersonaplex.datagen.merge --root '${OUT}' --valid-frac 0
# 数字の表記の A/B: npz と npz_kanji は音声の符号が同じで、テキストの行だけが違うこと
\${PY_MAIN} -c \"
import glob, numpy as np, os
for p in sorted(glob.glob('${OUT}/shard_0/npz/*.npz')):
    a, b = np.load(p), np.load(p.replace('/npz/', '/npz_kanji/'))
    same_audio = a['codes'].shape == b['codes'].shape and (a['codes'][1:] == b['codes'][1:]).all()
    print(os.path.basename(p), 'audio same', same_audio, 'text rows differ at', int((a['codes'][0] != b['codes'][0]).sum()), 'frames')
\"
\${PY_MAIN} -m japersonaplex.datagen.stats --out '${OUT}'
cat '${OUT}/shard_0/timing.log'
"
