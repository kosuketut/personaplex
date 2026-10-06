#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-dgshard
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# 台本を SHARDS 個に分け、配列ジョブの 1 タスク（GPU 1 枚）で 1 シャードを音声から npz まで作る（scripts/datagen.sh）。
#   RUN=pilot40 KIND=main sbatch --array=0-15 slurm/datagen_shard.sbatch
# KIND は data/datagen/${RUN}/ の下のディレクトリ名（main、qa、h_name、h_domain、h_voice）。
# KIND=h_voice のときは評価専用の声だけを使う。SHARDS は既定で配列の大きさ（SLURM_ARRAY_TASK_COUNT）。
# 出力: data/datagen/${RUN}/${KIND}/shard_${SLURM_ARRAY_TASK_ID}/。途中で止まっても、同じ配列番号で
# 投げ直せば TTS 済みの発話は作り直さずに続きから進む（datagen.sh の作り）。
# 配列の各タスクは別のジョブ id を持つので、ログ名は他のジョブと同じ logs/%x_%j.* で重ならない。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN="${RUN:-pilot40}"
KIND="${KIND:-main}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARDS="${SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
VOICE_SPLIT="${VOICE_SPLIT:-$([[ "${KIND}" == h_voice ]] && echo heldout || echo train)}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
TTS_BATCH="${TTS_BATCH:-128}"
TOKEN_BUDGET="${TOKEN_BUDGET:-12000}"
NPZ_KANJI="${NPZ_KANJI:-0}"  # 1 なら数字を漢数字にした npz も作る（数字の表記の A/B 用、scripts/datagen.sh）

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv（GPU を使う）
D="data/datagen/${RUN}/${KIND}"
OUT="${D}/shard_${TASK}"
echo "RUN=${RUN} KIND=${KIND} TASK=${TASK}/${SHARDS} VOICE_SPLIT=${VOICE_SPLIT} OUT=${OUT} NPZ_KANJI=${NPZ_KANJI}"
test -s "${D}/scripts.jsonl"
mkdir -p "${OUT}"
awk -v n="${SHARDS}" -v i="${TASK}" '(NR - 1) % n == i' "${D}/scripts.jsonl" > "${OUT}/scripts.jsonl"
wc -l "${OUT}/scripts.jsonl"

run_in_container "
VOICE_SPLIT='${VOICE_SPLIT}' MAX_ATTEMPTS=${MAX_ATTEMPTS} TTS_BATCH=${TTS_BATCH} TOKEN_BUDGET=${TOKEN_BUDGET} NPZ_KANJI=${NPZ_KANJI} \
  bash scripts/datagen.sh '${OUT}' '${OUT}/scripts.jsonl'
"
