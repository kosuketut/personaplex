#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-mkflip
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# flip 評価のセット v2（事実を直接聞く質問を user の声で TTS）を作り、質問の音声を whisper（CPU）で確かめる。
#   SRC=data/datagen/pilot40 OUT=data/datagen/pilot40/flip40v2 sbatch slurm/make_flip.sbatch
# SRC の h_name・h_domain・h_voice（train.jsonl）と main の valid.jsonl から、セットごとに N 件。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
SRC="${SRC:-data/datagen/pilot40}"
OUT="${OUT:-${SRC}/flip40v2}"
N="${N:-40}"
SEED="${SEED:-0}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv。コンテナの CUDA 向けの変数を外す
echo "SRC=${SRC} OUT=${OUT} N=${N} SEED=${SEED}"
test ! -e "${OUT}/flip.jsonl"  # 評価セットは作り直さない（比べる基準が変わるため）。作り直すなら手で消す

run_in_container "
data/envs/tts/bin/python -m japersonaplex.datagen.make_flip --direct --out '${OUT}' --n ${N} --seed ${SEED} \
  --set h_name='${SRC}/h_name' --set h_domain='${SRC}/h_domain' --set h_voice='${SRC}/h_voice' \
  --set valid='${SRC}/main:valid.jsonl'
data/envs/align/bin/python -m japersonaplex.datagen.make_flip --check --out '${OUT}'
"
