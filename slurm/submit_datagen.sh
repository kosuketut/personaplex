#!/bin/bash
# 本データの生成を Seiran に投入する: 台本 -> シャードごとの音声と npz（配列ジョブ）-> まとめ。
#
#   RUN=pilot40 bash slurm/submit_datagen.sh
#
# 前提: scripts/seiran_setup_envs.sh を済ませ、slurm/datagen_check.sbatch が通っていること。
# MAIN_SHARDS（既定 16）と QA_SHARDS（既定 4）で GPU の数を決める。評価セット 3 種は各 1 GPU。
# SKIP_SCRIPTS=1 なら台本のジョブを飛ばし、data/datagen/${RUN}/*/scripts.jsonl をそのまま使う。
# KINDS で作る種類を絞れる（例: KINDS="main" で QA と評価セットを作らない）。
set -euo pipefail

RUN="${RUN:-pilot40}"
MAIN_SHARDS="${MAIN_SHARDS:-16}"
QA_SHARDS="${QA_SHARDS:-4}"
KINDS="${KINDS:-main qa h_name h_domain h_voice}"
mkdir -p logs outputs

dep=()
if [[ "${SKIP_SCRIPTS:-0}" != 1 ]]; then
  scripts_id="$(sbatch --parsable --export=ALL,RUN="${RUN}" slurm/datagen_scripts.sbatch)"
  echo "Submitted scripts: ${scripts_id}"
  dep=(--dependency="afterok:${scripts_id}")
fi

shard_ids=()
for kind in ${KINDS}; do
  case "${kind}" in
    main) n="${MAIN_SHARDS}" ;;
    qa) n="${QA_SHARDS}" ;;
    *) n=1 ;;
  esac
  id="$(sbatch --parsable "${dep[@]}" --array="0-$((n - 1))" --export=ALL,RUN="${RUN}",KIND="${kind}" \
    slurm/datagen_shard.sbatch)"
  echo "Submitted shards ${kind} x${n}: ${id}"
  shard_ids+=("${id}")
done

# 一部のシャードが失敗してもまとめは走らせる（merge.py は出力のあるシャードだけを集める）
merge_id="$(sbatch --parsable --dependency="afterany:$(IFS=:; echo "${shard_ids[*]}")" \
  --export=ALL,RUN="${RUN}",KINDS="${KINDS}" slurm/datagen_merge.sbatch)"
echo "Submitted merge after ${shard_ids[*]}: ${merge_id}"
