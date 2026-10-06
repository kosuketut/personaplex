#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-dgmerge
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# シャードの出力を KIND ごとにまとめる（train/valid の分割、npz へのリンク、品質検査の集計、ターンテイキングの統計）。
#   RUN=pilot40 sbatch slurm/datagen_merge.sbatch
# GPU は使わないが、他のジョブと同じくコンテナの中で動かす。
# HALF_SHARDS=16 なら、main のシャード 0〜15 の npz だけへのリンクを npz_train_half に張る（シャードは台本を interleave で
# 分けるので、量を半分にした部分集合になる。データ v3 で同じ量どうしを比べるのに使った）。
# COMBINE_WITH="data/datagen/pilot40v4/main/npz_train ..." なら、main の npz_train とそれらを合わせたリンクを npz_train_plus に張る
# （リンク名の頭に RUN 名を付ける。同じ作り方で seed を変えたデータを足して学習するのに使う）。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN="${RUN:-pilot40}"
KINDS="${KINDS:-main qa h_name h_domain h_voice}"
HALF_SHARDS="${HALF_SHARDS:-}"
COMBINE_WITH="${COMBINE_WITH:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh
echo "RUN=${RUN} KINDS=${KINDS} HALF_SHARDS=${HALF_SHARDS} COMBINE_WITH=${COMBINE_WITH}"

run_in_container "
for kind in ${KINDS}; do
  d=data/datagen/${RUN}/\${kind}
  ls -d \"\${d}\"/shard_* >/dev/null 2>&1 || { echo \"skip \${kind}: no shards\"; continue; }
  frac=0.01; [[ \${kind} == h_* ]] && frac=0  # 評価セットは分けない
  echo \"== \${kind}\"
  \${PY_MAIN} -m japersonaplex.datagen.merge --root \"\${d}\" --valid-frac \${frac}
  \${PY_MAIN} -m japersonaplex.datagen.stats --out \"\${d}\" > /dev/null
  \${PY_MAIN} -c \"import json; s=json.load(open('\${d}/stats.json')); print({k: s[k] for k in ('hours', 'per_minute', 'backchannels_per_min')})\"
done
"

if [[ -n "${HALF_SHARDS}" ]]; then
  d="data/datagen/${RUN}/main"
  rm -rf "${d}/npz_train_half"
  mkdir -p "${d}/npz_train_half"
  for i in $(seq 0 $((HALF_SHARDS - 1))); do
    find "${d}/npz_train" -maxdepth 1 -name "shard_${i}__*.npz" -exec cp -P {} "${d}/npz_train_half/" \;
  done
  echo "npz_train_half: $(ls "${d}/npz_train_half" | wc -l) files (shards 0-$((HALF_SHARDS - 1)))"
fi

if [[ -n "${COMBINE_WITH}" ]]; then
  d="data/datagen/${RUN}/main"
  out="${d}/npz_train_plus"
  rm -rf "${out}"
  mkdir -p "${out}"
  for src in "${d}/npz_train" ${COMBINE_WITH}; do
    tag="$(echo "${src}" | cut -d/ -f3)"  # data/datagen/<RUN>/main/npz_train -> <RUN>
    test -d "${src}"
    for f in "${src}"/*.npz; do
      ln -s "$(realpath --relative-to="${out}" "$(readlink -f "${f}")")" "${out}/${tag}__$(basename "${f}")"
    done
  done
  echo "npz_train_plus: $(ls "${out}" | wc -l) files from ${d}/npz_train ${COMBINE_WITH}"
fi
