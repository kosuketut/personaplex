#!/bin/bash
#SBATCH -A aitc
#SBATCH -J japlex-dgscripts
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

# 台本 LLM で、学習用（サービス・雑談・QA）と評価用（H-name、H-domain、H-voice）の台本を作る。
#   RUN=pilot40 sbatch slurm/datagen_scripts.sbatch
# 出力: data/datagen/${RUN}/{main,qa,h_name,h_domain,h_voice}/scripts.jsonl（main はサービスと雑談）。
# QA は事実の扱いが未決のため、main と分けて作る（使うかどうかを後で決められるように）。
# 本数の既定は 40 時間（docs/ja-data-spec.md §1.2: サービス 26 h、雑談 8 h、QA 6 h）を、M2 の実測
# （自動検査の合格率 サービス 73%・雑談 75%・QA 53%、組み立てで残る割合 90%、平均 57.5 秒）から逆算した値。
# 0 にした種類は作らない（N_EVAL=0 なら評価用 3 種を作らない）。SERVICE_ARGS はサービス台本だけに足す引数
# （v3: SERVICE_ARGS="--fact-items sample --ask-facts 2 --ask-first 0.5"）。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/home/projects/aitc/yano/container/physicsnemo_25.11_user.sif}"
RUN="${RUN:-pilot40}"
SEED="${SEED:-1000}"
N_SERVICE="${N_SERVICE:-2600}"
N_CASUAL="${N_CASUAL:-750}"
N_QA="${N_QA:-900}"
N_EVAL="${N_EVAL:-150}"
SERVICE_ARGS="${SERVICE_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs
source slurm/datagen_common.sh  # run_in_container: singularity/apptainer exec --nv（GPU を使う）
D="data/datagen/${RUN}"
echo "RUN=${RUN} SEED=${SEED} N_SERVICE=${N_SERVICE} N_CASUAL=${N_CASUAL} N_QA=${N_QA} N_EVAL=${N_EVAL} SERVICE_ARGS=${SERVICE_ARGS}"

run_in_container "
mkdir -p '${D}'/{main,qa,h_name,h_domain,h_voice}
gen() {  # gen <out> <scenario> <n> <seed> [追加の引数]
  local out=\$1 scen=\$2 n=\$3 seed=\$4; shift 4
  HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 data/envs/llm/bin/python -m japersonaplex.datagen.scripts_llm \
    --n \"\${n}\" --scenario \"\${scen}\" --seed \"\${seed}\" --out \"\${out}\" --only-passing \
    --stats \"\${out%.jsonl}_stats.json\" --gpu-mem 0.85 --max-num-seqs 256 \"\$@\"
}
gen '${D}/main/service.jsonl' service ${N_SERVICE} $((SEED + 1)) ${SERVICE_ARGS}
if [[ ${N_CASUAL} -gt 0 ]]; then gen '${D}/main/casual.jsonl' casual ${N_CASUAL} $((SEED + 2)); else : > '${D}/main/casual.jsonl'; fi
cat '${D}/main/service.jsonl' '${D}/main/casual.jsonl' > '${D}/main/scripts.jsonl'
if [[ ${N_QA} -gt 0 ]]; then gen '${D}/qa/scripts.jsonl' qa ${N_QA} $((SEED + 3)); fi
if [[ ${N_EVAL} -gt 0 ]]; then
  gen '${D}/h_name/scripts.jsonl' service ${N_EVAL} $((SEED + 4)) --split heldout
  gen '${D}/h_domain/scripts.jsonl' service ${N_EVAL} $((SEED + 5)) --domains heldout
  gen '${D}/h_voice/scripts.jsonl' mix ${N_EVAL} $((SEED + 6))
fi
wc -l '${D}'/*/scripts.jsonl || true
"
