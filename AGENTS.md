# AGENTS

Seiran（`/home/projects/aitc/yano/japersonaplex`）での運用メモ。ワークスペース直下の `../AGENTS.md` の共通ルールに従う。
詳しい手順と品質検査は `docs/ja-datagen.md`、作業の進み具合は `TASKS.md`。

## 環境

- `bash scripts/seiran_setup_envs.sh` をログインノードで実行する（インターネットが要る）。uv の venv 4 つ（`data/envs/{tts,align,llm,main}`、
  torch 2.8.0+cu128）、モデル（`.cache/huggingface`）、ベースモデル（`checkpoints/llm-jp-moshi-v1-pp16`）、IPAdic（`data/dict/ipadic`）を作る。
- ジョブはコンテナ（既定 `../container/physicsnemo_25.11_user.sif`）の中で動かし、python は venv のものを使う（`slurm/datagen_common.sh`）。

## Slurm（アカウント `aitc`、ログは `logs/%x_%j.{out,err}`）

- 動作確認（GPU 1 枚）: `sbatch slurm/datagen_check.sbatch`。出力 `data/datagen/check`（毎回作り直す）。
- 本データ: `RUN=pilot40 bash slurm/submit_datagen.sh`。台本（1 GPU）→ 種類ごとの配列ジョブ（main 16、qa 4、h_name・h_domain・h_voice 各 1）→ まとめ。
  - 一部だけ: `KINDS="main"`、台本を作り直さない: `SKIP_SCRIPTS=1`。シャードは同じ配列番号で投げ直せば続きから進む。
- LoRA 学習（GPU 1 枚）: `RUN_NAME=pilot40_lora_v1 STEPS=1200 FLIP=none TRAIN_ARGS="--save-every 200 --eval-every 100" sbatch slurm/train_lora.sbatch`。
  学習データの既定は `data/datagen/pilot40/main`（QA は入れない）。評価専用セットの損失は `outputs/<RUN_NAME>/metrics.jsonl` の `valid_h_*`。
- 全パラメータ学習: `RUN_NAME=full_v1 NPROC=7 STEPS=500 TRAIN_ARGS="--user-weight 0 --eval-every 25 --save-every 100" sbatch --gres=gpu:7 slurm/train_full.sbatch`
  （既定は 8 GPU。空きに合わせて NPROC と --gres をそろえて変える。再開は同じ GPU 数のときだけ）。B200 7 枚で約 1 秒/step、
  CPU メモリ 332 GB、保存は推論用の重み 17 GB と再開用（最新だけ）約 100 GB。評価は `FULL=1` を付けて `eval_flip.sbatch`。
  `train_full.py` の `--user-weight` の既定は 1.0（user 側の損失あり）で、LoRA（既定 0）とそろえるには 0 を明示する。代表（full_mix）のコマンドは `docs/ja-pilot40-train.md` の「合わせたデータでの全パラメータ学習」。
- flip 評価: `RUN_NAME=... EVAL_STEPS="0 400 800 1200" sbatch --array=0-3 slurm/eval_flip.sbatch`（0 は LoRA なし）。
  評価セットは `data/pilot/flip.jsonl`（pyopenjtalk の声）と `data/datagen/pilot40/flip40v2/flip.jsonl`（事実を直接聞く質問を TTS。
  `slurm/make_flip.sbatch` で作る。v1 の `flip40` は質問が間接的で雑音が多い）。
- データ v3（10/04 の既定の作り方）: `RUN=pilot40v3 SEED=3000 KINDS="main" MAIN_SHARDS=32 N_SERVICE=7000 N_CASUAL=1700 N_QA=0 N_EVAL=0
  SERVICE_ARGS="--fact-items sample --ask-facts 2 --ask-first 0.5" bash slurm/submit_datagen.sh`（66 h、GPU 8 枚で約 2 時間）。
  評価は pilot40 の flip40 v2 を使い続ける。質問の位置を変えた版は `flip40v2u2`（`make_flip.py --from ... --before-user 2`）、
  質問の形を変えた版は `flip40v3_paraphrase`・`flip40v3_situational`（`sbatch slurm/make_flip_forms.sbatch`）。形の違う質問で順位が変わるので、4 つの形で見る。
  `flip_table.py` は flip40 v2 の不備のある 5 問を外して数える（109 問）。
- データ v4（10/04〜。user の聞き方を 4 つの形に混ぜる）: v3 の引数に `--ask-forms name:0.2,paraphrase:0.25,situational:0.3,indirect:0.25 --p-hours 0.55`
  を足す（コマンドは `docs/ja-datagen.md` の「データ v4」）。v4 だけだと項目名で直接聞く形が落ちるので、学習は v3・v4・v4b を合わせた
  `data/datagen/pilot40mix/main/npz_train`（リンク、191 h）で行う。**代表は全パラメータ学習の full_mix の step 1200**（同じデータ、`--user-weight 0`。`docs/ja-pilot40-train.md` の「合わせたデータでの全パラメータ学習」。LoRA なら pilot40mix の step 4800）。
  評価は seed 3 つ（`EVAL_ARGS="--seed 1" OUT_SUFFIX=_seed1` など）で回し、片側の正答で比べる（seed だけで 4〜15 動く）。
- 数字の表記の A/B: `NPZ_KANJI=1` を付けて `submit_datagen.sh` を流すと、同じ音声から漢数字版の `npz_kanji_{train,valid}` もできる。
  表は `python scripts/flip_table.py outputs/<RUN_NAME>/eval_*`。
- sbatch はプロジェクト直下から投入する（`PROJECT_DIR` は `SLURM_SUBMIT_DIR` から決まる）。

## 成果物

- `data/datagen/<RUN>/<種類>/`: `scripts.jsonl`、`shard_*/`（音声・アライメント・npz）、`{train,valid}.jsonl`、`npz_{train,valid}/`、
  `qc_summary.json`、`stats.json`。`train.py --train/--valid` にそのまま渡せる。
- `data/`、`.cache/`、`.uv/`、`checkpoints/`、`logs/`、`outputs/`、`tmp/` は Git に入れない。
