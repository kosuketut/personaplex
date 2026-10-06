# 日本語ペルソナ学習パイロット（仮データ）

作成日: 2026-09-30 / 対象: `japersonaplex/`（このリポジトリで新規実装）

## 結論

「プロンプトを付けた系列で LoRA 学習 → upstream の推論ランタイムで生成 → プロンプト差し替えで答えが変わるか評価」の全工程が mercury 上で通った。
仮データ（テンプレート台本 + pyopenjtalk、3.2 時間）で学習すると、学習に出てこない店名・担当者名の条件でも、プロンプト中の営業時間や価格を差し替えると答えが切り替わる。

これは仕組みが動くことの確認であって、モデルとして使える水準ではない。台本は 4 種類の質問のテンプレートで、声は単一話者の合成音声の高さを変えただけ。

## 結果

flip 評価: 同じユーザ音声に対し、営業時間か価格の 1 点だけが違うプロンプト A/B で生成する。A で A の値、B で B の値を答え、相手側の値を言わなければ合格。16 ケース、店名と担当者名は学習に出てこないもの。

| モデル | 合格（テキスト出力） | 片側正答（テキスト） | 合格（音声を ASR） | 片側正答（ASR） | 検証損失 |
|---|---|---|---|---|---|
| 学習前（llm-jp-moshi-v1、8 ケースのみ） | 0/8 | 0/16 | 未実施 | 未実施 | 3.91 |
| LoRA step 100 | 13/16 | 29/32 | 9/16 | 24/32 | 0.93 |
| LoRA step 200 | 15/16 | 31/32 | 9/16 | 25/32 | 0.87 |
| LoRA step 300 | 15/16 | 31/32 | 13/16 | 29/32 | 0.90 |

- 「テキスト出力」はモデルが音声と同時に出す内部テキスト。「ASR」は出力音声を Whisper（faster-whisper medium）で書き起こして同じ基準で採点したもの。
- 採点は数の表記ゆれ（漢数字、全角、「1,100円」のようなカンマ）をそろえてから比較する。カンマを扱う前の採点では step 300 の ASR が 11/16 と低く出ていた（2026-10-01 に再集計）。
- 学習前のモデルはプロンプトの事実を使わず雑談する（例: 「1時から8時までが本当に嫌いです」）。
- 学習後の例（step 200）: プロンプト A「…営業時間は7時から22時まで」→「はい、こはる喫茶店、山本でございます。七時から二十二時まで営業しております。」、B（20時に変更）→「…七時から二十時まで営業しております。」

### 分かった弱点

- **固有名詞が言えない**: step 200 で、プロンプトの店名を A/B 両方で正しく言えたのは 10/16、担当者名は 7/16。上の例でも担当者はプロンプトでは「松本」だが「山本」と答えている。学習に出てきた名前（店名 16 種、人名 12 種）に置き換える傾向がある。名前の種類が少なすぎるのが原因と考えられる（未検証）。
- **音声はテキストより不正確**: 内部テキストで正答でも、音声では数詞が崩れる場合がある。価格は ASR で 2〜5/8 しか通らない（営業時間は 6〜8/8）。ASR の聞き取り誤りと発音の崩れは区別していない。人による聴取はしていない。
- **過学習**: 検証損失は step 200 が最良で、step 300 はテキスト側が悪化。3.2 時間のデータを 6 周している。
- **評価の規模**: 16 ケース、seed 1 種。数字の差は参考程度。
- **合成音声しか聞かせていない**: 学習・検証・flip 評価のユーザ音声はすべて pyopenjtalk。学習後のモデルが人の声の入力を扱えるかは未確認。声プロンプトも学習に使った 5 種だけで評価している。

## 実装

| ファイル | 役割 |
|---|---|
| `japersonaplex/sequence.py` | 17 ストリーム系列の組み立て。プロンプト区間は upstream の `LMGen.step_system_prompts` と同じ並び（声 → 無音 → 役割文 → 無音、ユーザ音声は 440 Hz 正弦波）。単語の時刻からテキストストリームを作る |
| `japersonaplex/prepare.py` | ステレオ wav（左 = agent、右 = user）+ 時刻付き単語 json + 役割文 + 声プロンプト wav → npz |
| `japersonaplex/data.py` | データセットと損失。プロンプト区間は損失から除外。PAD テキスト 0.3、acoustic 0.02 の重み（PersonaPlex 論文の値） |
| `japersonaplex/lora.py` | LoRA（self-attention の入出力射影、Temporal と Depformer）。保存・読込・重みへの焼き込み |
| `japersonaplex/train.py` | 単一 GPU の学習ランナー |
| `japersonaplex/merge.py` | LoRA を焼き込み、upstream がそのまま読める重みを書き出す |
| `japersonaplex/eval_flip.py`、`scoring.py` | flip 評価。生成は upstream の `LMGen` をそのまま使う |
| `japersonaplex/synth_pilot.py` | 仮データの合成 |
| `scripts/asr_flip.py` | 出力音声の ASR 採点 |
| `tests/test_sequence.py` | 系列が upstream の推論時入力と 1 フレーム単位で一致することの確認ほか |

設計上の要点:

- **学習と推論の一致**: 学習系列を、upstream の推論コードが実際にモデルへ渡す入力列と突き合わせるテストを置いた。`LMGen` は最初の 1 フレームを捨てるので、学習系列も先頭 1 フレームを落としている。焼き込んだ重みを無改造の `moshi.offline` に渡し、flip の 1 ケースで `eval_flip.py` と同一のテキスト出力になることを確認した（`moshi.server` と Web UI は未確認）。
- **LoRA の精度**: ベース重みは bf16 のまま凍結し、LoRA の差分だけ fp32 で持つ。bf16 の重みを直接小さい学習率で更新すると更新が丸めで消えるため。
- **ユーザ音声ストリーム**: 既定では損失に入れない（`--user-weight 0`）。推論ではユーザ音声は入力で上書きされるので生成には不要。PersonaPlex と同じく補助損失として学習するなら重みを指定する。
- **テキストの規約**: ベースモデルの出力に合わせ、単語の直前フレームに EPAD、語頭の「▁」は付けない。

学習設定: LoRA rank 32（学習対象 3,810 万パラメータ）、AdamW、学習率 1e-4、cosine、batch 2 × 勾配蓄積 4、300 step。RTX 6000 Ada 1 枚で約 10 秒/step、ピーク 32.5 GiB。

## 学習したモデルを動かす

LoRA を焼き込んだ重みは upstream のスクリプトにそのまま渡せる。

```bash
.venv/bin/python -m japersonaplex.merge --moshi-weight $C/model.safetensors \
  --lora runs/pilot_v2/lora_step200.safetensors --out checkpoints/pilot_v2_step200/model.safetensors
cp $C/tokenizer_spm_32k_3.model checkpoints/pilot_v2_step200/

.venv/bin/python -m moshi.offline \
  --moshi-weight checkpoints/pilot_v2_step200/model.safetensors \
  --tokenizer checkpoints/pilot_v2_step200/tokenizer_spm_32k_3.model \
  --voice-prompt voice1_0.wav --voice-prompt-dir data/pilot/voices \
  --text-prompt "あなたはこはる喫茶店という喫茶店で働いています。名前は松本です。情報：定休日は月曜日。ナポリタンは1050円。営業時間は7時から22時まで。" \
  --input-wav data/pilot/flip/flip_014.wav --seed 1234 \
  --output-wav outputs/out.wav --output-text outputs/out.json
```

`moshi.server` も同じ `--moshi-weight` と `--tokenizer` を受け取る（未実行）。

## 再現手順

```bash
export NO_TORCH_COMPILE=1   # mercury の環境では torch.compile が失敗するため
C=checkpoints/llm-jp-moshi-v1-pp16   # scripts/convert_moshi_to_personaplex.py で作成

.venv/bin/python -m pytest -q tests
.venv/bin/python -m japersonaplex.synth_pilot --out data/pilot
for s in train heldout; do
  .venv/bin/python -m japersonaplex.prepare --manifest data/pilot/$s.jsonl --out data/pilot/npz/$s \
    --tokenizer $C/tokenizer_spm_32k_3.model
done
.venv/bin/python -m japersonaplex.train --moshi-weight $C/model.safetensors \
  --train data/pilot/npz/train --valid data/pilot/npz/heldout --out runs/pilot_v2 --steps 300 --save-every 100
.venv/bin/python -m japersonaplex.eval_flip --flip data/pilot/flip.jsonl --moshi-weight $C/model.safetensors \
  --tokenizer $C/tokenizer_spm_32k_3.model --lora runs/pilot_v2/lora_step200.safetensors --out runs/pilot_v2/flip_step200
uv run --no-project --python 3.10 --with faster-whisper python scripts/asr_flip.py runs/pilot_v2/flip_step200
```

所要時間の目安（mercury）: 合成 11 分、トークン化 30 分（4 プロセス並列）、学習 55 分、評価 15 分/チェックポイント。
成果物は `data/`、`runs/`、`checkpoints/`、`outputs/`（いずれも git 管理外）。

## Seiran

用意したもの（**いずれも未投入で、Seiran 上では一度も動かしていない**。lint は通過）:

| ファイル | 内容 |
|---|---|
| `slurm/setup_env.sbatch` | venv 作成、依存と moshi の導入、テスト、ベースモデル（`llm-jp/llm-jp-moshi-v1`）の取得と PersonaPlex 形式への変換 |
| `slurm/train_lora.sbatch` | 1 GPU の LoRA 学習と flip 評価 |
| `slurm/train_full.sbatch` | 1 ノード 8 GPU の全パラメータ学習（`japersonaplex/train_full.py`、FSDP）。同じ出力先で再投入すると自動で再開する |

- 既存 SIF（`physicsnemo_25.11_user.sif`）の PyTorch を使う venv に、`moshi` を `--no-deps` で入れる構成。upstream は `torch<2.5` を指定しているが、B200 には新しい PyTorch が必要なため。upstream のコードが新しい PyTorch で動くかは未確認で、`setup_env.sbatch` のテストがその確認になる。
- ベースモデルと Mimi は公開の `llm-jp/llm-jp-moshi-v1` から取得する（`nvidia/personaplex-7b-v1` は承認制で、Seiran に HF トークンが無い）。
- 置き場所とデータ・チェックポイントの保存先は未決定。`/home` は 93% 使用。

### 全パラメータ学習（`train_full.py`）

- FSDP（FULL_SHARD）で、重みと Adam 状態は fp32、計算は bf16。Temporal Transformer の各層は activation checkpointing。学習率は PersonaPlex と同じ Temporal 2e-6、Depformer 4e-6。ユーザ音声ストリームも既定で学習する（`--user-weight 1.0`）。
- 保存は推論用の bf16 重み（`model_step{N}.safetensors`、upstream がそのまま読める）と、再開用の rank ごとのシャード（`resume_step{N}/`）。再開は同じ GPU 数でのみ可能。
- `--stop-at N` で、学習率スケジュールを変えずに N ステップ目で保存して終了できる。Slurm の時間制限内でジョブをつなぐ用途。
- mercury の 2 GPU で小さいモデルを使って確認した（`tests/dist_smoke.py`）。6 ステップ設定で 3 ステップ目に止めて再開した結果は、中断なしの 6 ステップと bf16 で完全一致した。
- **実サイズのモデルでは動かしていない。** 8.4B を fp32 + Adam で学習するには約 134 GB が要り、mercury（48 GB × 2）には載らないため。B200 でのメモリ使用量と速度は未測定。

LoRA 学習（`train.py`）にも再開（`--resume`）と `--stop-at` を付けた。こちらはデータの順序までは復元しない。

## 次にやること

1. データを本物にする。LLM による台本（名前・業種・質問の種類を大幅に増やす）、多話者のゼロショット TTS、強制アライメントによる単語時刻。固有名詞の弱点はここで解消を狙う。
2. 割り込み・相槌・発話の重なりを含む対話。仮データは交互に話すだけ。
3. Seiran での全パラメータ学習（fp32、複数 GPU）とユーザ音声ストリームの学習。
4. 評価の拡充。声プロンプトへの追従（話者類似度）、自由な雑談での自然さ、人による聴取。今回は声の制御を評価していない。
