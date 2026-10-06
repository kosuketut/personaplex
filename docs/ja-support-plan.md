# PersonaPlex 日本語対応: 現状調査と方針

作成日: 2026-09-30 / 対象: upstream `3428dfd`（NVIDIA/personaplex）、重み `nvidia/personaplex-7b-v1`

## 結論

PersonaPlex は日本語に対応しておらず、コード修正だけでは対応できない。モデルの再学習が必要。

推奨方針は **「日本語 Moshi（`llm-jp/llm-jp-moshi-v1`、Apache-2.0）を土台に、PersonaPlex 方式の役割・声プロンプト学習を日本語の合成対話データで行う」**。
PersonaPlex の英語重みに日本語を教える方向は、下記の実測によりテキストトークナイザの時点で成立しない。

## 実測したこと

再現手順は末尾。出力は `outputs/baseline/`（git 管理外）。

### 1. 英語トークナイザは日本語をフレームレート内に収められない

Moshi 系はテキストを 1 フレーム（80 ms、12.5 Hz）に 1 トークンずつ出す。PersonaPlex の SentencePiece（英語 32k）は日本語を UTF-8 の byte fallback に分解する。

| 発話（pyopenjtalk 合成、約 6.5 モーラ/秒） | PersonaPlex 英語 SPM | llm-jp 日本語 SPM |
|---|---|---|
| こんにちは。ちょっと相談してもいいですか。 | 15.6 トークン/秒 | 2.8 |
| 最近、夜なかなか眠れなくて… | 14.3 | 2.8 |
| なるほど。寝る前にスマホを… | 15.8 | 4.3 |

上限は 12.5 トークン/秒。ゆっくりめの合成音声でも超えるので、英語トークナイザのまま日本語を話させる学習はできない。トークナイザの差し替えが前提になる。

### 2. 重みの互換性

`personaplex-7b-v1` と `llm-jp-moshi-v1` の safetensors ヘッダを比較した。

- 共通 355 テンソルのうち形状が違うのは depformer の `self_attn` 12 個だけ（8 → 16 コードブック分）。
- PersonaPlex にだけあるのは depformer のユーザ音声側（`gating/linears/depformer_in/depformer_emb` の 8〜15）120 個。
- テキスト埋め込みは両者とも 32001 × 4096、語彙サイズも 32000 で同じ。中身（語彙）は別物。
- Mimi コーデックは同一ファイル。

`scripts/convert_moshi_to_personaplex.py` で llm-jp の重みを PersonaPlex 形式（dep_q=16）に変換できる（拡張 12、コピー 119、ゼロ初期化 1）。
upstream のローダは `depformer_emb.7.weight` のコピー元が無いと meta tensor のまま落ちるので、変換スクリプト側でゼロ初期化している。

### 3. ベースライン（入力: `assets/test_ja/input_ja.wav`、日本語の睡眠相談 3 発話、40 秒）

| 構成 | 出力テキスト（抜粋） | 判定 |
|---|---|---|
| PersonaPlex + 日本語プロンプト | `Hello, this is Tina. … I'm sorry, I can't really understand you.` | 英語で応答、日本語を理解しない |
| PersonaPlex + 既定の英語プロンプト | レモンハーブソースの作り方を英語で話し続ける | 入力と無関係 |
| llm-jp-moshi（変換済み）+ 日本語プロンプト + wav 声プロンプト、seed 2 種 | `はいどうぞ。今いい天気だから出かけたくなってきましたけど…` / `健康のためによくするなら…何かありますか?` | 日本語で話すが、先生役の指示には従わず、自分から質問・雑談する。声プロンプトの文面（天気の話）を引きずる |

出力音声を Whisper（faster-whisper medium）で書き起こして確認した。PersonaPlex の出力は英語（言語判定 0.99）、llm-jp-moshi の出力は日本語（0.99〜1.00）で、内容もテキストトークンと一致した。

つまり PersonaPlex はペルソナ制御ができて日本語ができず、llm-jp-moshi はその逆。推論ランタイム（プロンプト注入、サーバ、UI）は重みとトークナイザを差し替えるだけでそのまま動く。

### 4. 学習の実現可能性（`scripts/train_smoke_test.py`、40 秒・batch 1）

| 重み | 学習対象 | agent 音声 loss (cb0..7) | user 音声 loss (cb0..7) | ピーク VRAM |
|---|---|---|---|---|
| llm-jp（変換済み） | 全パラメータ 8.37B | 1.8〜2.3 | 3.1〜13.0 | 31.7 GiB |
| PersonaPlex | depformer + テキスト層 1.66B | 2.1〜2.9 | 1.1〜1.6 | 21.7 GiB |

- `LMModel.forward_train` は 17 ストリーム（テキスト 1 + agent 8 + user 8）でそのまま動き、逆伝播もできる。
- 変換直後の llm-jp はユーザ音声ストリームの予測が未学習（loss 3〜13）。PersonaPlex と同じく、ここも学習対象になる。
- 全パラメータ学習は勾配だけで 31.7 GiB。Adam の状態を足すと 48 GB の RTX 6000 Ada 1 枚には載らない。このマシンでは LoRA か部分学習が現実的。活性の占める分は小さいので、系列長を PersonaPlex の 163 秒まで伸ばす余地はある。

## 調査で分かったこと（出典付き）

確認できなかった点は「未確認」と書いた。

- **PersonaPlex の学習レシピ**（[論文](https://arxiv.org/html/2602.06053)）: Moshi 重みから開始。合成サービス対話 1,840 h + QA 410 h、公開版はさらに Fisher 1,217 h。台本は LLM、音声は Chatterbox TTS。系列は「声プロンプト（agent 音声チャネル）→ テキストプロンプト（agent テキストチャネル）→ 対話」で、プロンプト中のユーザ音声は 440 Hz 正弦波。プロンプト部は loss をマスク。Adam、lr は temporal 2e-6 / depth 4e-6、24,576 step、batch 32、8×A100 で 6 時間。凍結範囲、テキストの時刻アライン方法、dep_q 8→16 の初期化は論文に記載なし。
- **llm-jp-moshi-v1**（[モデルカード](https://huggingface.co/llm-jp/llm-jp-moshi-v1)）: Moshi を J-CHAT 約 69,000 h で事前学習し、Zoom 雑談約 1,000 h でファインチューニング。Apache-2.0。トークナイザは rinna/japanese-gpt2-medium の SentencePiece。
- **J-Moshi**（[GitHub](https://github.com/nu-dialogue/j-moshi)）: 同様の 2 段階学習だが CC BY-NC 4.0（商用不可）。学習コード [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune) は Apache-2.0 で、ユーザストリーム拡張のオプションを持つ。
- **kyutai-labs/moshi-finetune**（[GitHub](https://github.com/kyutai-labs/moshi-finetune)）: LoRA 対応。データはステレオ wav + 時刻付き書き起こし json。プロンプト区間の loss マスクや 440 Hz 処理は無い。PersonaPlex の dep_q=16 を扱えるかは未確認。
- **先行事例**: NVIDIA の公式回答は「英語のみ」（issue #33）。韓国語版（RetentionLabs/personaplex-ko、語彙拡張 + LoRA）はカード自身が品質未達と記載。下記の moshi-finetune-r12 が日本語 Moshi 側から同じことを試みている。
- **既存の学習コード: [yusukek05/moshi-finetune-r12](https://github.com/yusukek05/moshi-finetune-r12)**（Apache-2.0、`45dabaf`、README と `docs/personaplex_poc_*.md`、`finetune.py` の引数を読んだ。実行はしていない）。LLM-jp-Moshi に PersonaPlex 方式のプロンプト条件付けを載せる PoC で、A 案の学習部分をほぼ実装済み。
  - `--system_prompt_conditioning`: 声/テキストプロンプトを系列の先頭に付け、その区間の loss をマスクする。
  - `--paper_prefix`: 論文 Fig.1 準拠（ユーザ音声 = 440 Hz 正弦波、agent 音声 = 無音、区切りトークン）。
  - `--model_user_stream` と `--extend_modules_for_user_stream`: dep_q=16 のユーザ音声ストリーム学習。
  - loss の重み（`--acoustic_loss_weight`、`--text_padding_loss_weight`）、台本生成・persona 付与・評価のスクリプトもある。
  - 全パラメータ学習（Accelerate + DeepSpeed、8×H100 想定）。`finetune.py` に LoRA は無い。
  - 同 PoC の結果: 800 対話（丁寧体/タメ口の 2 値制御）では、学習データ上でプロンプトが効く（82〜96%）が、held-out では約 50% で汎化しなかった。著者は原因をデータ量不足としている。
  - 公式リポジトリかどうか、upstream PersonaPlex の推論ランタイムとプロンプト形式（区切りトークン、`<system>` タグ）が一致するかは未確認。
- **ライセンス**: PersonaPlex のコードは MIT。重みは NVIDIA Open Model License で派生・再配布可（表示義務あり）。推奨方針は PersonaPlex の重みを使わないので、重み側は llm-jp の Apache-2.0 と、Kyutai 由来の Mimi コーデックのライセンス（CC-BY-4.0 とされるが今回は未確認）に従う。
- **日本語データ**: 公開されている 2 チャネル実対話は数十〜百時間規模（CALLHOME Japanese は LDC 有償、Tabidachi は同意書が必要）。J-CHAT は非商用。主データは PersonaPlex と同じく LLM 台本 + ゼロショット TTS の合成になる。TTS 候補は Chatterbox Multilingual（MIT）、Qwen3-TTS（Apache-2.0）、MOSS-TTSD（Apache-2.0）、CosyVoice2（Apache-2.0）。日本語での品質はいずれも未検証。

## Seiran と既存作業（2026-09-30 追記）

Seiran を読み取り専用で確認した（ジョブ投入・ファイル変更はしていない）。

### 計算資源

- `defq` に 12 ノード、各 `gpu:nvidia:8`、224 CPU、約 1.8 TB メモリ。確認時 7 ノード idle、2 ノード drain。
- GPU は B200（既存の検証記録による。今回は GPU 上で確認していない）。アカウントは `aitc`（制限なし）と `proj21999`（4 GPU、24 時間まで）。
- コンテナは `singularity`。既存 SIF は `physicsnemo_25.11_user.sif` など 3 つ。ログインノードから huggingface.co と pypi.org に到達できる。
- `/home` は 5.0 TB 中 366 GB 空き（93% 使用）。

これにより、全パラメータ学習が可能になる。「48 GB × 2 なので LoRA」という上の前提は Seiran を使う場合は外れる。

### 既に Seiran 上にある作業

`/home/projects/aitc/yano/personaplex/` に 2 系統、計 55 GB がある。git 管理されていない。

| | `moshi-finetune-llmjp-personaplex` | `japanese_persona_starter` |
|---|---|---|
| 時期 | 8/28〜9/7 | 9/7〜9/15 |
| ベース | llm-jp-moshi-v1（ユーザストリーム込み、dep_q=16） | nu-dialogue/j-moshi-ext（CC BY-NC、agent 側 8 のみ） |
| 学習 | 全パラメータ、2 ノード × 8 GPU、DeepSpeed | 1 GPU、Temporal の `out_proj` に LoRA |
| 実行済み | J-CHAT 1000 h の学習、役割アノテーション 2000〜3000 件、合成 TTS 160〜320 件、混合 640/800 件の学習、役割 × 文体の評価（ログ 2,250 本） | 8 例での役割上書き学習、発音診断、役割ベンチマーク |
| 分かっている結果 | 結果をまとめた文書が見当たらない。最後の評価ログ（5593）の出力は「なんか、ちょっとね」の反復 | 内部テキストでは時刻の切替を確認、音声は発音異常が報告され品質ゲート未通過（`SEIRAN_STATUS_ja.md`） |

- llm-jp 系統の学習済みチェックポイントと評価出力の置き場だった `/lustre/home/projects/proj21999/yano/` は現在存在しない。残っているのは `init_models/`（16 GB）と 1 step のスモーク出力だけ。
- mercury 側には `/data/japanese_persona_starter` と `/data/Irodori-TTS` がある。中身は今回見ていない。
- この文書の A 案は `moshi-finetune-llmjp-personaplex` とほぼ同じ方針で、学習コード・Seiran 実行層・J-CHAT の前処理はそちらに既にある。

llm-jp 系統の各実験の結論（どの設定で何が効いたか）は、ログを個別に読まないと分からない。今回は読んでいない。

## 決定: (c) ここで一から組む（2026-09-30）

既存 2 系統は引き継がず、参照もしない。学習・データ・評価をこのリポジトリの `japersonaplex/` に新しく実装する。
先行 2 系統が試したことを再度たどる部分があることは承知のうえでの選択。未回答の項目は次の既定で進めている:
研究用途（ただし部品は商用可ライセンスを優先）、台本生成はローカル、コミットは指示があるまでしない。

進め方は「仮データで mercury 上の全工程を通す → Seiran へ持っていく → 本データに差し替える」。
結果と使い方は [ja-pilot.md](ja-pilot.md)。

## 方針の比較

| | A. llm-jp-moshi + PersonaPlex 方式の学習（推奨） | B. PersonaPlex に日本語を教える |
|---|---|---|
| 日本語の発話・聞き取り | 学習済み（6.9 万時間分） | ゼロから。先行の J-Moshi は 128 GPU × 36 時間 |
| トークナイザ | 日本語 SPM をそのまま使う | 差し替えとテキスト埋め込みの再学習が必須 |
| 追加で学習するもの | 役割プロンプト、声プロンプト、ユーザ音声ストリーム | 日本語そのもの + 上記の維持 |
| 必要データ | 日本語の合成ロールプレイ対話 数百〜2,000 h | 日本語対話 数万時間 |
| このマシン（48 GB × 2）での実行 | LoRA / 部分学習で可能 | 非現実的 |

## 進め方（A 案）

1. **推論側**: 重み・トークナイザの切替（済、CLI 引数で可能）、日本語プロンプト例、Web UI の日本語プリセット。
2. **データ**: LLM で日本語の役割プロンプト付き対話台本を生成 → ゼロショット TTS で 2 話者を別チャネルに合成 → Mimi でトークン化、テキストは日本語 SPM で時刻アライン。まず 10 h 程度の小規模セットでパイプラインを通す。
3. **学習**: moshi-finetune-r12 を土台にする（系列化・loss マスク・ユーザストリームは実装済み）。このマシンで回すには LoRA か部分学習の追加が必要。推論時のプロンプト形式を upstream の `LMGen.step_system_prompts` と揃える。先行 PoC は 800 対話で汎化しなかったので、データ量が最大のリスク。
4. **評価**: 役割遵守（台本情報を正しく答えるか）、声の類似度、ターンテイキング、ASR による明瞭度。

## 未決事項（判断が必要）

- 用途は研究か商用か。TTS とデータの選択肢が変わる。
- 計算資源。このマシンの 2 枚だけか、クラスタを使えるか。LoRA か全パラメータ学習かが決まる。
- 台本生成に使う LLM（ローカルか API か）。

## 再現手順

```bash
# 環境（この環境では torch.compile が triton を見つけられず失敗するため NO_TORCH_COMPILE=1 が必要）
uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e moshi/ pyopenjtalk soundfile scipy
export NO_TORCH_COMPILE=1

# テスト入力とトークン数/秒
.venv/bin/python scripts/make_ja_test_input.py

# llm-jp-moshi を PersonaPlex 形式へ変換
.venv/bin/python scripts/convert_moshi_to_personaplex.py \
  --src <llm-jp-moshi-v1>/model.safetensors --dst checkpoints/llm-jp-moshi-v1-pp16/model.safetensors

# ベースライン
.venv/bin/python -m moshi.offline --voice-prompt NATF2.pt --text-prompt "あなたは賢くて親切な先生です。…" \
  --input-wav assets/test_ja/input_ja.wav --seed 42424242 \
  --output-wav outputs/baseline/pp_ja.wav --output-text outputs/baseline/pp_ja.json
.venv/bin/python -m moshi.offline --voice-prompt openjtalk_ja.wav --voice-prompt-dir assets/test_ja/voices \
  --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
  --tokenizer checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model \
  --text-prompt "あなたは賢くて親切な先生です。…" --input-wav assets/test_ja/input_ja.wav --seed 42424242 \
  --output-wav outputs/baseline/jp_wavvoice.wav --output-text outputs/baseline/jp_wavvoice.json

# 学習スモークテスト
.venv/bin/python scripts/train_smoke_test.py --moshi-weight checkpoints/llm-jp-moshi-v1-pp16/model.safetensors \
  --agent-wav outputs/baseline/jp_wavvoice.wav --user-wav assets/test_ja/input_ja.wav
```

## 限界

- ベースラインはテスト入力 1 本（合成音声）、seed 1〜2 種。傾向の確認であって定量評価ではない。
- 出力音声は人が聴取しておらず、判定は出力テキストトークンと Whisper の書き起こしに基づく。音質・自然さは未評価。
- スモークテストのテキストチャネルは全フレーム PAD で、text loss は参考値。
