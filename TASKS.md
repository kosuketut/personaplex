# PersonaPlex 日本語対応 タスク一覧

ブランチ: `ja-support`（`upstream` = NVIDIA/personaplex）

## フェーズ0: 現状把握
- [x] upstream を clone し、`.venv` に `moshi/` をインストール
- [x] コード調査（推論専用・学習ループ無し・英語 SPM 32k）
- [x] `nvidia/personaplex-7b-v1` と `llm-jp/llm-jp-moshi-v1` の重み・トークナイザ比較
- [x] 日本語テスト音声の用意
- [x] ベースライン1: PersonaPlex そのまま + 日本語入力
- [x] ベースライン2: llm-jp-moshi-v1 を PersonaPlex のローダで読み込み + 日本語入力
- [x] テキストトークン数/秒の実測（英語 SPM の byte fallback が 12.5 frame/s に収まるか）
- [x] 一次情報の調査（PersonaPlex 論文、llm-jp-moshi / J-Moshi、moshi-finetune、データ、ライセンス）

## フェーズ1: 方針決定
- [x] ベースライン結果と調査結果を `docs/ja-support-plan.md` にまとめる
- [x] llm-jp-moshi-v1 を PersonaPlex 形式へ変換（`scripts/convert_moshi_to_personaplex.py`）
- [x] 学習スモークテスト（`scripts/train_smoke_test.py`、forward_train と VRAM）
- [x] `pyloudnorm` を依存関係に追加（wav の声プロンプトで必要、upstream の記載漏れ）
- [ ] 方針を確定（推奨は A 案: llm-jp-moshi + PersonaPlex 方式の学習。用途・計算資源・台本生成 LLM の判断待ち）

## Seiran
- [x] Seiran の資源と既存作業の確認（読み取りのみ）
- [x] 既存 2 系統との関係: (c) を選択。ここで一から組み、既存 2 系統のコード・データは参照しない

## フェーズ2: パイプラインを mercury で通す（仮データ）
- [x] 系列組み立て `japersonaplex/sequence.py`（プロンプト区間を upstream `LMGen.step_system_prompts` と一致させる）
- [x] テスト `tests/test_sequence.py`（upstream がモデルへ渡す入力列と完全一致することを確認）
- [x] LoRA `japersonaplex/lora.py`、データセットと損失 `japersonaplex/data.py`
- [x] 前処理 `japersonaplex/prepare.py`（ステレオ wav + 時刻付き単語 json → npz）
- [x] 仮データ合成 `japersonaplex/synth_pilot.py`（テンプレート台本 + pyopenjtalk、400 対話 3.2 h）
- [x] 学習ランナー `japersonaplex/train.py`、flip 評価 `japersonaplex/eval_flip.py`
- [x] 仮データのトークン化
- [x] 学習前の flip 評価（0/8）
- [x] LoRA 学習と学習後の flip 評価（テキスト 15/16、結果は `docs/ja-pilot.md`）
- [x] 出力音声の ASR 採点（9〜11/16）
- [x] LoRA を焼き込んだ重みを無改造の `moshi.offline` で実行し、`eval_flip` と同一出力を確認
- [ ] `moshi.server` と Web UI で学習済み重みを動かす
- [ ] 人の声の入力で学習後モデルを確認（合成音声しか聞かせていない）
- [ ] 出力音声の人による聴取
- [x] 学習の途中再開（`train.py --resume`、`--stop-at`）

## フェーズ3: Seiran
- [x] sbatch の下書き `slurm/setup_env.sbatch`、`slurm/train_lora.sbatch`（未投入、lint 済み）
- [ ] Seiran 上の置き場所とデータ・チェックポイントの保存先を決める（判断待ち、/home は 93% 使用）
- [ ] `setup_env.sbatch` を投入して SIF の PyTorch で upstream コードが動くか確認（投入は指示待ち）
- [x] 複数 GPU・全パラメータ学習 `japersonaplex/train_full.py`（FSDP、再開が中断なしと一致することを 2 GPU・小モデルで確認）
- [x] `slurm/train_full.sbatch`（未投入）
- [ ] 実サイズでの全パラメータ学習の動作確認（B200 が必要）

## フェーズ4: 本データ
- [ ] 部品選定 workflow（参照音声・TTS 7 候補・台本 LLM・アライメント・データ仕様）← 10/01 12:19 の /clear で中断（11 件中 3 件完了）。軽い成果物は `data/bakeoff/` に退避済み
  - [x] 参照音声: YODAS2（CC-BY-3.0）から 48 話者 `data/voicebank/refs_v0/`
  - [x] 台本 LLM: 推奨 Qwen3-30B-A3B-Instruct-2507-FP8、共有 GPU 時は Qwen3-14B-AWQ（`data/bakeoff/llm/`）
  - [x] データ仕様: `data/bakeoff/design/data_spec.md`
  - [ ] アライメント: 合成音声の正解では CTC 系アンサンブルが ±1 フレーム 99%（モーラ按分は 51〜56%）。neural 正解（set C）と速度測定が未了
  - [x] TTS 7 候補の CER/SIM 採点（GPU で再採点、`data/bakeoff/README.md`）。本案件は非商用と確定（10/01）。CER 最良は IndexTTS-2.5（bilibili ライセンス。非商用モデルの学習には使えるが、学習したモデルが Derivative Work になり条件を引き継ぐ）、次が Qwen3-TTS（Apache-2.0、条件なし）。sarashina は出力を他モデルの改良に使うことを禁じているので非商用でも除外。**TTS は Qwen3-TTS-12Hz-1.7B-Base に確定（10/01 ユーザ承認）**
  - [x] 上位 2 候補のバッチ合成の速度（共有 GPU で Qwen3-TTS バッチ 36 が実時間の 5.0 倍、IndexTTS-2.5 4 並列が 1.4 倍。バッチでも品質は落ちない）
  - [x] 名前の読み: TTS に漢字のまま渡すと意図した読みは 5/16、かなに置き換えると 15/16。名前は読みのかなで TTS に渡す（どの TTS でも同じ）
  - [ ] TTS の残り: vLLM での速度、人による聴取、多くの声での頑健性、相槌・短い発話の扱い（Irodori・Chatterbox・MOSS-TTSD は「はい。」などで破綻）
  - [ ] 部品の最終決定（`data/bakeoff/README.md` の表をもとに判断）
- [x] **本データのパイプライン M1**（既存の LLM 台本 12 本で TTS → アライメント → 組み立て → npz を通す。コードは `japersonaplex/datagen/`、通しは `scripts/datagen.sh`）
  - [x] 永続の環境 `data/envs/{tts,align,llm}` を uv で作った。LLM の重みを `~/.cache/huggingface/hub` へコピー
  - [x] 1 対話でテキストストリームの整合を確認（句読点は前の単語に結合、§5.3 のあふれ 0、`data.py` で読める）
  - [x] 声バンクの 18 声で Qwen3-TTS の CER を確認（声ごとの平均 0.00〜0.10。極端に悪い声はなし）
  - [x] TTS 段（名前は読みのかなで渡す、声プロンプトも同じ声で合成）
  - [x] アライメント段（単位列を受け取る形に移植。品質検査: CTC の unexplained、whisper の CER（フィラーと名前の部分を除く）、名前の読みはひらがな CTC で照合）
  - [x] 組み立て段（§3.2 のギャップと重なり、相槌、割り込み、音量、user 側のゲインと床ノイズ）
  - [x] 12 対話 → 品質検査で最大 3 回作り直し → 10 対話（0.16 h）の npz。`train.py` 6 ステップで損失が下がることを確認
- [x] **M2**: 台本生成器の移植（`scripts_llm.py`。名前は IPAdic の読み付き辞書、案 A、語彙に無い文字の名前は除外、評価専用業種の除外）、165 台本 → 149 対話 2.38 h、40 h の所要時間の見積り（`docs/ja-datagen.md`）
  - [x] 品質検査の誤検出を減らす（whisper のフィラー・文頭の「はい」・珍しい名前の漢字、ひらがな CTC の濁点の取り違え）。user の CER は 0.30 に緩めた
  - [x] 作り直しでは漢字をかなで渡す（Qwen3-TTS の漢字の読み違いが繰り返されたため）
  - [x] ターンテイキング: 交替の間隔を最後の単語の終わりから測る、発話内の長い無音を詰める、聞き手の相槌を差し込む（相槌の音声は声ごとに事前に作り、読みで選別）。統計は `stats.py`
  - [x] 評価用の台本: H-name 67 本、H-domain 71 本（`data/datagen/eval/`。音声はまだ）。`tts.py --voice-split heldout` で H-voice
  - [x] M2 の npz 149 本と `train.py` 4 ステップ。`scripts/datagen.sh` を新しいディレクトリで 6 台本、頭から通した
- [x] **パイロット 40 h の生成を Seiran で**（10/02 に Seiran の使用許可。/tmp の bakeoff 245 GB は削除済み）
  - [x] Seiran 用のスクリプト: `scripts/seiran_setup_envs.sh`、`slurm/datagen_{check,scripts,shard,merge}.sbatch`、`slurm/submit_datagen.sh`、`japersonaplex/datagen/merge.py`（lint 済み。評価専用の声の経路と merge は mercury で確認）
  - [x] SSH の共有接続 → `/home/projects/aitc/yano/japersonaplex` に rsync（44 MB、AGENTS.md を追加）。/home は 1.4 TB 空き
  - [x] ログインノードで環境とモデル（`logs/setup_envs.log`。約 10 分、86 GB、テスト 15 件合格）
  - [x] `datagen_check.sbatch`: 8536 は vLLM が CUDA 13 の ptxas で停止 → `datagen_common.sh` でコンテナの CUDA 向けの変数を外す。
    8537 は声の manifest の絶対パス（mercury）で停止 → `tts.py` の `load_voices` で manifest の隣から読む。8538 で全工程が通った
    （4 分 19 秒、5/5 対話、MaxRSS 12 GB 弱、計算ノードから HF と PyPI に出られる）
  - [x] `RUN=pilot40 bash slurm/submit_datagen.sh`（10/02 13:54 投入、15:22 完了。台本 8539 → シャード 8540〜8544 → まとめ 8545、失敗 0）。
    学習用 main 29.9 h（2,071 対話）＋ QA 7.4 h（414）、評価用 H-name 1.2 h・H-domain 1.3 h・H-voice 1.4 h。npz 2,756 本を `PromptedDialogues` で読めた
  - [ ] main が目安より約 4 h 少ない（必要なら seed を変えて追加の RUN）
  - [ ] シャードごとに声プロンプトと相槌の音声を作り直している（1 本あたり約 6.5 分の重複）。次の RUN の前に共有するか検討
  - 10/02 13:20 時点で 12 ノード中 10 が drain（9 ノードは 10/01 18:25 の Prolog error）。使える GPU は dgx08 の 8 枚と dgx10 の 1 枚
  - [x] 初回学習は QA を除き main だけ（10/02 ユーザ決定。QA の無作為 12 本中 4 本に事実の誤り）
  - [x] 初回学習: 1 GPU の LoRA（pilot_v2 と同じ設定でデータだけ替える）。結果は `docs/ja-pilot40-train.md`
    - [x] `train.py --extra-valid`（H-name・H-domain・H-voice の損失）、`slurm/train_lora.sbatch` を pilot40 の置き方に
    - [x] 動作確認 8565（20 step、保存、flip の LMGen 経路と ASR 採点を B200 で、2 分）→ 本番 8566（1,200 step、34 分）
    - [x] 評価専用の名前・業種・声で flip 形式の評価セット flip40（78 問、`japersonaplex.datagen.make_flip`）、`slurm/eval_flip.sbatch`、`scripts/flip_table.py`
    - [x] 評価 8567〜8569（step 0・400・600・800・1200）。担当者名 7/16 → 16/16。代表は step 800
  - [ ] **データと評価を直してから全パラメータ学習（10/02 ユーザ決定）**
    - [x] テキストストリームの数字を算用数字に（`units._restore_numbers`。pilot40 の数字を含む 8,086 発話すべてで戻る。TTS とアライメントの入力は v1 と同一）。同じ音声から漢数字版の npz も作る（`NPZ_KANJI=1`、A/B 用）。動作確認 8576
    - [x] 台本の偏り: 例文の「来週の土曜日」を外す、用件 8 → 16 種（予約関係 3 種）、日付の言い方を指定、半分の台本で端数のある値段。v2 の台本（8574）で「来週の」49% → 15%、「予約」46% → 33%、端数のある値段 11% → 56%
    - [x] flip40 v2: 事実を直接聞く質問を user の声で TTS（`make_flip.py --direct`、`slurm/make_flip.sbatch`、8573）。126 問、質問の ASR 一致 125/126
    - [x] v1 の step 0・800 を flip40 v2（余白 9 秒）で評価（8575）。step 800 のテキスト: H-name 9/40、H-domain 13/40、H-voice 10/34、valid 6/12。外れ 143 のうち 50 が答えの途中で切れていた
    - [x] flip40 v2 の余白を 15 秒に延ばした（音声の末尾に無音を足しただけ。9 秒版は `flip40v2_t9`、結果は `eval_flip40v2t9_*`）
    - [x] v1 の step 800 を 15 秒版で評価（8602）。評価専用 114 問で 43
    - [x] pilot40 v2 の生成（シャード 8578〜8581、まとめ 8582。main 34.7 + 0.4 h）→ LoRA 2 本（算用数字 8590、漢数字 8591）→ 評価（8592、8593）
    - [x] 数字の A/B: 算用数字 43 対 漢数字 33（step 800）、言い間違い 19 対 26。テキストは算用数字に決定
    - [ ] データ側の変更は下がった可能性（v1 43 → v2 漢数字 33、ASR 35 → 23）。台本の変更か数字の置き方（v2 漢数字は数字を 1 単位にまとめた）かは未切り分け
    - [ ] 残る最大の外れは「聞かれた事実に触れない」（約 35%）。仮説は聞き取りの弱さ（LoRA は user 側の損失なし）。評価の交絡（質問が元の挨拶直後に置かれ、モデルの挨拶と重なる）も残る
      - 案 A: LoRA `--user-weight 1`（1 GPU、35 分）で user 側の損失の効果だけを見る
      - 案 B: 全パラメータ学習（8 GPU。動作確認の後に本番）
      - [x] 10/03 ユーザ指示: 案 A と案 B の動作確認を同時に流す。dgx08 の 1 GPU を別ユーザの対話型ジョブが使っていて 8 GPU が空かないので、B は 7 GPU にした
      - [x] 案 B の動作確認（9071 → 9072 → 9073）: 2 step で止めて保存、再開して 4 step、FULL=1 で評価まで通過。RSS 332 GB、1 GPU 40 GiB、約 1 秒/step
      - [x] 案 B の本番 full_v1（9074、7 GPU、500 step ≒ 6 エポック、学習率は既定 2e-6 / 4e-6）。H-name のテキスト損失は step 300 で最小、LoRA と同じ水準
      - [x] 評価: 案 A（9070、step 400・800・1200）、案 B（9075、step 200・300・500）。flip40 v2 の 114 問で A 39〜44、B 41〜44（user 側なしの LoRA は 43〜46）。「触れていない」は 82〜87 で変わらず。pilot の flip では A・B とも悪化
      - [x] 評価の交絡の確認: 質問と agent の発話の重なりが 25% 未満の回答でも 23〜24% が数字に触れない。交絡は主因ではない
      - [x] データ側の最初の集計: 3 モデルがそろって当てる問の半分は営業時間・受付時間、そろって外す問は種類がばらばら（まれな種類の事実に弱い）。台本の「聞かれてすぐ答える」組は 43% だが、単位ごとの割合は評価と合わない
      - [ ] 次の案: 台本で各事実を一度は直接聞かせて答えさせる、事実の種類を増やす、データを増やす。flip40 v2 の不備のある 2〜3 問を直す
      - [ ] 評価ジョブの ASR を GPU で回すか別ジョブにする（今回も GPU 時間の 2/3 が評価）
  - [ ] **データ v3（中身の直し。10/04 ユーザから方向の判断を任され、advisor と相談して決定）**
    - [x] 確認 1（CPU）: 最初の user 発話で数字の事実を聞いて次の agent 発話で答える対話は 9%。項目が学習台本に 100 回以上出る問は正答 78%、出ない問は 42%（`docs/ja-pilot40-train.md`）
    - [x] 確認 2（9174）: 質問を 2 番目の user 発話の直前に置いた flip40v2u2 でも、pilot40v2_digits step 800 のテキストは 43/109 で同じ（「触れていない」77 → 68）。位置のずれは主因ではない
    - [x] flip40 v2 の不備のある 5 問を `flip_table.py` の `EXCLUDE_V2` で外す（114 → 109 問。どのモデルも 5 問を当てていないので正答の数は変わらない）
    - [x] 評価ジョブの whisper を GPU に（`asr_flip.py`。9174 で動作を確かめる）
    - [x] 台本の変更（`scripts_llm.py --fact-items sample --ask-facts 2 --ask-first P`、テスト `tests/test_datagen_scripts_llm.py`）。項目は業種の一覧から 2〜3、営業時間の類は 30%、共通の一覧から半分の台本で 1、LLM が考える項目 0〜2
    - [x] 台本の試し生成 300 本（9179、`RUN=v3probe`）: 合格 70%。LLM が営業時間の類を足す（42%）ので落とす検査を追加（合格は 57% の見込み）
    - [x] 生成（`RUN=pilot40v3`、SEED=3000、サービス 7,000・雑談 1,700、main だけ、32 シャード。台本 9290 → シャード 9291 → まとめ 9292、01:10〜03:23）。train 4,809 対話 66.4 h、valid 49。シャード 0〜15 の部分集合 `npz_train_half` は 2,413 対話 33.1 h（v2 は 2,363 対話 34.7 h）
    - [x] 台本の集計（`docs/ja-datagen.md` の「データ v3」）: サービス 3,923・雑談 1,231 本。最初の発話で聞いて答える 9% → 59%、項目の種類 411 → 1,067、営業時間・受付時間 18% → 7%
    - [x] LoRA 2 本（設定は pilot40v2_digits と同じ）: pilot40v3_half（9473）、pilot40v3_full（9474、2,400 step）→ 評価 9475、9476、9592。
      109 問の正答: v2 43 → v3 部分集合 61（step 800）→ v3 全量 69（step 1200、ASR 53、pilot の flip 11/16）。質問の形を変えると順位が変わる（下の flip40v3）
    - [x] 間接の質問の flip40 v1 で確認（9617、9618）: v2 21 対 v3 21/72。v3 の落ち込みは時刻（「やってますかね」型）に集中、時刻以外は同じ。伸びの一部は評価の形に寄せた効果
    - [x] 質問の形を変えた評価セット flip40v3_paraphrase（105 問）・flip40v3_situational（91 問）（`flip_questions.py`、`make_flip.py --questions`、`slurm/make_flip_forms.sbatch`、9624）→ 評価 9625〜9627。
      言い換え: v2 38 / v3 部分集合 55 / v3 全量 56。事情を添えた質問: 36 / 46 / 34。v3 全量は長めの質問・間接の質問では v2 と同じ
  - [ ] **データ v4（10/04 ユーザ承認「進めて」）**: user の聞き方を「項目名で直接・言い換え・事情を添えて・間接」に混ぜる（`--ask-forms`）、営業時間の類を入れる台本を 30% → 55%（`--p-hours`）。ほかは v3 と同じ
    - [x] `scripts_llm.py --ask-forms --p-hours`、検査（項目名以外の形では user が項目名を言わない）、テスト 9 件
    - [x] 試し生成 3 回: 9628（合格 34%。事情を添えた形が守られない、user が値を先に言う）→ 例と検査を足して 9643（36%。例文をそのまま写す）→ 例文を外して mercury で 400 本（28%。営業時間の類 14%、写しなし）
    - [x] 生成（`RUN=pilot40v4`、SEED=4000、サービス 13,000・雑談 1,700、32 シャード、`HALF_SHARDS=16`。台本 10032 → シャード 10033 → まとめ 10034、13:12〜16:37）。
      台本 サービス 3,369（合格 26%）・雑談 1,235。train 4,289 対話 62.8 h、valid 34。部分集合 2,148 対話 31.5 h（v3 は 66.4 h と 33.1 h）
    - [x] LoRA 2 本（v3 と同じ設定）: pilot40v4_half（10035）、pilot40v4_full（10036）。損失の底は部分集合 step 800、全量 step 1600
    - [x] 評価（10037〜10045、10622〜10626）: 4 つの形の合計 v2 138、v3 部分集合 179、v3 全量 180、v4 部分集合 174、**v4 全量 step 1600 205**。
      seed 3 つの片側の平均で、v4 全量 step 1600 は v3 全量 step 1200 より 事情 +31・間接 +20・言い換え +11・直接 −21。代表は pilot40v4_full step 1600。pilot の flip の 6/16 は seed による振れ（5〜12/16）
    - [x] `eval_flip.sbatch` に `EVAL_ARGS`・`OUT_SUFFIX`（seed を変えた評価用）、`datagen_merge.sbatch` に `HALF_SHARDS`
  - [ ] **v4 の作り方でデータを 2 倍に（10/05 ユーザ決定「A で進めて」）**
    - [x] 生成 `RUN=pilot40v4b`（SEED=5000、ほかは v4 と同じ。`COMBINE_WITH` で v4 と合わせた `npz_train_plus` も作る）: 台本 10628 → シャード 10629 → まとめ 10630。train 4,210 対話 62.1 h、v4 と合わせて 8,499 対話 124.9 h
      - 10629 は 32 本中 30 本が失敗。GPU が空いていて 30 本が同時に始まり、アライメントのモデルの読み込みで HF の API が 429 Too Many Requests を返した。
        まとめ（afterany）は 267 対話だけで走り、学習 10631 がその不足したデータで始まったので止めた（出力も消した）。
      - `align.py` をキャッシュのローカルパスから読むように直し（`local_or_repo`、`whisper_model`。ネットワークを切って読めることを確認）、失敗した 30 本を同じ番号で投げ直し（10675、SHARDS=32）→ まとめ 10676（全シャード成功が条件）→ 学習 10677
    - [x] LoRA pilot40v4x2（10677、v4 + v4b 124.9 h、4,800 step、2 時間 20 分）。検証損失の底は step 2000（H-name 0.634）
    - [x] 評価 step 2000 × 4 つの形 × seed 3 つ（10717〜10728）: 片側の平均 直接 135・言い換え 145・事情 128・間接 89。v4 全量（136・143・132・82）と誤差の範囲で同じ。63 h から先は量で伸びない。step 2800 の直接の形（10729）も片側 145 で同じ
    - [x] 比べる相手の v4 全量 step 1600 の seed 1・2（10632〜10639）: 片側の平均 直接 136・言い換え 143・事情 132・間接 82。pilot の flip は 5〜12/16 と seed で大きく振れる
  - [x] **v3 と v4・v4b を合わせて 1 本（10/05 ユーザ決定「1 で進める」、8 GPU 並列可）**。v4 で項目名で聞く形を 2 割に減らしたのが直接の形の低下の原因か、を見る
    - [x] 結合データ `data/datagen/pilot40mix/main/npz_train`（v3・v4・v4b の npz_train へのリンク 13,308 本、約 191 h。壊れたリンク 0）
    - [x] LoRA pilot40mix（10730、7,500 step = v4 全量・v4x2 と同じ約 4.5 エポック、3 時間 37 分）。底は `valid_h_name` の text_nonpad で step 4800（0.597）
    - [x] 評価 step 4800 × 4 つの形 × seed 3 つ（10731〜10742、10717〜10728 と同じ 12 本、同時に 8 本まで）: 片側の平均 直接 161・言い換え 152・事情 139・間接 76（v4x2 は 135・145・128・89）。
      基準は満たさない（間接 −13）。間接の落ち込みは営業時間の問で「営業しております」とだけ答え時刻を言わない分（80 → 54/120）。誤った値は増えていない
    - [x] 代表を pilot40mix step 4800 に替えた（10/06 ユーザ判断「可」: 時刻を添えない「営業しております」の答えを可とする）
    - 判定（結果の前に決めた）: 直接の形の片側平均が 150 以上、かつ 言い換え・事情・間接が v4x2（145・128・89）から −10 以内なら成功。seed の幅（4〜15）に収まる差は「変わらない」とする
  - [ ] 次の候補（判断待ち）: 合わせたデータ（pilot40mix、191 h）で全パラメータ学習
  - [ ] 人の声での確認と聴取（pilot40mix step 4800。`outputs/pilot40mix/eval_pilot_step4800/*_A.wav`、`eval_flip40v3_situational_step4800/`）← ユーザ。10/06 flip_000_A（営業時間、正答）を聴取: 内容・話し方・音質とも問題なし。ほかの応答と、合成でない人の声での確認は未了。pilot の flip の 6/16 は seed による振れ（5〜13/16）で、弱くなった証拠は無い
    - 10/04 10:00 時点、Seiran の GPU 16 枚は同じアカウントの transolver_aime の配列（約 780 タスク、12 並列）と別ユーザで埋まっている。v4 はその後に動く
    - [ ] 人の声での確認と聴取（v3 の対話の自然さも）
    - [ ] flip_questions.py の「答えに期待値が出るか」の判定が 1 件も落とさない（甘い）。作り直すなら判定の聞き方を変える
      - [x] 10/04 ユーザ承認のうえ `outputs/full_smoke`（125 GB）と `outputs/full_v1/resume_step500`（94 GB）を削除。推論用の重み 5 つ（85 GB）は残した
  - [x] `train_full.sbatch` を pilot40 v2 の置き方と uv の環境に書き換え、`train_full.py --extra-valid`。`eval_flip.sbatch` に `FULL=1`。2 GPU の再開テストは通過（B200 の実サイズは未実行）
  - [ ] flip40 を事実を直接聞く質問だけに絞る（今は「触れていない」が 51/156 で雑音）→ 上の v2 で対応
  - [ ] 4 桁の値段の言い換えが弱い（pilot の flip の値段 0/8、step 800）。台本の偏り（サービスの 49% に「来週の〇曜日」、46% に「予約」。モデルへの漏れは 52 回答中 1 件ではっきりしない）
  - [ ] 8 GPU の全パラメータ学習（データと評価の直しは済んだ。投入は判断待ち。`slurm/train_full.sbatch` は準備済み）
  - [ ] QA の事実の扱い（未決。QA は main と分けて作ってあり、初回学習には入れない）
- [ ] 重なり（Overlap）が実データの 1/3（2.8 対 8.1 秒/分）。相槌の回数や交替の重なりの調整
- [ ] 評価セット・valid の分割（今は全部 split=train。40 h では 1% を valid に、H-name / H-domain / H-voice の音声化）
- [ ] 台本の MinHash による重複除去（§6.3）、声プロンプトの品質ゲート（§4.4）、user の発話内ポーズ（§3.4）、声の加工（§4.3）
- [ ] 固有名詞（店名・担当者名）を言えない弱点の解消（名前の種類を増やす。step 200 で店名 10/16、人名 7/16）
- [ ] LLM による台本生成（テンプレートの置き換え）
- [ ] 多話者・ゼロショット TTS の選定（品質比較は上の部品選定で実施済み: Chatterbox Multilingual / Qwen3-TTS / MOSS-TTSD / CosyVoice3 / IndexTTS2 / Irodori / sarashina。決定は未了）
- [ ] 日本語の強制アライメントで単語時刻を付ける（現状はモーラ数按分の近似）
- [ ] 割り込み・相槌・重なりを含む台本と合成
- [ ] 評価の拡充（声の類似度、ターンテイキング、役割遵守の採点）

## 残件
- [x] 学習コードの独立レビュー（10 件中 3 件確定: 未知語片が EPAD になる、Depformer 出力層の学習率、ASR 採点のカンマ。修正とテスト追加済み）
- [x] テキストプロンプトが空のとき upstream の `_step_text_prompt_core` が落ちる件の修正
