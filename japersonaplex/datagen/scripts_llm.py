#!/usr/bin/env python
"""ローカル LLM（vLLM）で、日本語 PersonaPlex 用のロールプレイ対話の台本を作る。

data/bakeoff/llm/generate_scripts.py の移植（部品選定で Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 を推奨）。変更点:
- 固有名詞（担当者名・客の名前・店名）は IPAdic の読み付き辞書から引く（names.NamePool、data_spec §1.7）。
  出力の "readings" に表記 -> 読み（ひらがな）を入れ、TTS にはこの読みを渡す（tts.py）
- 評価専用の業種（data_spec §1.3）は学習に使わない。--split heldout は評価専用の名前（H-name）、
  --domains heldout は評価専用の業種（H-domain）を使う
- 店名は「地名・名字・かなの語」+「業種の呼び名」で作る（読みは部品の読みをつなぐ）

出力 1 行（jsonl）:
  {"id", "scenario", "split", "seed", "model", "system_prompt", "facts",
   "turns": [{"speaker": "agent"|"user", "text", "type"}], "readings": {表記: 読み}, "spec", "check"}

環境は data/envs/llm（vLLM 0.11.0、torch 2.8.0+cu128、transformers 4.57.1）:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 data/envs/llm/bin/python -m japersonaplex.datagen.scripts_llm \
      --n 96 --scenario mix --out data/datagen/m2/scripts.jsonl --seed 0 --only-passing
GPU を他と共有していて空きが約 35 GiB 未満なら --model Qwen/Qwen3-14B-AWQ --gpu-mem 0.6。
プロンプトと検査の説明は docs/datagen-prompt-template.md（bakeoff の prompt_template.md の写し）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
IPADIC_DIR = Path(__file__).resolve().parents[2] / "data" / "dict" / "ipadic"
TOKENIZER = Path(__file__).resolve().parents[2] / "checkpoints" / "llm-jp-moshi-v1-pp16" / "tokenizer_spm_32k_3.model"

# ---------------------------------------------------------------------------------------------
# Pools (deterministic sampling from --seed)
# ---------------------------------------------------------------------------------------------
# 店名の前半に使うかなの語（読みは表記そのもの）
KANA_WORDS = (
    "あおぞら さくら みどり ひかり やまびこ こもれび つばめ わかば はるかぜ しらゆき たんぽぽ ほしぞら あさひ みなと かえで "
    "すずらん ゆうなぎ こはる あかつき いずみ ふたば みつば よつば まるいち なごみ ほっこり いろは ひまわり もみじ つくし "
    "すみれ れんげ あじさい なのはな こまち ことぶき はなまる スマイル グリーン サニー パール ルミエール フローラ ベル "
    "ハーモニー リバーサイド オリーブ ポプラ ミモザ アトリエ"
).split()
# 業種 -> (店名に付ける呼び名, 事実の例。LLM が具体的な値を作る)
BUSINESSES = {
    "喫茶店": ("喫茶店", "営業時間、定休日、人気メニューの値段、席数、駐車場"),
    "ラーメン店": ("ラーメン", "営業時間、定休日、ラーメンの値段、大盛り料金、混雑する時間"),
    "寿司店": ("寿司", "営業時間、定休日、おまかせの値段、予約の空き、席数"),
    "居酒屋": ("酒場", "営業時間、飲み放題の料金、コースの値段、個室の有無、予約の人数"),
    "洋菓子店": ("洋菓子店", "営業時間、ケーキの値段、予約の締め切り、定休日、日持ち"),
    "パン屋": ("ベーカリー", "営業時間、焼き上がり時刻、人気のパンの値段、予約の可否、定休日"),
    "花屋": ("生花店", "営業時間、花束の値段、配達料金、配達エリア、定休日"),
    "書店": ("書店", "営業時間、取り寄せにかかる日数、ポイント、駐車場、定休日"),
    "家電量販店": ("電機", "営業時間、商品の値段、在庫数、配送料、保証期間"),
    "ドラッグストア": ("薬局", "営業時間、処方せんの受付時間、待ち時間、駐車場、定休日"),
    "眼鏡店": ("眼鏡店", "営業時間、眼鏡の値段、仕上がり日数、視力測定の時間、保証期間"),
    "クリーニング店": ("クリーニング", "受付時間、ワイシャツの料金、仕上がり日数、コートの料金、定休日"),
    "美容院": ("美容室", "営業時間、カット料金、カラー料金、予約の空き、定休日"),
    "整骨院": ("整骨院", "受付時間、施術料金、1回の時間、予約の空き、休診日"),
    "引越し業者": ("引越センター", "見積もりの料金、作業人数、空いている日、段ボールの数、支払い方法"),
    "水道修理業者": ("水道設備", "出張料金、到着までの時間、作業料金、受付時間、支払い方法"),
    "鍵の修理店": ("ロックサービス", "出張料金、到着までの時間、鍵交換の料金、受付時間、支払い方法"),
    "不動産会社": ("不動産", "物件の家賃、間取り、駅からの徒歩分数、内見できる日、初期費用"),
    "マンション管理会社": ("管理サービス", "管理費、受付時間、ごみ出しの曜日、駐輪場の料金、修繕の予定"),
    "ガス会社": ("ガス", "受付時間、開栓の手続き、基本料金、立ち会いの時間、支払い方法"),
    "温泉旅館": ("旅館", "チェックイン時刻、1泊2食の料金、大浴場の時間、送迎、夕食の時間"),
    "ビジネスホテル": ("ホテル", "チェックイン時刻、チェックアウト時刻、1泊の料金、朝食の料金と時間、駐車場"),
    "旅行代理店": ("トラベル", "ツアー料金、出発日、日数、集合場所、キャンセル料"),
    "レンタカー店": ("レンタカー", "営業時間、6時間の料金、24時間の料金、保険料、乗り捨て料金"),
    "タクシー会社": ("交通", "迎車料金、到着までの時間、予約の締め切り、支払い方法、車いすの対応"),
    "高速バス会社": ("バス", "運賃、出発時刻、所要時間、予約の締め切り、乗り場"),
    "航空会社": ("航空", "運賃、出発時刻、手荷物の重さの制限、変更手数料、搭乗手続きの締め切り"),
    "銀行": ("銀行", "窓口の営業時間、ATMの時間、手数料、必要な持ち物、待ち時間"),
    "信用金庫": ("信用金庫", "窓口の営業時間、振込手数料、本人確認、住所変更の手続き、待ち時間"),
    "損害保険会社": ("損害保険", "保険料、補償の内容、受付時間、必要な書類、支払いまでの日数"),
    "クレジットカード会社": ("カード", "年会費、利用限度額、受付時間、再発行の日数、ポイント"),
    "携帯電話ショップ": ("モバイル", "営業時間、機種の値段、月額料金、手続きの所要時間、持ち物"),
    "光回線のサポート": ("ネット", "月額料金、工事の日程、受付時間、工事費、解約金"),
    "通販のサポート": ("ショップ", "受付時間、送料、返品の期限、届くまでの日数、支払い方法"),
    "内科クリニック": ("内科クリニック", "診療時間、休診日、初診の持ち物、予約の空き、駐車場"),
    "歯科医院": ("歯科医院", "診療時間、休診日、初診の持ち物、予約の空き、駐車場"),
    "介護施設": ("ケアセンター", "見学の時間、月額の費用、定員、面会の時間、送迎"),
    "市役所の窓口": ("市役所", "受付時間、必要な書類、手数料、混雑する時間、休日窓口"),
    "図書館": ("図書館", "開館時間、休館日、貸出冊数、貸出期間、予約の受け取り"),
    "学習塾": ("学習塾", "授業料、授業の曜日と時間、体験授業、クラスの人数、入塾金"),
    "英会話教室": ("英会話スクール", "レッスン料金、1回の時間、体験レッスン、開講曜日、講師の人数"),
    "自動車教習所": ("自動車学校", "教習料金、入校日、教習の時間、送迎バス、卒業までの日数"),
    "料理教室": ("クッキングスタジオ", "レッスン料金、開催曜日、定員、持ち物、1回の時間"),
    "宅配便の営業所": ("運輸", "受付時間、送料、再配達の締め切り時刻、荷物の大きさ制限、営業所の場所"),
    "スポーツジム": ("フィットネス", "営業時間、月会費、入会金、休館日、プールの有無"),
    "カラオケ店": ("カラオケ", "営業時間、1時間の料金、フリータイムの料金、部屋の人数、飲み放題"),
    "映画館": ("シネマ", "上映時刻、料金、割引の日、座席の予約、駐車場"),
    "美術館": ("美術館", "開館時間、休館日、入館料、展示の期間、駐車場"),
    "ゴルフ場": ("カントリークラブ", "プレー料金、予約の空き、スタート時刻、レンタルクラブの料金、送迎"),
    "写真館": ("写真館", "営業時間、撮影料金、撮影時間、予約の空き、データの受け取り日数"),
}
# data_spec §1.3 の評価専用の業種（--split heldout でだけ使う）。動物病院と類義のペットホテルは学習から外した
EVAL_BUSINESSES = {
    "料亭": ("料亭", "営業時間、コースの値段、個室の有無、予約の締め切り、送迎"),
    "楽器店": ("楽器", "営業時間、楽器の値段、修理にかかる日数、教室の月謝、定休日"),
    "葬儀社": ("葬祭", "受付時間、基本プランの料金、式場の人数、相談の予約、支払い方法"),
    "リフォーム会社": ("リフォーム", "見積もりの料金、工事の日数、受付時間、保証期間、支払い方法"),
    "フェリー会社": ("フェリー", "運賃、出港時刻、所要時間、車の料金、予約の締め切り"),
    "質屋": ("質店", "営業時間、預かり期間、利息、必要な持ち物、定休日"),
    "動物病院": ("動物病院", "診療時間、休診日、予防接種の料金、夜間対応、予約方法"),
    "結婚相談所": ("ブライダルサロン", "入会金、月会費、お見合いの料金、面談の予約、営業時間"),
}
# 予約に関わる用件は 16 中 3 にする（pilot40 v1 は 8 中 3 で、サービス台本の 46% に「予約」が出た）
SERVICE_GOALS = [
    "営業時間や休みの日を確認したい", "値段や料金を知りたい", "予約をしたい", "予約を変更またはキャンセルしたい",
    "行き方や場所を知りたい", "サービスの内容を詳しく聞きたい", "空き状況を確認したい", "必要な持ち物や手続きを確認したい",
    "支払い方法を確認したい", "扱っている商品やサービスがあるか知りたい", "料金の内訳や追加料金を確認したい",
    "仕上がりや届くまでの日数を知りたい", "前に利用したときのことで問い合わせたい（忘れ物、請求、仕上がりなど）",
    "混み具合や待ち時間を知りたい", "家族（子どもや高齢の親）と利用できるか相談したい", "時間と料金の両方をまとめて確認したい",
]
# 日付が話に出るときの言い方（v1 はサービス台本の 49% に「来週の〇曜日」が出た）
DATE_PHRASES = [
    "明日", "あさって", "今日の夕方", "今週の金曜日", "この土曜日", "今度の日曜日", "来週の月曜日", "来週の水曜日", "週明け",
    "10日", "15日", "20日の土曜日", "来月の3日", "月末", "来月の頭", "連休中", "金曜日の夜", "28日",
]
# v3（--fact-items sample、--ask-facts）: v2 の学習台本では、数字の事実の項目は営業時間（763）と受付時間（420）が飛び抜けて
# 多く、最初の user 発話で数字の事実を聞いて次の agent 発話で答える対話は 9%。flip40 v2 の正答は、学習台本に 100 回以上
# 出る項目で 78%、出ない項目で 41%。そこで項目を業種の一覧と共通の一覧から抜いて指定し、そのうち数字の項目を user に
# 項目名で直接聞かせ、agent にすぐ答えさせる（docs/ja-pilot40-train.md）
HOURS_ITEMS = {"営業時間", "受付時間", "診療時間", "開館時間", "窓口の営業時間"}
P_HOURS = 0.3
GENERIC_ITEMS = [
    "キャンセル料", "延長料金", "子ども料金", "学生割引", "最終受付の時刻", "電話の受付時間", "初回の割引", "ポイントの還元率",
    "回数券の料金", "会員の年会費", "支払いの期限", "当日の予約の締め切り", "予約の受付開始", "1日の受付人数", "平均の待ち時間",
    "送料無料になる金額", "領収書の再発行の手数料", "返事までの日数",
]
NUMERIC_ITEM = re.compile(r"料金|値段|料|費|額|金|時間|時刻|日数|期間|期限|締め切り|人数|数|分数|制限|割引|還元率|定員|日程|日$|空き|駐車場|開始")
NOT_NUMERIC_ITEM = re.compile(r"有無|方法|持ち物|場所|書類|エリア|対応|可否|予定|本人確認|乗り場|送迎|間取り|受け取り|曜日|休|体験|手続き$|内容")
# v4（--ask-forms）: v3 は user が項目名で聞く形だけで、項目名を使わない言い換えでは伸びが残ったが、事情を添えた質問と
# 元の台本の間接の質問（「〇曜日ってやってますかね」）では v2 と同じだった。聞き方を 4 つの形に混ぜる。
# 営業時間の類は v3 で数字の事実の 7%（v2 は 18%）まで減り、「やってますか」に時刻で答えなくなったので戻す（--p-hours）
ASK_FORMS = {
    "name": "「{item}」を、項目名をそのまま言って直接たずねる",
    "paraphrase": "「{item}」を、「{item}」という語を使わずに言い換えて、短く直接たずねる（例: 営業時間なら「何時まで開いてます？」）",
    # 例文を入れると、業種に合わなくてもそのまま写す（試し生成で「母と二人で行こうと思ってる」が通販の送料の質問に出た）
    "situational": "「{item}」を、同じ発話の中で、先にこの店の用件に合う自分の事情（日にち、人数、目的など）を言ってから"
                   "たずねる。「{item}」という語は使わない",
    "indirect": "「{item}」の値を直接は聞かず、行きたい日時や、したいことを言って、大丈夫かどうかをたずねる。「{item}」という語は"
                "使わない（例: 営業時間なら user「明日の夕方って、やってますかね」→ agent「はい、明日は10時から19時までやっております」）",
}
SITUATIONAL_MIN_CHARS = 25  # 事情を添えた質問の発話の最小の長さ（事情を別の発話で言って、質問だけ短くするのを落とす）
CASUAL_TOPICS = [
    "最近見た映画", "週末のキャンプ", "好きなラーメン", "猫を飼い始めたこと", "朝の散歩", "家庭菜園", "最近始めたランニング",
    "好きな音楽", "旅行の思い出", "料理の失敗談", "地元のお祭り", "子どものころの遊び", "最近読んだ本", "通勤電車",
    "カフェ巡り", "温泉", "サッカー観戦", "梅雨の過ごし方", "夏休みの計画", "紅葉狩り", "お正月の過ごし方", "引っ越し",
    "新しい趣味", "好きな季節", "コンビニスイーツ", "ゲーム", "釣り", "山登り", "写真撮影", "カラオケ", "ドラマ",
    "自転車通勤", "パン作り", "ペットの犬", "推し活", "健康診断", "在宅勤務", "語学の勉強", "祖父母の家", "花見",
]
OCCUPATIONS = [
    "会社員", "看護師", "高校の先生", "大学生", "パン屋の店員", "エンジニア", "主婦", "定年退職した元銀行員", "美容師",
    "バスの運転手", "保育士", "農家", "フリーのデザイナー", "介護士", "図書館の司書", "料理人", "消防士", "大学院生",
]
HOMETOWNS = ["北海道", "青森", "仙台", "新潟", "金沢", "長野", "静岡", "名古屋", "京都", "大阪", "神戸", "岡山", "広島",
             "高松", "松山", "福岡", "熊本", "鹿児島", "沖縄", "東京の下町", "横浜", "千葉", "埼玉"]
QA_SUBJECTS = [
    "天気と気象", "宇宙と星", "人の体と健康", "料理と栄養", "日本の歴史", "動物の生態", "植物の育て方", "お金の管理",
    "勉強の仕方", "睡眠", "運動とストレッチ", "地震と防災", "電気と節電", "環境とリサイクル", "日本の地理", "パソコンの使い方",
    "英語の勉強", "算数の考え方", "時間の使い方", "日本の伝統行事",
]
QA_LEARNERS = ["小学生", "中学生", "高校生", "大学生", "社会人", "子育て中の親", "高齢の方"]

SCENARIOS = ("service", "casual", "qa")
KNOWN_SURNAMES: set[str] = set()  # main で辞書から埋める（客の名前の取り違えの検査用）

# ---------------------------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------------------------
SYSTEM_MESSAGE = """あなたは日本語の音声対話データを作る脚本家です。全二重（同時に話せる）音声対話モデルの学習用に、電話や対面での「話し言葉」の会話台本を JSON で書きます。

# 出力
1つの JSON オブジェクトだけを出力する。キーは次の3つ。
- "facts": エージェントが知っている事実。キーは短い日本語の項目名、値は具体的な文字列（例 {"営業時間": "10時から19時まで"}）。3〜6項目。
- "system_prompt": エージェントへの役割指示。指定された書式で書き、facts の値をすべてそのまま含める。
- "turns": 会話。各要素は {"speaker": "agent" または "user", "text": 発話, "type": "utterance" / "backchannel" / "interruption"}。

# 会話のルール
1. 最初の発話は必ず agent。
2. agent が話す事実は system_prompt に書いた事実だけ。書いていないこと（場所、道順、所要時間、割引など）を聞かれたら、作らずに「申し訳ありません、そちらはちょっと分かりかねます」のように答えるか、確認すると言う。user の名前は、user が名乗った場合だけ使う。
3. 書き言葉ではなく、実際に声に出す話し言葉。1発話は短く（だいたい40字以内、長くても60字）。「えっと」「あの」「えー」「うーん」「そうですねえ」などのフィラーを自然に混ぜる。言い直しや「〜んですけど」「〜ですかね」も可。割り込まれた発話以外は、文の途中で切らない。
4. 相槌（「はい」「うん」「ええ」「なるほど」「へえ」「そうなんですね」）は、相手が話している途中に聞き手が打つもの。相手の発話を2つに分け、その間に聞き手の相槌ターン（type は "backchannel"）を置き、相槌の後は同じ相手が話を続ける。例: user「えっと、この前そちらで買ったものなんですけど、」→ agent「はい。」（backchannel）→ user「返品ってできますかね。」。自分の答えの前置きの「はい」は相槌ではなく、答えの発話に含める。相槌は2〜4回入れる。
5. 指定があれば割り込みを1回入れる。割り込みは user の発話で type を "interruption" にし、直前の agent の発話は途中で切れた形（「。」で終わらず「、」や言いかけで終わる）にする。agent は次のターンで割り込みに応じる。
6. 数字は読み上げたとおりの書き起こしにする。算用数字と助数詞で書き（7時半、22時、1050円、3名様、15分）、「:」「〜」「~」「-」「/」「%」「¥」「,」などの記号や桁区切りは使わない（「10時から19時まで」「10パーセント」「60から100回」）。電話番号は使わない。この決まりは system_prompt と facts にも当てはめる。
7. 英字（略語や単位を含む）、ローマ字、絵文字、顔文字、括弧書き、ト書きは、turns にも system_prompt にも使わない。カタカナ語は可。
8. ターン数は指定の範囲に収める。
"""

SERVICE_FORMAT = "あなたは{店名}という{業種}で働いています。名前は{名前}です。情報：{facts を「。」区切りで全部}"
CASUAL_FORMAT = "あなたは会話を楽しむのが好きです。{話題}について気軽に話してください。名前は{名前}です。{人物設定 facts を「。」区切りで全部}"
QA_FORMAT = "あなたは賢くて親しみやすい先生です。質問に答えたり、アドバイスをしたりしてください。わかりやすく、楽しく説明してください。名前は{名前}です。情報：{facts を「。」区切りで全部}"

EXAMPLE = {
    "facts": {"営業時間": "7時から22時まで", "定休日": "月曜日", "ナポリタン": "1050円"},
    "system_prompt": "あなたはこはる喫茶店という喫茶店で働いています。名前は松本です。情報：営業時間は7時から22時まで。定休日は月曜日。ナポリタンは1050円。",
    "turns": [
        {"speaker": "agent", "text": "はい、こはる喫茶店、松本でございます。", "type": "utterance"},
        {"speaker": "user", "text": "あ、もしもし。えっと、そちらって明日やってますかね。", "type": "utterance"},
        {"speaker": "agent", "text": "はい、明日は火曜日ですので、7時から22時まで営業して", "type": "utterance"},
        {"speaker": "user", "text": "あ、よかった。じゃあナポリタンって", "type": "interruption"},
        {"speaker": "agent", "text": "はい。", "type": "backchannel"},
        {"speaker": "user", "text": "おいくらでしたっけ。", "type": "utterance"},
        {"speaker": "agent", "text": "ナポリタンは1050円になります。", "type": "utterance"},
        {"speaker": "user", "text": "うん、わかりました。ありがとうございます。", "type": "utterance"},
    ],
}


def json_schema(min_turns: int = 6, max_turns: int = 16) -> dict:
    return {
        "type": "object",
        "properties": {
            "facts": {"type": "object", "additionalProperties": {"type": "string"}},
            "system_prompt": {"type": "string"},
            "turns": {
                "type": "array",
                "minItems": min_turns,
                "maxItems": max_turns,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["agent", "user"]},
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": ["utterance", "backchannel", "interruption"]},
                    },
                    "required": ["speaker", "text", "type"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts", "system_prompt", "turns"],
        "additionalProperties": False,
    }


def reading_of(text: str) -> str:
    """業種の呼び名など、普通の語の読み（pyopenjtalk の read）。固有名詞の読みには使わない。"""
    import pyopenjtalk

    from .units import kata2hira
    return kata2hira("".join(f["read"] for f in pyopenjtalk.run_frontend(text)))


def store_name(rng: random.Random, pool, suffix: str) -> tuple[str, str]:
    """店名と読み。前半は地名・名字・かなの語のどれか（data_spec §1.7-2）。"""
    kind = rng.random()
    if kind < 0.4:
        head = pool.place(rng)
    elif kind < 0.7:
        head = pool.surname(rng)
    else:
        w = rng.choice(KANA_WORDS)
        from .units import kata2hira
        return w + suffix, kata2hira(w) + reading_of(suffix)
    return head.surface + suffix, head.reading + reading_of(suffix)


def sample_fact_items(rng: random.Random, hint: str, ask_facts: int, ask_first: float,
                      ask_forms: dict[str, float] | None = None, p_hours: float = P_HOURS) -> dict:
    """v3: facts に入れる項目（業種の一覧から 2〜3、営業時間の類は p_hours、共通の一覧から半分の台本で 1）、
    LLM に考えさせる項目の数、user が直接聞く項目（数字になりそうな項目から ask_facts 個）。
    v4: ask_forms（形 -> 重み）があれば、聞く項目ごとに聞き方の形を抜く。"""
    items = hint.split("、")
    hours = [i for i in items if i in HOURS_ITEMS]
    others = [i for i in items if i not in HOURS_ITEMS]
    chosen = rng.sample(others, k=min(len(others), rng.randint(2, 3)))
    if hours and rng.random() < p_hours:
        chosen.append(hours[0])
    if rng.random() < 0.5:
        chosen.append(rng.choice([g for g in GENERIC_ITEMS if g not in chosen]))
    rng.shuffle(chosen)
    n_extra = min(rng.randint(0, 2), 6 - len(chosen))
    n_extra = max(n_extra, 3 - len(chosen))
    cands = [i for i in chosen if NUMERIC_ITEM.search(i) and not NOT_NUMERIC_ITEM.search(i)]
    ask = rng.sample(cands, k=min(ask_facts, len(cands)))
    out = {"fact_items": chosen, "n_extra": n_extra, "ask_items": ask, "ask_first": bool(ask) and rng.random() < ask_first}
    if ask_forms:  # v3（形の指定なし）と同じ seed で同じ台本になるよう、乱数はこの後でだけ使う
        out["ask_forms"] = rng.choices(list(ask_forms), weights=list(ask_forms.values()), k=len(ask))
    return out


def sample_spec(scenario: str, rng: random.Random, pool, businesses: dict, ask_facts: int = 0, ask_first: float = 0.0,
                fact_items: str = "hint", ask_forms: dict[str, float] | None = None, p_hours: float = P_HOURS) -> dict:
    lo = rng.choice([6, 8, 10, 12])
    spec = {"scenario": scenario, "turns_min": lo, "turns_max": min(16, lo + 4),
            "interruption": rng.random() < 0.5}
    readings: dict[str, str] = {}
    if scenario == "service":
        btype = rng.choice(sorted(businesses))
        suffix, hint = businesses[btype]
        store, store_reading = store_name(rng, pool, suffix)
        agent, user = pool.surname(rng), pool.surname(rng)
        while user.surface == agent.surface:
            user = pool.surname(rng)
        readings = {store: store_reading, agent.surface: agent.reading, user.surface: user.reading}
        spec.update(business_type=btype, store_name=store, agent_name=agent.surface,
                    user_name=user.surface, user_goal=rng.choice(SERVICE_GOALS), date_phrase=rng.choice(DATE_PHRASES),
                    odd_prices=rng.random() < 0.5,
                    fact_hint=hint, n_facts=rng.randint(3, 6), user_asks_unknown=rng.random() < 0.3)
        if fact_items == "sample":  # v3。v2（hint）と同じ seed で同じ台本になるよう、乱数はこの後でだけ使う
            spec.update(sample_fact_items(rng, hint, ask_facts, ask_first, ask_forms, p_hours))
            spec["n_facts"] = len(spec["fact_items"]) + spec["n_extra"]
    elif scenario == "casual":
        name = pool.full_name(rng) if rng.random() < 0.5 else pool.surname(rng)
        readings = {name.surface: name.reading}
        spec.update(topic=rng.choice(CASUAL_TOPICS), agent_name=name.surface, occupation=rng.choice(OCCUPATIONS),
                    hometown=rng.choice(HOMETOWNS), n_facts=rng.randint(3, 5))
    elif scenario == "qa":
        name = pool.surname(rng)
        readings = {name.surface: name.reading}
        spec.update(subject=rng.choice(QA_SUBJECTS), learner=rng.choice(QA_LEARNERS),
                    agent_name=name.surface, n_facts=rng.randint(3, 5))
    else:
        raise ValueError(scenario)
    spec["readings"] = readings
    return spec


def user_message(spec: dict) -> str:
    s = spec
    common = (f"- ターン数: {s['turns_min']}〜{s['turns_max']}\n"
              f"- 割り込み: {'1回入れる' if s['interruption'] else '入れない'}\n")
    if s["scenario"] == "service":
        unknown = ("- user は途中で1回、facts に無いことを聞く。agent は作り話をせず、分からないと答えるか確認すると言う。\n"
                   if s["user_asks_unknown"] else "")
        prices = "- 値段は 1280円、2350円、980円 のように端数のある値にする。\n" if s.get("odd_prices") else ""
        if "fact_items" in s:  # v3
            named = "".join(f"「{i}」" for i in s["fact_items"])
            extra = (f"ほかに、この業種で客が聞きそうな別の項目を{s['n_extra']}個、自分で考えて加える（営業時間や受付時間は、上に無ければ入れない）。"
                     if s["n_extra"] else "ほかの項目は入れない。")
            facts_line = (f"- facts: 次の項目を、この項目名のまま入れる: {named}。{extra}"
                          f"具体的な値（時刻、値段、日数、人数など数字を含むものを2つ以上）を作る。\n")
            ask = s["ask_items"]
            asked = "と".join(f"「{i}」" for i in ask)
            if ask and "ask_forms" in s:  # v4
                for item, form in zip(ask, s["ask_forms"]):
                    facts_line += f"- user は{ASK_FORMS[form].format(item=item)}。\n"
                facts_line += (f"- agent は、そのたずね方の直後の発話で、facts の値をそのまま言って答える（「はい、大丈夫です」だけで"
                               f"終わらせない）。{asked}の値は、数字を含む具体的な値にする。\n")
            elif ask:
                facts_line += (f"- user は{asked}を、項目名をそのまま言って、自然な言い方で直接たずねる。agent はその直後の発話で、"
                               f"facts の値をそのまま答える。{asked}の値は、数字を含む具体的な値にする。\n")
            if ask and s["ask_first"]:
                facts_line += f"- user は最初の発話で「{ask[0]}」をたずねる（名乗りやあいさつは短く）。\n"
        else:
            facts_line = f"- facts: {s['n_facts']}項目。具体的な値（時刻、値段、日数など数字を含むものを2つ以上）を作る。参考: {s['fact_hint']}\n"
        body = (f"種類: 店・施設への電話の問い合わせ\n"
                f"- 業種: {s['business_type']}\n- 店名: {s['store_name']}\n- agent の名前: {s['agent_name']}（スタッフ）\n"
                f"- user: 客。名前を聞かれたら「{s['user_name']}」と名乗る\n- user の用件: {s['user_goal']}\n"
                f"{facts_line}"
                f"{prices}"
                f"- 日付や曜日が話に出る場合は「{s.get('date_phrase', '明日')}」を使う。日付の話が要らない用件なら出さなくてよい。\n"
                f"{unknown}{common}"
                f"- system_prompt の書式: {SERVICE_FORMAT}\n"
                f"- agent は店名と名前を名乗って電話に出る。丁寧だが硬すぎない店員の話し方。")
    elif s["scenario"] == "casual":
        body = (f"種類: 気軽な雑談\n- 話題: {s['topic']}\n- agent の名前: {s['agent_name']}\n"
                f"- agent の人物設定: {s['occupation']}、出身は{s['hometown']}。facts に {s['n_facts']} 項目（職業、出身、話題に関する本人の具体的な経験や好みなど。数字を含むものを1つ以上）\n"
                f"{common}"
                f"- system_prompt の書式: {CASUAL_FORMAT}\n"
                f"- agent は友だちと話すようなくだけた口調（です・ますは混ぜてもよい）。agent も user に質問を返し、話を広げる。"
                f"agent が自分について話す内容は facts の範囲で。")
    else:
        body = (f"種類: 先生への質問\n- 分野: {s['subject']}\n- user: {s['learner']}\n- agent の名前: {s['agent_name']}（先生）\n"
                f"- facts: {s['n_facts']}項目。この分野で user が聞きそうな質問への答えになる、正確で一般的な知識（確かな事実だけ。数字を含むものを1つ以上）\n"
                f"{common}"
                f"- system_prompt の書式: {QA_FORMAT}\n"
                f"- user が質問し、agent は facts を使ってやさしく説明し、たとえや確認の質問をはさむ。user の年齢に合った話し方。")
    return (f"次の条件で台本を1本書いてください。\n\n{body}\n\n"
            f"出力例（形式の参考。内容はまねしない）:\n{json.dumps(EXAMPLE, ensure_ascii=False)}")


def build_messages(spec: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_message(spec)}]


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------
_KDIG = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KUNIT = {"十": 10, "百": 100, "千": 1000}
_KBIG = {"万": 10_000, "億": 100_000_000}
_KANJI_NUM = re.compile(r"[〇零一二三四五六七八九十百千万億]+")
# kanji that are numerals only inside fixed words (一緒, 一番 ...) — we only convert runs adjacent to counters
_COUNTER = r"(?=[時分秒円日月年人名階回件個本枚杯冊台泊歳才週度号番倍割キロメートルグラムパーセント])"


def kanji_to_int(s: str) -> int | None:
    total, section, digit = 0, 0, None
    if all(c in _KDIG for c in s):  # 二〇二五 style
        return int("".join(str(_KDIG[c]) for c in s))
    for c in s:
        if c in _KDIG:
            digit = _KDIG[c]
        elif c in _KUNIT:
            section += (digit if digit is not None else 1) * _KUNIT[c]
            digit = None
        elif c in _KBIG:
            section += digit or 0
            total += (section or 1) * _KBIG[c]
            section, digit = 0, None
        else:
            return None
    return total + section + (digit or 0)


def normalize_numbers(text: str) -> str:
    text = text.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(_KANJI_NUM.pattern + _COUNTER, lambda m: str(kanji_to_int(m.group()) or m.group()), text)
    return re.sub(r"(\d+)時半", r"\1時30分", text)


def numbers_in(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", normalize_numbers(text))}


def _norm(s: str) -> str:
    return re.sub(r"[\s、。・「」『』（）()：:]", "", normalize_numbers(s))


LATIN = re.compile(r"[A-Za-z]{2,}")
HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
BAD_SYMBOLS = re.compile(r"[:：~〜\-%％¥￥*#\[\]{}<>]|\d,\d")
FILLER = re.compile(r"えっと|えーっと|えー|あのー?|うーん|そうですねえ|まあ|その[ー、]")
SENT_END = ("。", "？", "?", "！", "!")


def sanitize(obj: dict) -> dict:
    """Cheap deterministic fixes: '9〜18時' -> '9から18時' (prompt, facts, turns); cut off interrupted agent turns."""
    fix = lambda x: re.sub(r"(?<=[0-9０-９])\s*[〜~～]\s*(?=[0-9０-９])", "から", x) if isinstance(x, str) else x  # noqa: E731
    obj["system_prompt"] = fix(obj.get("system_prompt", ""))
    obj["facts"] = {k: fix(v) for k, v in (obj.get("facts") or {}).items()}
    turns = obj.get("turns") or []
    for t in turns:
        t["text"] = fix(t.get("text", ""))
    # An agent turn followed by a user interruption must sound cut off: cut at the last "、" in its
    # second half (or just drop the final punctuation) so TTS/alignment see an unfinished phrase.
    # 相槌の付け間違い（12 字を超える、話し手が変わらない位置にある）は普通の発話に付け替える。M2 の 96 本では
    # 不合格の理由の最多がこれで、組み立て（assemble.py）は相槌も普通の発話も扱えるので、捨てる必要がない
    for i, t in enumerate(turns):
        if t.get("type") != "backchannel":
            continue
        prev = turns[i - 1] if i else {}
        if len(t.get("text", "")) > 12 or not prev.get("speaker") or prev.get("speaker") == t.get("speaker"):
            t["type"] = "utterance"
            t["relabeled_from_backchannel"] = True
    for prev, t in zip(turns, turns[1:]):
        if t.get("type") == "interruption" and prev.get("speaker") == "agent" and prev["text"].rstrip().endswith(SENT_END):
            txt = prev["text"].rstrip()
            cut = txt.rfind("、", len(txt) // 2)
            prev["text"] = txt[:cut + 1] if cut > 0 else txt.rstrip("。？?！!")
            prev["cut_off_by_sanitize"] = True
    return obj


def check_script(obj: dict, spec: dict) -> dict:
    issues: list[str] = []
    turns = obj.get("turns") or []
    sp = obj.get("system_prompt", "")
    facts = obj.get("facts") or {}
    # --- schema-level
    n = len(turns)
    if not (6 <= n <= 16):
        issues.append(f"turn_count={n}")
    if not (spec["turns_min"] <= n <= spec["turns_max"]):
        issues.append(f"turn_count_outside_requested({spec['turns_min']}-{spec['turns_max']})={n}")
    speakers = {t.get("speaker") for t in turns}
    if speakers != {"agent", "user"}:
        issues.append("missing_speaker")
    if turns and turns[0].get("speaker") != "agent":
        issues.append("first_turn_not_agent")
    if any(not str(t.get("text", "")).strip() for t in turns):
        issues.append("empty_text")
    if not (3 <= len(facts) <= 6):
        issues.append(f"n_facts={len(facts)}")
    # --- role prompt format
    head = {"service": "あなたは", "casual": "あなたは会話を楽しむのが好きです。", "qa": "あなたは賢くて親しみやすい先生です。"}[spec["scenario"]]
    if not sp.startswith(head):
        issues.append("system_prompt_format")
    if spec["scenario"] != "casual" and "情報：" not in sp:
        issues.append("system_prompt_no_info_section")
    if spec.get("agent_name") and spec["agent_name"] not in sp:
        issues.append("agent_name_not_in_prompt")
    if spec["scenario"] == "service":
        if spec["store_name"] not in sp:
            issues.append("store_name_not_in_prompt")
        first_agent = next((t["text"] for t in turns if t.get("speaker") == "agent"), "")
        if spec["store_name"] not in first_agent and spec["agent_name"] not in first_agent:
            issues.append("greeting_without_store_or_name")
    # --- facts must be in the system prompt (Moshi only ever sees system_prompt)
    sp_norm, sp_nums = _norm(sp), numbers_in(sp)
    facts_missing = []
    for k, v in facts.items():
        v = str(v)
        if _norm(v) in sp_norm:
            continue
        if numbers_in(v) and numbers_in(v) <= sp_nums:
            continue  # same numbers, wording changed
        facts_missing.append(k)
    if facts_missing:
        issues.append(f"facts_not_in_prompt={facts_missing}")
    # --- v3: 指定した項目が facts にあり、user が項目名で聞いて、agent が次の発話でその値を言う
    if spec.get("fact_items"):
        missing_items = [i for i in spec["fact_items"] if i not in facts]
        if missing_items:
            issues.append(f"fact_items_missing={missing_items}")
        # 指示しても LLM が営業時間・受付時間を足すことが多い（試し生成 209 本の 42%）。項目を平らにするため落とす
        extra_hours = [k for k in facts if k in HOURS_ITEMS and k not in spec["fact_items"]]
        if extra_hours:
            issues.append(f"unrequested_hours_item={extra_hours}")
        talk = [t for t in turns if t.get("type") != "backchannel"]
        forms = spec.get("ask_forms") or ["name"] * len(spec.get("ask_items") or [])
        for j, (item, form) in enumerate(zip(spec.get("ask_items") or [], forms)):
            nums = numbers_in(str(facts.get(item, "")))
            if not nums:
                issues.append(f"ask_item_not_numeric={item}")
                continue
            # agent の答えに、直前の user の発話に無い値の数字が出ること（user が自分で値を言って確かめる
            # 「1280円って一人分ですか」→「はい、1280円です」は、値を答えたことにしない）
            answered = [i for i, t in enumerate(talk[:-1]) if t.get("speaker") == "user"
                        and talk[i + 1].get("speaker") == "agent"
                        and (nums & numbers_in(talk[i + 1].get("text", ""))) - numbers_in(t.get("text", ""))]

            def in_form_at(text: str) -> bool:
                # 項目名で聞く形（v3）は項目名の先頭 2 文字が出ること、ほかの形（v4）は項目名そのものが出ないこと
                if form == "name":
                    return item[:2] in text
                return item not in text and (form != "situational" or len(text) >= SITUATIONAL_MIN_CHARS)
            in_form = [i for i in answered if in_form_at(talk[i].get("text", ""))]
            first_user = next((i for i, t in enumerate(talk) if t.get("speaker") == "user"), None)
            if not answered:
                issues.append(f"ask_item_not_answered={item}")
            elif not in_form:
                issues.append(f"ask_form_wrong={item}:{form}")
            elif j == 0 and spec.get("ask_first") and in_form[0] != first_user:
                issues.append("ask_first_missing")
    # --- every number the agent says must come from the prompt or from what the user said earlier
    allowed = set(sp_nums) | set(range(0, 3))  # 1回, 2人 etc. trivial counts are allowed
    unsupported = []
    for t in turns:
        nums = numbers_in(t.get("text", ""))
        if t.get("speaker") == "user":
            allowed |= nums
        else:
            bad = sorted(nums - allowed)
            if bad:
                unsupported.append({"text": t["text"], "numbers": bad})
    if unsupported:
        issues.append("agent_number_not_in_prompt")
    # --- interruption shape
    for i, t in enumerate(turns):
        if t.get("type") == "interruption":
            prev = turns[i - 1] if i else None
            if t.get("speaker") != "user" or prev is None or prev.get("speaker") != "agent":
                issues.append("interruption_not_user_after_agent")
            elif prev["text"].rstrip().endswith(SENT_END):
                issues.append("interrupted_turn_not_cut_off")
    n_int = sum(t.get("type") == "interruption" for t in turns)
    if spec["interruption"] and n_int == 0:
        issues.append("interruption_missing")
    if not spec["interruption"] and n_int > 0:
        issues.append("unrequested_interruption")
    # --- backchannel placement: listener's short turn sandwiched inside the other speaker's talk
    misplaced = 0
    for i, t in enumerate(turns):
        if t.get("type") == "backchannel":
            prev, nxt = (turns[i - 1] if i else {}), (turns[i + 1] if i + 1 < len(turns) else {})
            if not (prev.get("speaker") and prev.get("speaker") != t.get("speaker") and nxt.get("speaker") in (None, prev.get("speaker"))):
                misplaced += 1
    if misplaced:
        issues.append(f"backchannel_misplaced={misplaced}")
    too_long = sum(1 for t in turns if t.get("type") == "backchannel" and len(t.get("text", "")) > 12)
    if too_long:
        issues.append(f"backchannel_too_long={too_long}")
    # --- agent must not invent the caller's name
    if spec["scenario"] == "service":
        for t in turns:
            if t.get("speaker") == "agent":
                for m in re.finditer(r"([一-龥]{1,3})(?:様|さん)", t.get("text", "")):
                    if m.group(1) in KNOWN_SURNAMES and m.group(1) != spec["user_name"]:
                        issues.append(f"wrong_user_name={m.group(1)}")
    if LATIN.search(sp) or HANGUL.search(sp) or re.search(r"[〜~/／%％]", sp):
        issues.append("latin_or_symbols_in_system_prompt")
    # --- placeholders and turn-taking
    if re.search(r"〇〇|○○|××|ＸＸ|XX", sp + "".join(t.get("text", "") for t in turns)):
        issues.append("placeholder")
    same = sum(1 for a, b in zip(turns, turns[1:])
               if a.get("speaker") == b.get("speaker") and "backchannel" not in (a.get("type"), b.get("type")))
    if same:
        issues.append(f"same_speaker_consecutive={same}")
    # --- surface form
    all_text = "".join(t.get("text", "") for t in turns)
    if LATIN.search(all_text) or HANGUL.search(all_text):
        issues.append("latin_or_hangul_in_turns")
    if BAD_SYMBOLS.search(all_text):
        issues.append("symbols_in_turns")
    utt = [t for t in turns if t.get("type") != "backchannel"]
    lens = [len(t.get("text", "")) for t in utt] or [0]
    stats = {
        "n_turns": n,
        "n_backchannel": sum(t.get("type") == "backchannel" for t in turns),
        "n_interruption": n_int,
        "filler_turn_rate": round(sum(bool(FILLER.search(t.get("text", ""))) for t in utt) / max(1, len(utt)), 3),
        "mean_chars_per_utterance": round(sum(lens) / len(lens), 1),
        "max_chars_per_utterance": max(lens),
    }
    if stats["max_chars_per_utterance"] > 80:
        issues.append("long_turn>80")
    fact_ok = not facts_missing and not unsupported
    return {"ok": not issues, "fact_consistent": fact_ok, "issues": issues,
            "unsupported_agent_numbers": unsupported, **stats}


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--scenario", choices=SCENARIOS + ("mix",), default="mix")
    ap.add_argument("--out", required=True, help="output .jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "heldout"], default="train",
                    help="heldout は評価専用の名前だけを使う（data_spec §6.1 の H-name）")
    ap.add_argument("--domains", choices=["train", "heldout"], default="train",
                    help="heldout は評価専用の業種だけを使う（H-domain）。評価セットでは軸を混ぜない（§6.1）")
    ap.add_argument("--fact-items", choices=["hint", "sample"], default="hint",
                    help="サービス台本の facts の項目。hint は業種の例を参考に LLM 任せ（v2）、sample は項目を抜いて指定する（v3）")
    ap.add_argument("--ask-facts", type=int, default=0, help="--fact-items sample: user が項目名で直接聞く数字の項目の数")
    ap.add_argument("--ask-first", type=float, default=0.0, help="--fact-items sample: 最初の user 発話で聞く台本の割合")
    ap.add_argument("--ask-forms", default="", help="--fact-items sample: 聞き方の形と重み（v4: "
                    "name:0.3,paraphrase:0.25,situational:0.25,indirect:0.2）。空なら項目名で聞く形だけ（v3）")
    ap.add_argument("--p-hours", type=float, default=P_HOURS, help="--fact-items sample: 営業時間の類を入れる台本の割合")
    ap.add_argument("--ipadic", default=str(IPADIC_DIR))
    ap.add_argument("--tokenizer", default=str(TOKENIZER), help="この語彙に無い文字を含む名前は使わない")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-mem", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--no-guided", action="store_true", help="disable JSON-schema structured output")
    ap.add_argument("--guided-backend", default="auto", help="xgrammar | guidance | outlines | auto")
    ap.add_argument("--only-passing", action="store_true",
                    help="keep only scripts passing all checks; write the rest to <out>.rejected.jsonl")
    ap.add_argument("--probe-single", action="store_true", help="also time one request alone (tokens/s per stream)")
    ap.add_argument("--stats", help="write timing / quality summary json here")
    ap.add_argument("--prompts-out", help="also write the exact chat messages per item (jsonl)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:  # vLLM 0.11
        from vllm.sampling_params import GuidedDecodingParams as _GDP
        guided_kw = lambda schema: {"guided_decoding": _GDP(json=schema, backend=None if args.guided_backend == "auto" else args.guided_backend)}  # noqa: E731
    except ImportError:  # newer vLLM renamed it
        from vllm.sampling_params import StructuredOutputsParams as _SOP
        guided_kw = lambda schema: {"structured_outputs": _SOP(json=schema)}  # noqa: E731

    from .names import NamePool
    global KNOWN_SURNAMES
    pool = NamePool(args.ipadic, args.split, args.tokenizer if Path(args.tokenizer).exists() else None)
    KNOWN_SURNAMES = {n.surface for n in pool.surnames}
    businesses = EVAL_BUSINESSES if args.domains == "heldout" else BUSINESSES
    rng = random.Random(args.seed)
    scen = [args.scenario] * args.n if args.scenario != "mix" else [SCENARIOS[i % 3] for i in range(args.n)]
    ask_forms = {k: float(w) for k, w in (x.split(":") for x in args.ask_forms.split(",") if x)} or None
    assert not ask_forms or set(ask_forms) <= set(ASK_FORMS), ask_forms
    specs = [sample_spec(s, rng, pool, businesses, args.ask_facts, args.ask_first, args.fact_items, ask_forms, args.p_hours)
             for s in scen]
    msgs = [build_messages(s) for s in specs]

    llm_kw = dict(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager, seed=args.seed)
    if args.max_num_batched_tokens:
        llm_kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.guided_backend != "auto":
        llm_kw["guided_decoding_backend"] = args.guided_backend
    t_load = time.time()
    llm = LLM(**llm_kw)
    load_s = time.time() - t_load

    # Some chat templates (llm-jp-3.1 instruct4) replace the system message with a fixed sentence.
    # Detect that and put our instructions at the top of the user message instead.
    tok = llm.get_tokenizer()
    probe = tok.apply_chat_template([{"role": "system", "content": "SENTINEL_9431"}, {"role": "user", "content": "x"}],
                                    tokenize=False, add_generation_prompt=True)
    system_merged = "SENTINEL_9431" not in probe
    if system_merged:
        msgs = [[{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}] for m in msgs]
    n_prompt_tokens = [len(tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)) for m in msgs]

    params = []
    for i, s in enumerate(specs):
        kw = {} if args.no_guided else guided_kw(json_schema(s['turns_min'], s['turns_max']))
        params.append(SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                     presence_penalty=args.presence_penalty,
                                     repetition_penalty=args.repetition_penalty,
                                     max_tokens=args.max_tokens, seed=args.seed * 100003 + i, **kw))
    t0 = time.time()
    outs = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    gen_s = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tok, n_valid, n_ok, n_fact = 0, 0, 0, 0
    rej_f = open(str(out_path) + ".rejected.jsonl", "w") if args.only_passing else None
    with out_path.open("w") as f:
        for i, (s, o) in enumerate(zip(specs, outs)):
            c = o.outputs[0]
            n_tok += len(c.token_ids)
            rec = {"id": f"{s['scenario']}_{args.split}_{args.seed}_{i:05d}", "scenario": s["scenario"],
                   "split": args.split, "domains": args.domains, "seed": args.seed, "readings": s["readings"],
                   "model": args.model, "finish_reason": c.finish_reason, "n_output_tokens": len(c.token_ids), "raw_text": c.text}
            try:
                obj = sanitize(json.loads(c.text))
                n_valid += 1
                rec.update(system_prompt=obj.get("system_prompt"), facts=obj.get("facts"), turns=obj.get("turns"))
                rec["check"] = check_script(obj, s)
                n_ok += rec["check"]["ok"]
                n_fact += rec["check"]["fact_consistent"]
            except Exception as e:  # noqa: BLE001
                rec.update(raw=c.text, error=f"{type(e).__name__}: {e}")
            rec["spec"] = s
            dst = rej_f if (rej_f and not rec.get("check", {}).get("ok")) else f
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rej_f:
        rej_f.close()
    if args.prompts_out:
        with open(args.prompts_out, "w") as f:
            for s, m in zip(specs, msgs):
                f.write(json.dumps({"spec": s, "messages": m}, ensure_ascii=False) + "\n")
    single = None
    if args.probe_single:  # one request alone: per-stream decode speed (batch number above is aggregate)
        t1 = time.time()
        o1 = llm.chat([msgs[0]], [params[0]], use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0].outputs[0]
        single = round(len(o1.token_ids) / (time.time() - t1), 1)
    stats = {"model": args.model, "n": args.n, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
             "output_tokens": n_tok, "output_tokens_per_s": round(n_tok / gen_s, 1),
             "single_stream_tokens_per_s": single, "system_merged_into_user": system_merged,
             "prompt_tokens_max": max(n_prompt_tokens),
             "json_valid_rate": n_valid / args.n, "all_checks_pass_rate": n_ok / args.n,
             "fact_consistent_rate": n_fact / args.n, "args": vars(args)}
    print(json.dumps(stats, ensure_ascii=False))
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
