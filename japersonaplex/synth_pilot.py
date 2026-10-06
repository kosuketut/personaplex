"""パイプライン検証用の合成データ（店舗への問い合わせ電話）を作る。

台本はテンプレート、音声は pyopenjtalk（単一話者の HTS 音声を half_tone でずらしたもの）。
どちらも仮置きで、LLM 台本と多話者 TTS に差し替える前提。単語時刻はモーラ数で按分した近似値。

出力（--out 以下）:
    audio/<id>.wav   ステレオ 24 kHz（左 = agent、右 = user）
    words/<id>.json  時刻付き単語
    voices/*.wav     声プロンプト
    train.jsonl / heldout.jsonl   prepare.py 用マニフェスト
    flip.jsonl       評価用。同じ user 音声に対し、事実だけが違うプロンプト A/B と期待される答え
    flip/<id>.wav    評価用の user 音声（モノラル）
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyopenjtalk
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
SHOP_TYPES = {
    "パン屋": ["食パン", "クロワッサン", "メロンパン", "カレーパン"],
    "花屋": ["バラの花束", "ひまわり", "観葉植物", "カーネーション"],
    "書店": ["図書カード", "手帳", "カレンダー", "ブックカバー"],
    "クリーニング店": ["ワイシャツ", "コート", "スーツ", "毛布"],
    "自転車店": ["パンク修理", "ブレーキ調整", "ライト", "ヘルメット"],
    "喫茶店": ["コーヒー", "紅茶", "ナポリタン", "チーズケーキ"],
}
SHOP_NAMES = {
    "train": ["あおぞら", "さくら", "みどり", "ひかり", "やまびこ", "こもれび", "つばめ", "わかば", "はるかぜ",
              "しらゆき", "たんぽぽ", "ほしぞら", "あさひ", "みなと", "かえで", "すずらん"],
    "heldout": ["ゆうなぎ", "こはる", "あかつき", "いずみ"],
}
STAFF = {
    "train": ["田中", "鈴木", "佐藤", "高橋", "伊藤", "渡辺", "山本", "中村", "小林", "加藤", "吉田", "山田"],
    "heldout": ["松本", "井上", "木村", "清水"],
}
DAYS = ["月", "火", "水", "木", "金", "土", "日"]
GREETINGS = ["お電話ありがとうございます。{shop}の{staff}です。", "はい、{shop}、{staff}でございます。",
             "お電話ありがとうございます。{shop}、担当の{staff}です。"]
QA = {
    "hours": (["営業時間を教えてください。", "何時から何時まで開いていますか。", "今日は何時までやっていますか。"],
              ["{open}時から{close}時まで営業しております。", "営業時間は{open}時から{close}時までです。"]),
    "closed": (["定休日はいつですか。", "お休みの日を教えてください。"],
               ["定休日は{day}曜日です。", "毎週{day}曜日にお休みをいただいております。"]),
    "price": (["{item}はいくらですか。", "{item}の値段を教えてください。"],
              ["{item}は{price}円です。", "{item}は{price}円でございます。"]),
    "name": (["すみません、お名前をもう一度お願いします。", "担当の方のお名前を教えてください。"],
             ["{staff}と申します。", "はい、{staff}です。"]),
}
CLOSINGS = [("わかりました。ありがとうございます。", "またのお電話をお待ちしております。"),
            ("ありがとうございました。", "こちらこそ、ありがとうございました。失礼いたします。")]
VOICE_TEXTS = ["本日はお問い合わせいただき、ありがとうございます。ご用件をどうぞ。",
               "いつもご利用ありがとうございます。何かお困りのことはございますか。"]
AGENT_VOICES = [(-4.0, 0.95), (-2.0, 1.0), (0.0, 1.0), (2.0, 1.05), (4.0, 1.0)]  # (half_tone, speed)


def tts(text: str, half_tone: float, speed: float) -> np.ndarray:
    wav, sr = pyopenjtalk.tts(text, speed=speed, half_tone=half_tone)
    wav = resample_poly(wav / 32768.0, SR, sr).astype(np.float32)
    return wav * (0.3 / max(1e-6, np.abs(wav).max()))


def word_times(text: str, wav: np.ndarray, offset: float) -> list[dict]:
    """発話内の単語開始時刻を、有音区間をモーラ数で按分して近似する。"""
    voiced = np.flatnonzero(np.abs(wav) > 0.01)
    begin, end = voiced[0] / SR, voiced[-1] / SR
    words = [(w["string"], max(1, w["mora_size"])) for w in pyopenjtalk.run_frontend(text)]
    total = sum(m for _, m in words)
    out, acc = [], 0
    for string, mora in words:
        start = begin + (end - begin) * acc / total
        acc += mora
        out.append({"word": string, "start": round(offset + start, 3),
                    "end": round(offset + begin + (end - begin) * acc / total, 3)})
    return out


def sample_facts(rng: random.Random, split: str) -> dict:
    shop_type = rng.choice(list(SHOP_TYPES))
    return {
        "shop": rng.choice(SHOP_NAMES[split]) + shop_type, "type": shop_type, "staff": rng.choice(STAFF[split]),
        "open": rng.randint(7, 11), "close": rng.randint(17, 22), "day": rng.choice(DAYS),
        "item": rng.choice(SHOP_TYPES[shop_type]), "price": rng.randrange(200, 3000, 50),
    }


def prompt_text(facts: dict, rng: random.Random) -> str:
    info = ["営業時間は{open}時から{close}時まで。", "定休日は{day}曜日。", "{item}は{price}円。"]
    rng.shuffle(info)
    return ("あなたは{shop}という{type}で働いています。名前は{staff}です。情報：" + "".join(info)).format(**facts)


def render(turns: list[tuple[str, str]], agent_voice, user_voice, rng: random.Random, tail: float):
    """turns: [(speaker, text)]。重なりなしで順に並べる。戻り値は (stereo [T, 2], words)。"""
    cursor, pieces, words = 0.5, [], []
    for speaker, text in turns:
        wav = tts(text, *(agent_voice if speaker == "A" else user_voice))
        pieces.append((speaker, int(cursor * SR), wav))
        words += [dict(w, speaker=speaker) for w in word_times(text, wav, cursor)]
        cursor += len(wav) / SR + rng.uniform(0.25, 0.7)
    stereo = np.zeros((int((cursor + tail) * SR), 2), dtype=np.float32)
    for speaker, begin, wav in pieces:
        stereo[begin:begin + len(wav), 0 if speaker == "A" else 1] += wav
    return stereo, words


def make_dialogue(rng: random.Random, split: str) -> dict:
    facts = sample_facts(rng, split)
    kinds = rng.sample(list(QA), rng.randint(2, 3))
    turns = [("A", rng.choice(GREETINGS).format(**facts))]
    for kind in kinds:
        questions, answers = QA[kind]
        turns += [("B", rng.choice(questions).format(**facts)), ("A", rng.choice(answers).format(**facts))]
    closing = rng.choice(CLOSINGS)
    turns += [("B", closing[0]), ("A", closing[1])]
    return {"facts": facts, "turns": turns, "text_prompt": prompt_text(facts, rng)}


def spoken(text: str) -> str:
    """単語時刻と同じ表記（数字は漢数字）にそろえる。"""
    return "".join(w["string"] for w in pyopenjtalk.run_frontend(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/pilot")
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--flip", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    for sub in ("audio", "words", "voices", "flip"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for v, voice in enumerate(AGENT_VOICES):
        for t, text in enumerate(VOICE_TEXTS):
            sf.write(out / "voices" / f"voice{v}_{t}.wav", tts(text, *voice), SR, subtype="PCM_16")

    def user_voice(agent_index: int):
        half_tone = rng.choice([h for h in (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0) if h != AGENT_VOICES[agent_index][0]])
        return half_tone, rng.choice([0.95, 1.0, 1.1])

    for split, count in (("train", args.train), ("heldout", args.heldout)):
        with open(out / f"{split}.jsonl", "w") as manifest:
            for i in range(count):
                dialogue = make_dialogue(rng, split)
                v = rng.randrange(len(AGENT_VOICES))
                stereo, words = render(dialogue["turns"], AGENT_VOICES[v], user_voice(v), rng, tail=1.0)
                name = f"{split}_{i:05d}"
                sf.write(out / "audio" / f"{name}.wav", stereo, SR, subtype="PCM_16")
                (out / "words" / f"{name}.json").write_text(json.dumps(words, ensure_ascii=False))
                manifest.write(json.dumps({
                    "id": name, "audio": f"audio/{name}.wav", "words": f"words/{name}.json",
                    "text_prompt": dialogue["text_prompt"], "voice_prompt": f"voices/voice{v}_{rng.randrange(2)}.wav",
                    "facts": dialogue["facts"],
                }, ensure_ascii=False) + "\n")

    # flip 評価: 未知の店名・担当者で、営業時間（偶数番）か価格（奇数番）だけが違うプロンプト A/B
    with open(out / "flip.jsonl", "w") as manifest:
        for i in range(args.flip):
            facts_a = sample_facts(rng, "heldout")
            facts_b = dict(facts_a)
            kind = "hours" if i % 2 == 0 else "price"
            if kind == "hours":
                facts_b["close"] = rng.choice([c for c in range(17, 23) if c != facts_a["close"]])
                expect = [spoken(f"{f['close']}時") for f in (facts_a, facts_b)]
            else:
                facts_b["price"] = rng.choice([p for p in range(200, 3000, 50) if p != facts_a["price"]])
                expect = [spoken(f"{f['price']}円") for f in (facts_a, facts_b)]
            v = rng.randrange(len(AGENT_VOICES))
            question = rng.choice(QA[kind][0]).format(**facts_a)
            greeting = rng.choice(GREETINGS).format(**facts_a)
            stereo, _ = render([("A", greeting), ("B", question)], AGENT_VOICES[v], user_voice(v), rng, tail=9.0)
            name = f"flip_{i:03d}"
            sf.write(out / "flip" / f"{name}.wav", stereo[:, 1], SR, subtype="PCM_16")
            # A/B で情報の並び順は同じにし、違いを事実 1 点だけにする
            manifest.write(json.dumps({
                "id": name, "user_audio": f"flip/{name}.wav", "voice_prompt": f"voices/voice{v}_0.wav",
                "kind": kind, "question": question,
                "prompts": [prompt_text(facts, random.Random(i)) for facts in (facts_a, facts_b)],
                "expect": expect,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
