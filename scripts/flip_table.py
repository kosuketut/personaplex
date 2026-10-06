"""flip 評価の結果（eval_flip の results.json と asr_flip の results_asr.json）を、セットごとの表にする。

    python scripts/flip_table.py outputs/pilot40_lora_v1/eval_flip40_step*
    python scripts/flip_table.py --breakdown outputs/pilot40_lora_v1/eval_flip40v2_step800

セットは id の最後の「_番号」を除いた部分（flip40 の h_name_003 なら h_name、pilot の flip_003 なら flip）。
列: テキストの正答（A/B とも正しく、取り違えなし）、片側ずつの正答、店名と担当者名（A/B 両方で言えたか）、音声の ASR での正答。
--breakdown: A・B の回答を 1 つずつ、期待値の単位（時・円・日など）ごとに「正しい / 反対側の値 / 同じ単位の別の数字 /
触れていない」に分けて数える（テキスト出力で）。
flip40 v2 から作ったセット（eval_flip40v2*、eval_flip40v3*）では、不備のある 5 問（EXCLUDE_V2）を外して数える。--all で外さない。
「評価専用」の行は h_name・h_domain・h_voice の合計。
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from japersonaplex.scoring import mentions, normalize_numbers  # noqa: E402

KINDS = ("correct", "other_side", "wrong_number", "no_mention")
# flip40 v2 の不備のある問。期待値が質問の答えになっていないか、期待値の切り出しを誤った。flip40v2 から作ったセット
# （flip40v2_t9、flip40v2u2、質問の形だけを替えた flip40v3_*）も同じ id で同じ問なので一緒に外す
EXCLUDE_V2 = {
    "h_name_015": "学生割引日を聞いて、期待値が料金（1000円）",
    "h_name_037": "体験授業（無料）を聞いて、期待値が予約の期限（1週）",
    "h_domain_025": "予防接種の料金（3500円）を聞いて、期待値が「1回につき」の回数",
    "h_voice_011": "休館日「第1・第3月曜日」から「1・」を切り出した",
    "h_voice_023": "「0.5パーセント」から「0.」を切り出した",
}
INCLUDE_ALL = False


def load_rows(d: Path, name: str = "results.json") -> list[dict]:
    rows = json.loads((d / name).read_text())["rows"]
    if d.name.startswith(("eval_flip40v2", "eval_flip40v3")) and not INCLUDE_ALL:
        rows = [r for r in rows if r["id"] not in EXCLUDE_V2]
    return rows


def classify(expect: list[str], text: str, side: int) -> str:
    unit = re.sub(r"\d", "", normalize_numbers(expect[side]))
    if mentions(expect[side], text):
        return "correct"
    if mentions(expect[1 - side], text):
        return "other_side"
    if re.search(r"\d+" + re.escape(unit), normalize_numbers(text)):
        return "wrong_number"
    return "no_mention"


def breakdown(dirs: list[Path]):
    print("| 評価 | 単位 | 回答数 | " + " | ".join(KINDS) + " |")
    print("|---|---|---|" + "---|" * len(KINDS))
    for d in dirs:
        by_unit: dict[str, Counter] = defaultdict(Counter)
        for r in load_rows(d):
            for side in (0, 1):
                unit = re.sub(r"\d", "", normalize_numbers(r["expect"][side]))[:1]
                by_unit[unit][classify(r["expect"], r["text"][side], side)] += 1
        total = sum(by_unit.values(), Counter())
        for unit, c in [("計", total)] + sorted(by_unit.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"| {d.name} | {unit} | {sum(c.values())} | " + " | ".join(str(c[k]) for k in KINDS) + " |")


def main(args: list[str]):
    global INCLUDE_ALL
    if "--all" in args:
        INCLUDE_ALL = True
        args = [a for a in args if a != "--all"]
    if args and args[0] == "--breakdown":
        breakdown(list(map(Path, args[1:])))
        return
    print("| 評価 | セット | n | テキスト | 片側 | 店名 | 担当者 | ASR |")
    print("|---|---|---|---|---|---|---|---|")
    for d in map(Path, args):
        rows = load_rows(d)
        asr = {r["id"]: r for r in load_rows(d, "results_asr.json")} if (d / "results_asr.json").exists() else {}
        groups = defaultdict(list)
        for r in rows:
            groups[r["id"].rsplit("_", 1)[0]].append(r)
        if sum(name.startswith("h_") for name in groups) > 1:
            groups["評価専用"] = [r for r in rows if r["id"].startswith("h_")]
        for name, rs in groups.items():
            n = len(rs)
            asr_pass = f"{sum(asr[r['id']]['pass'] for r in rs if r['id'] in asr)}/{n}" if asr else "-"
            count = lambda key: sum(bool(r.get(key)) for r in rs)  # 古い results.json には無い列がある
            print(f"| {d.name} | {name} | {n} | {count('pass')}/{n} | {count('a_ok') + count('b_ok')}/{2 * n} | "
                  f"{count('shop_ok')}/{n} | {count('staff_ok')}/{n} | {asr_pass} |")


if __name__ == "__main__":
    main(sys.argv[1:])
