"""flip 評価の質問を、項目名を言わない形で台本 LLM に書かせる（make_flip.py --questions に渡す）。

flip40 v2 の質問は「{項目}を教えてもらえますか。」のように項目名で直接聞く形だけで、v3 の台本（user が項目名で聞く）に
近い。形の違う質問でも答えられるかを測るため、同じ問（対話・プロンプト・期待値）について 2 つの形を作る。
- paraphrase: 項目名を使わず、言い換えて直接聞く短い質問
- situational: 客の事情を一言添えてから聞く、少し長めの質問（項目名は使わない）
どちらも、正しく答えるには期待値（「10時」など）を言う必要がある聞き方にする。3 以上の数は質問に入れない（答えを写せないように。
「一度」「二人」などは許す）。期待値の数字そのものは質問に書かない。条件を満たす候補を、同じ LLM に「この質問に正しく
答えると期待値が必ず出るか」を判定させ（「何時まで」と聞いて開店時刻を期待する、のような食い違いを落とす）、通った最初の候補を使う。

    data/envs/llm/bin/python -m japersonaplex.datagen.flip_questions \
        --flip data/datagen/pilot40/flip40v2/flip.jsonl --out data/datagen/pilot40/flip40v2_questions.jsonl
"""
import argparse
import json
import re

from .scripts_llm import DEFAULT_MODEL, normalize_numbers, numbers_in

FORMS = {"paraphrase": 40, "situational": 80}  # 列名 -> 最大の文字数
SYSTEM = "あなたは日本語の音声対話モデルの評価データを作る人です。電話の客の話し言葉の質問を書きます。"
SCHEMA = {"type": "object", "properties": {f: {"type": "string"} for f in FORMS}, "required": list(FORMS),
          "additionalProperties": False}
JUDGE_SCHEMA = {"type": "object", "properties": {"contains": {"type": "boolean"}}, "required": ["contains"],
                "additionalProperties": False}


def fact_of(row: dict) -> tuple[str, str, str]:
    """(業種, 項目の値, 期待値) をプロンプト A から取り出す。"""
    prompt = row["prompts"][0]
    business = re.search(r"という(.+?)で働いています", prompt).group(1)
    value = re.search(re.escape(row["kind"]) + r"は(.+?)。", prompt).group(1)
    return business, value, row["expect"][0]


def user_message(row: dict) -> str:
    business, value, expect = fact_of(row)
    key = row["kind"]
    banned = f"「{key}」という語を"
    return (f"店・施設: {business}\n店の情報: {row['prompts'][0].split('情報：', 1)[-1]}\n"
            f"対象の項目: 「{key}」（値: {value}）\n\n"
            f"客がこの項目の値を電話で聞く質問を 2 つ書いてください。\n"
            f"- paraphrase: {banned}使わず、言い換えて直接聞く短い質問（{FORMS['paraphrase']}字以内）\n"
            f"- situational: 客の事情（いつ、誰と、何のためにか）を一言添えてから聞く、少し長めの質問（{FORMS['situational']}字以内）。"
            f"{banned}使わない\n"
            f"どちらも、正しい答えに必ず「{expect}」が含まれる聞き方にする（「はい」「いいえ」だけで答えられる質問にしない）。"
            f"話し言葉で、「えっと」「あの」などを入れてよい。数字（漢数字も）、英字、記号は使わない。\n"
            'JSON で {"paraphrase": "...", "situational": "..."} を出力する。')


def judge_message(row: dict, question: str) -> str:
    """書いた質問に正しく答えると期待値が出るか（「何時まで」と聞いて開店時刻を期待する、のような食い違いを落とす）。"""
    return (f"店の情報: {row['prompts'][0].split('情報：', 1)[-1]}\n客の質問: 「{question}」\n\n"
            f"店員がこの情報だけを使ってこの質問に正しく答えるとき、その答えに「{row['expect'][0]}」は必ず含まれますか。"
            '含まれるなら {"contains": true}、含まれないか、含まれなくても正しい答えになりうるなら {"contains": false} を出力する。')


def problems(row: dict, form: str, text: str) -> list[str]:
    key = row["kind"]
    out = []
    if not text.strip():
        out.append("empty")
    if key in text:
        out.append("key")
    # 「一度」「二人」のような 2 以下の数は許す（scripts_llm の検査と同じ考え方）
    if any(int(n) > 2 for n in re.findall(r"\d+", normalize_numbers(text))) or re.search(r"[A-Za-z]", text):
        out.append("number_or_latin")
    # 期待値の数字（A・B どちらも）を質問に書かない（「2週間前までって言われたから」のような答えの写し）
    if numbers_in(text) & (numbers_in(row["expect"][0]) | numbers_in(row["expect"][1])):
        out.append("leaks_answer")
    if len(text) > FORMS[form]:
        out.append("long")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=8,
                    help="1 問あたりの候補の数。検査と LLM の判定（答えに期待値が出るか）を通る最初の候補を使う")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    rows = [json.loads(line) for line in open(args.flip)]
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096, seed=args.seed)
    params = SamplingParams(n=args.samples, temperature=0.8, top_p=0.9, max_tokens=300, seed=args.seed,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(r)}] for r in rows]
    outs = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    cands = []  # 問ごとの候補（dict）
    for o in outs:
        cs = []
        for c in o.outputs:
            try:
                cs.append(json.loads(c.text))
            except json.JSONDecodeError:
                continue
        cands.append(cs)
    # 検査を通った候補を、答えに期待値が出るかで判定する（温度 0）
    todo = [(i, form, c[form]) for i, (r, cs) in enumerate(zip(rows, cands)) for form in FORMS for c in cs
            if c.get(form) and not problems(r, form, c[form])]
    judge = SamplingParams(temperature=0.0, max_tokens=20, guided_decoding=GuidedDecodingParams(json=JUDGE_SCHEMA))
    jmsgs = [[{"role": "user", "content": judge_message(rows[i], q)}] for i, _, q in todo]
    verdicts = llm.chat(jmsgs, judge, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}) if todo else []
    passed = {}
    for (i, form, q), v in zip(todo, verdicts):
        try:
            ok = json.loads(v.outputs[0].text)["contains"]
        except (json.JSONDecodeError, KeyError):
            ok = False
        if ok:
            passed.setdefault((i, form), q)  # 候補の順で最初のもの
    counts = {f: {"no_candidate": 0, "judge_rejected": 0} for f in FORMS}
    with open(args.out, "w") as f:
        for i, (r, cs) in enumerate(zip(rows, cands)):
            rec = {"id": r["id"], "kind": r["kind"], "expect": r["expect"][0]}
            for form in FORMS:
                rec[form] = passed.get((i, form))  # 通る候補が無い問はその形では使わない
                if rec[form] is None:
                    checked = [c[form] for c in cs if c.get(form) and not problems(r, form, c[form])]
                    counts[form]["judge_rejected" if checked else "no_candidate"] += 1
                    rec[f"{form}_rejected"] = (checked or [c.get(form, "") for c in cs])[:2]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "unused": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
