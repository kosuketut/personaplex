"""datagen.scripts_llm の v3（項目の指定と、user が項目名で直接聞く検査）。"""
import random

from japersonaplex.datagen.scripts_llm import BUSINESSES, check_script, sample_fact_items


def _spec(**kw):
    spec = {"scenario": "service", "turns_min": 6, "turns_max": 10, "interruption": False, "store_name": "こはる喫茶店",
            "agent_name": "松本", "user_name": "田中", "fact_items": ["ナポリタンの値段", "席数"], "n_extra": 1,
            "ask_items": ["ナポリタンの値段"], "ask_first": True}
    spec.update(kw)
    return spec


def _script(first_user: str, answer: str):
    return {
        "facts": {"ナポリタンの値段": "1050円", "席数": "20席", "定休日": "月曜日"},
        "system_prompt": "あなたはこはる喫茶店という喫茶店で働いています。名前は松本です。情報：ナポリタンの値段は1050円。席数は20席。定休日は月曜日。",
        "turns": [
            {"speaker": "agent", "text": "はい、こはる喫茶店、松本でございます。", "type": "utterance"},
            {"speaker": "user", "text": first_user, "type": "utterance"},
            {"speaker": "agent", "text": answer, "type": "utterance"},
            {"speaker": "user", "text": "席数って何席ありますか。", "type": "utterance"},
            {"speaker": "agent", "text": "20席でございます。", "type": "utterance"},
            {"speaker": "user", "text": "わかりました。ありがとうございます。", "type": "utterance"},
        ],
    }


def _ask_issues(check):
    return [i for i in check["issues"] if i.startswith(("ask_", "fact_items"))]


def test_asked_first_and_answered_passes():
    check = check_script(_script("あの、ナポリタンの値段っておいくらですか。", "ナポリタンは1050円です。"), _spec())
    assert _ask_issues(check) == []


def test_not_answered_in_next_turn_is_rejected():
    check = check_script(_script("あの、ナポリタンの値段っておいくらですか。", "少々お待ちください。"), _spec())
    assert "ask_item_not_answered=ナポリタンの値段" in check["issues"]


def test_asked_later_fails_only_when_first_is_required():
    s = _script("もしもし、ちょっと聞きたいんですけど。", "はい、どうぞ。")
    s["turns"][3:5] = [{"speaker": "user", "text": "ナポリタンの値段って。", "type": "utterance"},
                       {"speaker": "agent", "text": "1050円です。", "type": "utterance"}]
    assert "ask_first_missing" in check_script(s, _spec())["issues"]
    assert _ask_issues(check_script(s, _spec(ask_first=False))) == []


def test_missing_item_and_non_numeric_value():
    s = _script("ナポリタンの値段は。", "1050円です。")
    s["facts"] = {"ナポリタンの値段": "お手頃です", "定休日": "月曜日", "駐車場": "3台"}
    issues = check_script(s, _spec())["issues"]
    assert "fact_items_missing=['席数']" in issues
    assert "ask_item_not_numeric=ナポリタンの値段" in issues


def test_unrequested_hours_item_is_rejected():
    s = _script("あの、ナポリタンの値段っておいくらですか。", "ナポリタンは1050円です。")
    s["facts"]["営業時間"] = "7時から22時まで"
    s["system_prompt"] += "営業時間は7時から22時まで。"
    assert "unrequested_hours_item=['営業時間']" in check_script(s, _spec())["issues"]
    assert _ask_issues(check_script(s, _spec(fact_items=["ナポリタンの値段", "席数", "営業時間"]))) == []


def test_v4_paraphrase_must_not_use_item_name():
    spec = _spec(ask_forms=["paraphrase"])
    named = _script("あの、ナポリタンの値段っておいくらですか。", "ナポリタンは1050円です。")
    assert "ask_form_wrong=ナポリタンの値段:paraphrase" in check_script(named, spec)["issues"]
    para = _script("あの、ナポリタンっておいくらですか。", "ナポリタンは1050円です。")
    assert _ask_issues(check_script(para, spec)) == []


def test_v4_indirect_answered_with_value():
    spec = _spec(ask_items=["席数"], ask_forms=["indirect"], ask_first=False)
    s = _script("あの、明日10人で行きたいんですけど、入れますかね。", "はい、20席ございますので大丈夫です。")
    s["turns"][3:5] = [{"speaker": "user", "text": "よかったです。", "type": "utterance"},
                       {"speaker": "agent", "text": "お待ちしております。", "type": "utterance"}]
    assert _ask_issues(check_script(s, spec)) == []
    s["turns"][2]["text"] = "はい、大丈夫です。"
    assert "ask_item_not_answered=席数" in check_script(s, spec)["issues"]


def test_sample_fact_items_forms():
    rng = random.Random(0)
    forms = {"name": 0.3, "paraphrase": 0.25, "situational": 0.25, "indirect": 0.2}
    for _ in range(200):
        _, hint = rng.choice(list(BUSINESSES.values()))
        s = sample_fact_items(rng, hint, 2, 0.5, forms, 0.55)
        assert len(s["ask_forms"]) == len(s["ask_items"]) and set(s["ask_forms"]) <= set(forms)


def test_sample_fact_items_bounds():
    rng = random.Random(0)
    for _ in range(500):
        _, hint = rng.choice(list(BUSINESSES.values()))
        s = sample_fact_items(rng, hint, 2, 0.5)
        assert 3 <= len(s["fact_items"]) + s["n_extra"] <= 6
        assert set(s["ask_items"]) <= set(s["fact_items"]) and len(s["ask_items"]) <= 2
        assert not s["ask_first"] or s["ask_items"]
