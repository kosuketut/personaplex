"""TTS で作った発話ごとの音声に、単語の時刻を付ける（強制アライメント）。あわせて一次の品質検査をする。

data/bakeoff/align/align_words.py の移植。違いは、台本の文ではなく units.utterance_units の単位列を受け取ること
（固有名詞は 1 単位で、読みは辞書の値）。合成音声の正解で、単語の開始が ±1 フレーム（80 ms）に入る割合は
アンサンブル（ens）で 99.5%、CTC のみ（ctc）で 95.6〜96.6%（bakeoff の set A/B）。

環境は data/envs/align（torch 2.8.0+cu128、transformers 4.57.6、qwen-asr、faster-whisper）。

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 data/envs/align/bin/python -m japersonaplex.datagen.align --out data/datagen/m1 [--asr]

入力は <out>/tts/utts.jsonl（同じ key は attempt が最大の行）。出力は <out>/align/words.jsonl に 1 発話 1 行:
    {"key", "attempt", "words": [{"word", "start", "end", "punct", "entity"}], "qc": {...}}
時刻は発話の wav の先頭からの秒。qc:
    unexplained_speech_s  CTC が音声を聞いているのに台本の文字が割り当たらない秒数（TTS の付け足し・繰り返し）
    max_onset_spread      アンサンブルの 3 手法で単語の開始がばらついた最大幅
    mora_per_s            話速（data_spec §5.3 の上限 9）
    cer, cer_raw          --asr のとき。faster-whisper medium（GPU、beam 5）の書き起こしと台本の読みのカナ CER。
                          cer はフィラーを両側から除いた値（whisper がフィラーを書き起こさないため）。固有名詞は数えない
    entities              --asr のとき。固有名詞ごとに、読みとひらがな CTC の書き起こしの部分文字列の編集距離と合否
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

CTC_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
QWEN_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"


def local_or_repo(repo: str) -> str:
    """キャッシュにあればそのローカルパス（tts.model_path と同じ考え方）。リポジトリ名のまま渡すと読み込みのたびに HF の API に
    問い合わせるので、シャードを 30 本同時に始めたとき 429 Too Many Requests で落ちた（ジョブ 10629）。"""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001  キャッシュに無ければ HF から取る
        return repo


def whisper_model(device: str, compute_type: str):
    """faster-whisper の medium。キャッシュにあればネットワークに出ない。"""
    from faster_whisper import WhisperModel
    try:
        return WhisperModel("medium", device=device, compute_type=compute_type, local_files_only=True)
    except Exception:  # noqa: BLE001
        return WhisperModel("medium", device=device, compute_type=compute_type)
# CTC の放出は音素の始まりより遅れるので、予測した開始時刻から引く（bakeoff で合成音声の正解に合わせた値）
ONSET_SHIFT = {"ctc_surf": 0.020, "ctc_hira": 0.010, "qwen": 0.0}
FRAME = 0.02  # wav2vec2 のフレーム間隔（16 kHz）
_VOWEL = {c: v for row, v in [("あかさたなはまやらわがざだばぱぁゃ", "あ"), ("いきしちにひみりぎじぢびぴぃ", "い"),
                               ("うくすつぬふむゆるぐずづぶぷぅゅゔ", "う"), ("えけせてねへめれげぜでべぺぇ", "え"),
                               ("おこそとのほもよろをごぞどぼぽぉょ", "お")] for c in row}


def _expand_long(h: str, prev: str = "") -> str:
    """長音記号を直前の母音に置き換える（CTC の語彙に「ー」が無い場合）。"""
    out: list[str] = []
    for c in h:
        out.append(_VOWEL.get(out[-1] if out else prev[-1:], "う") if c == "ー" else c)
    return "".join(out)


def ctc_viterbi(logp: np.ndarray, tokens: list[int], blank: int):
    """CTC の Viterbi 整列。トークンごとの (最初, 最後) の放出フレームと、空白状態にいたフレームのマスクを返す。"""
    T, L = logp.shape[0], len(tokens)
    if L == 0:
        return [], np.ones(T, dtype=bool)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    neg = -1e30
    dp = np.full(S, neg)
    dp[0], dp[1] = logp[0, blank], logp[0, ext[1]]
    bp = np.zeros((T, S), dtype=np.int8)
    skip = np.zeros(S, dtype=bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    idx = np.arange(S)
    for t in range(1, T):
        b = np.concatenate([[neg], dp[:-1]])
        c = np.where(skip, np.concatenate([[neg, neg], dp[:-2]]), neg)
        st = np.stack([dp, b, c])
        arg = st.argmax(0)
        dp = st[arg, idx] + logp[t, ext]
        bp[t] = arg
    s = int(S - 1 if dp[S - 1] >= dp[S - 2] else S - 2)
    spans = [[-1, -1] for _ in range(L)]
    in_blank = np.zeros(T, dtype=bool)
    for t in range(T - 1, -1, -1):
        in_blank[t] = s % 2 == 0
        if s % 2 == 1:
            k = s // 2
            spans[k][1] = t if spans[k][1] < 0 else spans[k][1]
            spans[k][0] = t
        s -= int(bp[t, s])
    return [tuple(x) for x in spans], in_blank


class Aligner:
    def __init__(self, method: str = "ens", device: str | None = None):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ctc_path = local_or_repo(CTC_REPO)
        self.fe = AutoFeatureExtractor.from_pretrained(ctc_path)
        self.ctc = Wav2Vec2ForCTC.from_pretrained(ctc_path).to(self.device).eval()
        if Path(ctc_path).is_dir():
            self.vocab = json.loads((Path(ctc_path) / "vocab.json").read_text())
        else:
            from huggingface_hub import hf_hub_download
            self.vocab = json.loads(Path(hf_hub_download(CTC_REPO, "vocab.json")).read_text())
        self.blank = self.ctc.config.pad_token_id
        if method == "ens":
            from qwen_asr import Qwen3ForcedAligner
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            self.qwen = Qwen3ForcedAligner.from_pretrained(local_or_repo(QWEN_REPO), dtype=dtype, device_map=self.device)

    def _ctc_words(self, logp, units, mode):
        ids, owner, prev = [], [], ""
        for wi, u in enumerate(units):
            if u["punct"]:
                continue
            s = u["surf"] if mode == "surf" else (
                u["hira"] if all(c in self.vocab for c in u["hira"]) else _expand_long(u["hira"], prev))
            prev = u["hira"] or prev
            for c in s:
                if c in self.vocab:
                    ids.append(self.vocab[c])
                    owner.append(wi)
        spans, in_blank = ctc_viterbi(logp, ids, self.blank)
        unexplained = float(np.sum(in_blank & (np.exp(logp[:, self.blank]) < 0.5)) * FRAME)
        res: list = [None] * len(units)
        for (a, b), wi in zip(spans, owner):
            if res[wi] is None:
                res[wi] = [a * FRAME, (b + 1) * FRAME]
            else:
                res[wi][1] = (b + 1) * FRAME
        return res, unexplained

    def _qwen_words(self, wav16, units):
        proc = self.qwen.aligner_processor
        keep = [i for i, u in enumerate(units) if not u["punct"] and proc.clean_token(u["surf"])]
        toks = [proc.clean_token(units[i]["surf"]) for i in keep]
        proc.tokenize_japanese = lambda _t: toks
        items = self.qwen.align(audio=(wav16, 16000), text="x", language="Japanese")[0].items
        res: list = [None] * len(units)
        for i, it in zip(keep, items):
            res[i] = [float(it.start_time), float(it.end_time)]
        return res

    @torch.inference_mode()
    def align(self, wav: np.ndarray, sr: int, units: list[dict]) -> tuple[list[dict], dict]:
        # 数字の単位は表記が算用数字なので、音声と合わせるのは漢数字の spoken（units._restore_numbers）
        units = [dict(u, surf=unicodedata.normalize("NFKC", u.get("spoken", u["word"]))) for u in units]
        wav16 = resample_poly(wav, 16000, sr).astype(np.float32) if sr != 16000 else wav.astype(np.float32)
        dur = len(wav16) / 16000
        iv = self.fe(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        logp = self.ctc(iv).logits[0].float().log_softmax(-1).cpu().numpy()
        surf, unexplained = self._ctc_words(logp, units, "surf")
        cands = [(surf, ONSET_SHIFT["ctc_surf"])]
        if self.method == "ens":
            cands.append((self._ctc_words(logp, units, "hira")[0], ONSET_SHIFT["ctc_hira"]))
            cands.append((self._qwen_words(wav16, units), ONSET_SHIFT["qwen"]))
        out, last_on, last_end, spreads = [], 0.0, 0.0, []
        for wi, u in enumerate(units):
            st = [c[wi][0] - sh for c, sh in cands if c[wi] is not None]
            en = [c[wi][1] for c, sh in cands if c[wi] is not None]
            if u["punct"] or not st:
                s = e = max(last_on, last_end)
            else:
                s = float(np.clip(np.median(st), last_on, dur))  # 開始時刻は単調にする
                e = float(np.clip(np.median(en), s, dur))
                spreads.append(max(st) - min(st))
                last_on, last_end = s, e
            out.append({"word": u["word"], "start": round(s, 3), "end": round(e, 3), "punct": u["punct"],
                        "entity": u["entity"], **({"spoken": u["spoken"]} if "spoken" in u else {})})
        qc = {"unexplained_speech_s": round(unexplained, 3),
              "max_onset_spread": round(float(max(spreads)), 3) if spreads else 0.0,
              "n_spread_gt_120ms": int(sum(x > 0.12 for x in spreads))}
        return out, qc


def latest_rows(path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for line in open(path):
        r = json.loads(line)
        if r["key"] not in rows or r["attempt"] >= rows[r["key"]]["attempt"]:
            rows[r["key"]] = r
    return list(rows.values())


def _ensure_cuda_libs():
    """faster-whisper（ctranslate2）は cublas/cudnn を LD_LIBRARY_PATH から探すので、torch が持っている版を足して実行し直す。"""
    import nvidia.cublas
    import nvidia.cudnn
    libs = [os.path.join(m.__path__[0], "lib") for m in (nvidia.cublas, nvidia.cudnn)]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if not all(p in cur for p in libs):
        env = dict(os.environ, LD_LIBRARY_PATH=":".join(libs + ([cur] if cur else [])))
        os.execve(sys.executable, [sys.executable, "-m", "japersonaplex.datagen.align", *sys.argv[1:]], env)


# whisper はフィラーや言いよどみを書き起こさないことが多い（M1 の 12 対話で CER の不合格の大半がこれだった）。
# 内容の誤りだけを見るため、CER は両側からこれらの語を除いて測る
# 「はい」も、発話の頭にあると whisper が書き起こさないことが多かった（M2 の 1,976 発話）
FILLERS = {"えっと", "えーと", "えーっと", "えー", "え", "えっ", "あの", "あのー", "うーん", "うん", "うんうん", "あー", "あ",
           "あっ", "おお", "おー", "おっ", "まあ", "ええ", "ん", "へえ", "へー", "ふーん", "はい", "はいはい"}
WILDCARD = "*"
HIRAGANA_CTC = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"


_ROW = {v: set(row) for v, row in [("ア", "アカサタナハマヤラワガザダバパァャ"), ("イ", "イキシチニヒミリギジヂビピィ"),
                                      ("ウ", "ウクスツヌフムユルグズヅブプゥュ"), ("エ", "エケセテネヘメレゲゼデベペェ"),
                                      ("オ", "オコソトノホモヨロヲゴゾドボポォョ")]}
_DEVOICE = str.maketrans("ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ", "カキクケコサシスセソタチツテトハヒフヘホハヒフヘホウ")


def _canon(kana: str) -> str:
    """長音の書き方の揺れ（トウ／トー、ケイ／ケー、ニイ／ニー）をそろえ、語末の促音を落とす。"""
    out: list[str] = []
    for c in kana:
        prev = out[-1] if out else ""
        if prev and ((c == "ウ" and prev in _ROW["オ"]) or (c == "イ" and prev in _ROW["エ"])
                     or (c in _ROW and c != "ン" and prev in _ROW[c])):
            c = "ー"
        out.append(c)
    return "".join(out).rstrip("ッ")


def _loose(kana: str) -> str:
    """名前の照合用。ひらがな CTC は清音と濁音・半濁音をよく取り違えるので、濁点を外してから比べる。"""
    return _canon(kana).translate(_DEVOICE)


def wildcard_edit_distance(ref: str, hyp: str) -> int:
    """編集距離。ref の WILDCARD は hyp の任意の部分文字列（空も含む）と無料で対応する。"""
    d = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        if ref[i - 1] == WILDCARD:
            d = np.minimum.accumulate(d)
            continue
        prev, d[0] = d[0], d[0] + 1
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return int(d[len(hyp)])


def substring_distance(pat: str, txt: str) -> int:
    """pat を txt のどこかの部分文字列に合わせたときの最小編集距離。"""
    return wildcard_edit_distance(WILDCARD + pat + WILDCARD, txt)


def _to_kata(h: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in h)


class AsrCheck:
    """内容の検査。whisper の CER（固有名詞の部分は数えない）と、固有名詞の読みの照合（ひらがなを出す CTC）。

    whisper は珍しい名前を別の漢字で書くので（谷口レンタカー店 -> 谷口蓮太カーテン）、読みでは比べられない。
    名前の部分は CER から外し（ワイルドカード）、名前の読みは、ひらがなを直接出す CTC モデルの書き起こしの中に
    近い部分文字列があるかで別に確かめる。
    """

    def __init__(self, device: str = "cuda"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.model = whisper_model("cuda", "float16")
        self.device = device
        hpath = local_or_repo(HIRAGANA_CTC)
        self.hproc = Wav2Vec2Processor.from_pretrained(hpath)
        self.hctc = Wav2Vec2ForCTC.from_pretrained(hpath).to(device).eval()

    @torch.inference_mode()
    def hiragana(self, wav16: np.ndarray) -> str:
        x = self.hproc(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        return self.hproc.decode(self.hctc(x).logits.argmax(-1)[0].cpu()).replace(" ", "")

    def check(self, wav16: np.ndarray, units: list[dict]) -> dict:
        from .units import utterance_units

        def kana(us, drop_fillers):
            parts = []
            for u in us:
                if u["punct"] or (drop_fillers and u["word"] in FILLERS):
                    continue
                parts.append(WILDCARD if u.get("entity") else _canon(_to_kata(u["hira"])))
            return "".join(parts)

        # greedy だと最初の「はい。」で書き起こしを打ち切ることがあった（音声は正常）。beam 5・時刻なしにする
        segs, _ = self.model.transcribe(wav16, language="ja", beam_size=5, without_timestamps=True,
                                        condition_on_previous_text=False)
        hyp = "".join(s.text for s in segs)
        hyp_units = utterance_units(hyp)
        out = {"asr": hyp}
        for name, drop in (("cer", True), ("cer_raw", False)):
            r, h = kana(units, drop), kana(hyp_units, drop)
            n = len(r.replace(WILDCARD, ""))
            out[name] = round(wildcard_edit_distance(r, h) / max(1, n), 4)
        ents = [u for u in units if u.get("entity")]
        if ents:
            dec = _loose(_to_kata(self.hiragana(wav16)))
            res = []
            for u in ents:
                # 許容幅は M2 で、ひらがな CTC の書き起こしの雑音（ミヤトヒロユキ -> ミアトヒロイキ）を見て決めた
                pat = _loose(_to_kata(u["hira"]))
                dist = substring_distance(pat, dec)
                res.append({"word": u["word"], "reading": u["hira"], "dist": dist, "ok": dist <= max(1, len(pat) // 3)})
            out["entities"] = res
            out["hiragana"] = dec
        return out


def check_backchannels(out: Path) -> None:
    """tts.py --backchannels で作った相槌の音声のうち、読みが合うものだけを tts/backchannels.jsonl に残す。

    ひらがな CTC の書き起こしに、相槌の読みに近い部分文字列があり、余計な音が 2 文字分を超えないこと。
    全部の行は tts/backchannels_all.jsonl に書き起こしと距離つきで残す。M2 では 2,293 本中 1,855 本が残った
    （「うん」「うんうん」は鼻音で書き起こしにくく、落ちやすい）。
    """
    from .units import utterance_units
    path = out / "tts" / "backchannels.jsonl"
    all_path = out / "tts" / "backchannels_all.jsonl"
    rows = [json.loads(line) for line in open(all_path if all_path.exists() else path)]
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(local_or_repo(HIRAGANA_CTC))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(local_or_repo(HIRAGANA_CTC)).to(dev).eval()
    keep = []
    for r in rows:
        wav, sr = sf.read(out / r["path"], dtype="float32")
        w16 = resample_poly(np.pad(wav, (int(0.1 * sr), int(0.1 * sr))), 16000, sr).astype(np.float32)
        with torch.inference_mode():
            ids = model(proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.argmax(-1)[0]
        dec = _loose(_to_kata(proc.decode(ids.cpu()).replace(" ", "")))
        pat = _loose(_to_kata("".join(u["hira"] for u in utterance_units(r["word"]) if not u["punct"])))
        r["asr_hira"], r["dist"] = dec, substring_distance(pat, dec)
        if r["dist"] <= max(1, len(pat) // 3) and len(dec) <= len(pat) + 2:
            keep.append(r)
    all_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print(f"backchannels: kept {len(keep)}/{len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="ens", choices=["ens", "ctc"])
    ap.add_argument("--asr", action="store_true", help="faster-whisper で CER も測る（遅い。抜き取り検査向け）")
    ap.add_argument("--check-backchannels", action="store_true", help="相槌の音声の読みを確かめて終わる")
    args = ap.parse_args()
    if args.check_backchannels:
        check_backchannels(Path(args.out))
        return
    if args.asr:
        _ensure_cuda_libs()
    out = Path(args.out)
    (out / "align").mkdir(exist_ok=True)
    done_path = out / "align" / "words.jsonl"
    done = {(r["key"], r["attempt"]) for r in map(json.loads, open(done_path))} if done_path.exists() else set()
    todo = [r for r in latest_rows(out / "tts" / "utts.jsonl") if (r["key"], r["attempt"]) not in done]
    aligner = Aligner(args.method)
    asr = AsrCheck() if args.asr else None
    with open(done_path, "a") as f:
        for n, r in enumerate(todo):
            wav, sr = sf.read(out / r["wav"], dtype="float32")
            dur = len(wav) / sr
            if dur < 0.1:
                words, qc = [], {"unexplained_speech_s": 0.0, "max_onset_spread": 0.0, "n_spread_gt_120ms": 0}
            else:
                words, qc = aligner.align(wav, sr, r["units"])
            qc["mora_per_s"] = round(r["mora"] / max(dur, 1e-3), 2)
            if asr is not None and dur >= 0.1:
                qc.update(asr.check(resample_poly(wav, 16000, sr).astype(np.float32), r["units"]))
            f.write(json.dumps({"key": r["key"], "attempt": r["attempt"], "words": words, "qc": qc},
                               ensure_ascii=False) + "\n")
            if (n + 1) % 50 == 0 or n + 1 == len(todo):
                print(f"align {n + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
