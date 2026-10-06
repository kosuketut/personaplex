"""シャードごとに作った対話（scripts/datagen.sh の出力）を 1 つのデータセットにまとめる。

    python -m japersonaplex.datagen.merge --root data/datagen/pilot40/main --valid-frac 0.01

<root>/shard_*/ の train.jsonl を集め、パスをシャードのディレクトリ付きに書き換えて <root>/train.jsonl と
<root>/valid.jsonl に分ける（対話 id のハッシュで valid-frac を検証用に、data_spec §6.1）。npz は
<root>/npz_train、<root>/npz_valid にシャードの npz への相対シンボリックリンクを張る（train.py の --train と
--valid にそのまま渡せる。数字を漢数字にした npz_kanji があれば npz_kanji_train、npz_kanji_valid にも）。
品質検査の集計を <root>/qc_summary.json に書く。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def is_valid(dialogue_id: str, frac: float) -> bool:
    return int(hashlib.sha1(("valid:" + dialogue_id).encode()).hexdigest()[:8], 16) % 10000 < frac * 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--valid-frac", type=float, default=0.01)
    args = ap.parse_args()
    root = Path(args.root)
    shards = sorted(p for p in root.glob("shard_*") if (p / "train.jsonl").exists())
    assert shards, f"no shard_*/train.jsonl under {root}"
    rows, status, missing_npz = {"train": [], "valid": []}, Counter(), []
    hours = Counter()
    for shard in shards:
        for line in open(shard / "train.jsonl"):
            r = json.loads(line)
            if not (shard / "npz" / f"{r['id']}.npz").exists():
                missing_npz.append(f"{shard.name}/{r['id']}")
                continue
            for k in ("audio", "words", "voice_prompt"):
                r[k] = f"{shard.name}/{r[k]}"
            r["shard"] = shard.name
            split = "valid" if is_valid(r["id"], args.valid_frac) else "train"
            r["split"] = split
            rows[split].append(r)
            hours[split] += r["duration"] / 3600
        qc = json.loads((shard / "qc.json").read_text())
        status.update(d["status"] for d in qc["dialogues"])
    for split, rs in rows.items():
        with open(root / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # npz_kanji（NPZ_KANJI=1 で作る、数字を漢数字にした版）があれば npz_kanji_<split> にも同じく張る
        for kind in ("npz", "npz_kanji"):
            if kind != "npz" and not all((root / r["shard"] / kind / f"{r['id']}.npz").exists() for r in rs):
                continue
            d = root / f"{kind}_{split}"
            d.mkdir(exist_ok=True)
            for old in d.glob("*.npz"):
                old.unlink()
            for r in rs:
                # シャードをまたいで id が重なっても上書きしないよう、リンク名にシャード名を付ける
                os.symlink(os.path.relpath(root / r["shard"] / kind / f"{r['id']}.npz", d), d / f"{r['shard']}__{r['id']}.npz")
    summary = {"shards": len(shards), "dialogues": dict(status), "train": len(rows["train"]),
               "valid": len(rows["valid"]), "hours": {k: round(v, 3) for k, v in hours.items()},
               "kinds": dict(Counter(r["kind"] for rs in rows.values() for r in rs)),
               "missing_npz": missing_npz}
    (root / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
