#!/usr/bin/env python3
"""Round 3 (KO): full-poem fine-grained emotion trajectory data.

Design (user spec):
  - emotion labels = KPoEM's 44 fine-grained classes AS-IS (no 6-class collapse)
  - the trajectory block covers ALL lines of the poem (future lines included),
    given in BOTH training and test prompts (deliberate oracle, like EN Round 3)
  - per-line label = majority vote over all 5 annotators' label instances
  - prefix/target split at 30% of lines (test-time input = first 30%)
  - train/dev = all poets; test = Kim Sowol only
Outputs (data/processed/):
  poem_ko_train_fulltraj_fine_r30.jsonl   (368, all poets)
  poem_ko_dev_fulltraj_fine_r30.jsonl     (46, all poets)
  poem_ko_test_fulltraj_fine_r30_sowol.jsonl (Sowol only)
  poem_ko_test_plain_r30_sowol.jsonl      (unconditioned baseline eval)
  kpoem_fulltraj_fine_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import read_jsonl, write_jsonl
from experiments.kpoem_emotion import mapping_to_lookup
from experiments.prompts import KO_EMOTION_LABELS

D = Path("data/processed")
RATIO = 0.3


def load_fine_line_labels(raw_dir=Path("data/raw"), mode="fine"):
    """poem_id -> ordered list of per-line majority labels.

    mode='fine': majority over the 44 fine-grained labels as-is.
    mode='six' : map each label instance to the 6-class space (poetic mapping,
                 Korean names) BEFORE counting — same majority logic otherwise.
    """
    lookup = mapping_to_lookup("poetic") if mode == "six" else None
    df = pd.read_csv(raw_dir / "kpoem" / "KPoEM_line_dataset_v4.tsv",
                     sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
    acols = [c for c in df.columns if c.startswith("annotator_")]
    by_poem: dict[int, list[str]] = defaultdict(list)
    for _, row in df.sort_values(["poem_id", "line_id"]).iterrows():
        counts: Counter = Counter()
        for c in acols:
            v = row[c]
            if pd.notna(v):
                instances = [l.strip() for l in str(v).split(",") if l.strip()]
                if lookup is not None:
                    instances = [KO_EMOTION_LABELS[lookup[l]] for l in instances if l in lookup]
                counts.update(instances)
        # deterministic majority: highest count, tie broken alphabetically
        label = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counts else None
        by_poem[int(row["poem_id"])].append(label)
    return by_poem


def split_lines(full_text):
    return [l for l in full_text.split("\n") if l.strip()]


def build_block(labels):
    body = [f"{i}: {lab}" for i, lab in enumerate(labels, 1)]
    return "\n".join(["[감정흐름]", *body, "[/감정흐름]"])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--label_mode", choices=["fine", "six"], default="fine")
    args = ap.parse_args()
    tag = args.label_mode
    fine = load_fine_line_labels(mode=tag)
    summary = {"ratio": RATIO, "label_mode": tag, "splits": {}}

    for split in ["train", "dev", "test"]:
        rows = read_jsonl(D / f"poem_ko_{split}.jsonl")
        out, fallback = [], 0
        for row in rows:
            lines = split_lines(row["full_text"])
            n = len(lines)
            n_prefix = max(1, min(n - 1, int(round(n * RATIO))))
            labs = fine.get(int(row["poem_id"]), [])
            # align to this poem's line count; fall back to poem-majority then '슬픔'
            poem_major = Counter(l for l in labs if l)
            default = poem_major.most_common(1)[0][0] if poem_major else "슬픔"
            full_labels = []
            for i in range(n):
                lab = labs[i] if i < len(labs) and labs[i] else None
                if lab is None:
                    lab, fallback = default, fallback + 1
                full_labels.append(lab)
            block = build_block(full_labels)
            r = dict(row)
            r.update({
                "prefix": "\n".join(lines[:n_prefix]),
                "target": "\n".join(lines[n_prefix:]),
                "num_prefix_lines": n_prefix,
                "trajectory_policy": f"oracle_full_poem_{tag}",
                "trajectory_labels": full_labels,
                "trajectory_prompt": block,
                "model_input": f"{block}\n\n" + "\n".join(lines[:n_prefix]),
                "conditioned_full_text": f"{block}\n\n{row['full_text']}",
            })
            out.append(r)

        if split == "test":
            sowol = [r for r in out if "김소월" in r.get("poet", "")]
            write_jsonl(D / f"poem_ko_test_fulltraj_{tag}_r30_sowol.jsonl", sowol)
            plain = []
            for r in sowol:
                p = {k: r[k] for k in ["id", "poem_id", "title", "poet", "prefix", "target",
                                       "full_text", "num_prefix_lines"]}
                p["trajectory_policy"] = "none"
                p["model_input"] = r["prefix"]
                plain.append(p)
            write_jsonl(D / "poem_ko_test_plain_r30_sowol.jsonl", plain)
            summary["splits"]["test_sowol"] = {"rows": len(sowol), "fallback_lines": fallback}
        else:
            write_jsonl(D / f"poem_ko_{split}_fulltraj_{tag}_r30.jsonl", out)
            summary["splits"][split] = {"rows": len(out), "fallback_lines": fallback}

    label_pool = Counter()
    for r in read_jsonl(D / f"poem_ko_train_fulltraj_{tag}_r30.jsonl"):
        label_pool.update(r["trajectory_labels"])
    summary["distinct_labels_in_train"] = len(label_pool)
    summary["top_labels"] = label_pool.most_common(8)
    (D / f"kpoem_fulltraj_{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
