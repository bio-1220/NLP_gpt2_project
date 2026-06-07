#!/usr/bin/env python3
"""Create Kim Sowol poem-level emotion-prefix generation datasets.

This script matches the intended Korean experiment:

  1. use only Kim Sowol poems from KPoEM,
  2. use a fixed ratio of each poem's lines as the input prefix,
  3. condition generation on the poem-level/global KPoEM emotion label,
  4. compare no-emotion baseline, oracle emotion prefix, and random control.

It intentionally does not use line-level emotion trajectories.

Outputs for each ratio, e.g. r30:

  data/processed/sowol_ko_train_r30_base.jsonl
  data/processed/sowol_ko_train_r30_oracle.jsonl
  data/processed/sowol_ko_train_r30_random.jsonl
  data/processed/sowol_ko_dev_r30_*.jsonl
  data/processed/sowol_ko_test_r30_*.jsonl
  data/processed/sowol_global_emotion_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import EMOTION_LABELS, EMOTION_LABEL_TO_ID, read_jsonl, write_jsonl
from experiments.kpoem_emotion import collapse_kpoem_emotions


KO_EMOTION_LABELS = {
    "sadness": "슬픔",
    "joy": "기쁨",
    "love": "사랑",
    "anger": "분노",
    "fear": "공포",
    "surprise": "놀람",
}


def emotion_prefix(label: str) -> str:
    return f"[감정: {KO_EMOTION_LABELS.get(label, label)}]"


def split_by_ratio(row: dict[str, Any], ratio: float) -> dict[str, Any]:
    lines = [line.strip() for line in row["full_text"].splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Poem {row.get('id')} has fewer than 2 non-empty lines.")
    n_prefix = max(1, min(len(lines) - 1, int(round(len(lines) * ratio))))
    updated = dict(row)
    updated["prefix"] = "\n".join(lines[:n_prefix])
    updated["target"] = "\n".join(lines[n_prefix:])
    updated["full_text"] = "\n".join(lines)
    updated["num_lines"] = len(lines)
    updated["num_prefix_lines"] = n_prefix
    updated["prefix_ratio"] = ratio
    return updated


def add_common_fields(row: dict[str, Any], oracle: dict[str, Any], policy: str) -> dict[str, Any]:
    updated = dict(row)
    updated.update(
        {
            "global_emotion_policy": policy,
            "poem_level_oracle_label": oracle["label"],
            "poem_level_oracle_label_id": oracle["label_id"],
            "poem_level_oracle_label_ko": KO_EMOTION_LABELS.get(oracle["label"], oracle["label"]),
            "poem_level_oracle_counts": oracle["label_counts"],
            "poem_level_oracle_source_labels": oracle["source_labels"],
            "poem_level_oracle_mapping_mode": oracle["mapping_mode"],
        }
    )
    return updated


def make_base_row(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    updated = add_common_fields(row, oracle, "none")
    updated["model_input"] = row["prefix"]
    updated["conditioned_full_text"] = row["full_text"]
    return updated


def make_oracle_row(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    prompt = emotion_prefix(oracle["label"])
    updated = add_common_fields(row, oracle, "poem_level_oracle")
    updated["condition_label"] = oracle["label"]
    updated["condition_label_id"] = oracle["label_id"]
    updated["condition_label_ko"] = KO_EMOTION_LABELS.get(oracle["label"], oracle["label"])
    updated["emotion_prompt"] = prompt
    updated["model_input"] = f"{prompt}\n\n{row['prefix']}"
    updated["conditioned_full_text"] = f"{prompt}\n\n{row['full_text']}"
    return updated


def make_random_row(row: dict[str, Any], oracle: dict[str, Any], split: str, seed: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}-{split}-{row['id']}")
    label = rng.choice(EMOTION_LABELS)
    prompt = emotion_prefix(label)
    updated = add_common_fields(row, oracle, "random")
    updated["condition_label"] = label
    updated["condition_label_id"] = EMOTION_LABEL_TO_ID[label]
    updated["condition_label_ko"] = KO_EMOTION_LABELS.get(label, label)
    updated["emotion_prompt"] = prompt
    updated["model_input"] = f"{prompt}\n\n{row['prefix']}"
    updated["conditioned_full_text"] = f"{prompt}\n\n{row['full_text']}"
    return updated


def ratio_tag(ratio: float) -> str:
    return f"r{int(round(ratio * 100))}"


def process_split(args: argparse.Namespace, split: str, ratio: float) -> dict[str, Any]:
    rows = read_jsonl(args.processed_dir / f"poem_ko_{split}.jsonl")
    base_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    dropped = Counter()

    for row in rows:
        if row.get("poet") != args.poet:
            continue
        try:
            row = split_by_ratio(row, ratio)
        except ValueError:
            dropped["too_short"] += 1
            continue
        oracle = collapse_kpoem_emotions(
            row.get("emotion_metadata", {}),
            mode=args.mode,
            min_count=args.min_count,
            require_unique_top=True,
        )
        if oracle is None:
            dropped["ambiguous_or_unmapped_emotion"] += 1
            continue
        base_rows.append(make_base_row(row, oracle))
        oracle_rows.append(make_oracle_row(row, oracle))
        random_rows.append(make_random_row(row, oracle, split, args.seed))

    tag = ratio_tag(ratio)
    write_jsonl(args.processed_dir / f"sowol_ko_{split}_{tag}_base.jsonl", base_rows)
    write_jsonl(args.processed_dir / f"sowol_ko_{split}_{tag}_oracle.jsonl", oracle_rows)
    write_jsonl(args.processed_dir / f"sowol_ko_{split}_{tag}_random.jsonl", random_rows)

    return {
        "split": split,
        "ratio": ratio,
        "ratio_tag": tag,
        "rows": len(base_rows),
        "dropped": dict(dropped),
        "oracle_label_counts": dict(Counter(row["poem_level_oracle_label"] for row in oracle_rows)),
        "prefix_line_counts": dict(Counter(row["num_prefix_lines"] for row in base_rows)),
    }


def parse_ratios(text: str) -> list[float]:
    ratios = [float(part.strip()) for part in text.split(",") if part.strip()]
    for ratio in ratios:
        if ratio <= 0 or ratio >= 1:
            raise ValueError("--ratios must be comma-separated floats between 0 and 1.")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--poet", type=str, default="김소월")
    parser.add_argument("--ratios", type=str, default="0.3,0.5")
    parser.add_argument("--mode", choices=["strict", "poetic"], default="poetic")
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11711)
    args = parser.parse_args()

    summary = {
        "purpose": "Kim Sowol subset with poem-level/global emotion prefix; no line trajectories.",
        "poet": args.poet,
        "ratios": parse_ratios(args.ratios),
        "mode": args.mode,
        "min_count": args.min_count,
        "seed": args.seed,
        "splits": [],
    }
    for ratio in summary["ratios"]:
        for split in ["train", "dev", "test"]:
            summary["splits"].append(process_split(args, split, ratio))

    output_path = args.processed_dir / "sowol_global_emotion_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

