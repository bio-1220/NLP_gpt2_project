#!/usr/bin/env python3
"""Create KPoEM oracle-emotion datasets for diagnostic experiments.

Outputs:
  - data/processed/poem_ko_{split}_oracle.jsonl
    Full poem-continuation rows plus oracle_label fields.
  - data/processed/emotion_kpoem_prefix_{split}.jsonl
    Emotion-classification/evaluation rows using the first three lines as text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import read_jsonl, write_jsonl
from experiments.kpoem_emotion import collapse_kpoem_emotions


def add_oracle_fields(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated.update(
        {
            "oracle_label": oracle["label"],
            "oracle_label_id": oracle["label_id"],
            "oracle_label_counts": oracle["label_counts"],
            "oracle_source_labels": oracle["source_labels"],
            "oracle_mapping_mode": oracle["mapping_mode"],
            "oracle_min_count": oracle["min_count"],
        }
    )
    return updated


def make_prefix_emotion_row(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "text": row["prefix"],
        "label": oracle["label"],
        "label_id": oracle["label_id"],
        "poem_id": row.get("poem_id"),
        "title": row.get("title", ""),
        "poet": row.get("poet", ""),
        "target": row.get("target", ""),
        "full_text": row.get("full_text", ""),
        "oracle_label_counts": oracle["label_counts"],
        "oracle_source_labels": oracle["source_labels"],
        "oracle_mapping_mode": oracle["mapping_mode"],
        "oracle_min_count": oracle["min_count"],
    }


def output_name(base: str, split: str, mode: str) -> str:
    suffix = "" if mode == "poetic" else f"_{mode}"
    return f"{base}_{split}{suffix}.jsonl"


def process_split(processed_dir: Path, split: str, mode: str, min_count: int) -> dict[str, Any]:
    input_path = processed_dir / f"poem_ko_{split}.jsonl"
    rows = read_jsonl(input_path)
    oracle_rows = []
    emotion_rows = []
    dropped = Counter()

    for row in rows:
        oracle = collapse_kpoem_emotions(
            row.get("emotion_metadata", {}),
            mode=mode,
            min_count=min_count,
            require_unique_top=True,
        )
        if oracle is None:
            dropped["ambiguous_or_unmapped"] += 1
            continue
        oracle_rows.append(add_oracle_fields(row, oracle))
        emotion_rows.append(make_prefix_emotion_row(row, oracle))

    write_jsonl(processed_dir / output_name("poem_ko", f"{split}_oracle", mode), oracle_rows)
    write_jsonl(processed_dir / output_name("emotion_kpoem_prefix", split, mode), emotion_rows)

    label_counts = Counter(row["oracle_label"] for row in oracle_rows)
    return {
        "split": split,
        "input_rows": len(rows),
        "kept_rows": len(oracle_rows),
        "dropped": dict(dropped),
        "label_counts": dict(label_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--mode", choices=["strict", "poetic"], default="poetic")
    parser.add_argument("--min_count", type=int, default=2)
    args = parser.parse_args()

    summary = {
        "mode": args.mode,
        "min_count": args.min_count,
        "splits": [
            process_split(args.processed_dir, split, args.mode, args.min_count)
            for split in ["train", "dev", "test"]
        ],
    }
    summary_suffix = "" if args.mode == "poetic" else f"_{args.mode}"
    output_path = args.processed_dir / f"kpoem_oracle_emotion_summary{summary_suffix}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
