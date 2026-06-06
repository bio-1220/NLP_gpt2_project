#!/usr/bin/env python3
"""Create KPoEM line-level emotion trajectory inputs for generation.

This script prepares the "trajectory guidance" datasets discussed in the
additional experiment proposal. It only uses emotions for the observed prefix
lines, not future target lines, so it can be used as a leakage-safe diagnostic.

Outputs:
  data/processed/poem_ko_{split}_line_traj.jsonl
  data/processed/poem_ko_{split}_line_traj_avg.jsonl
  data/processed/poem_ko_{split}_line_traj_random.jsonl
  data/processed/kpoem_line_trajectory_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import EMOTION_LABEL_TO_ID, read_jsonl, write_jsonl
from experiments.kpoem_emotion import collapse_kpoem_emotions
from experiments.prompts import KO_EMOTION_LABELS


EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_line_metadata(raw_dir: Path, mode: str, min_count: int) -> dict[int, list[dict[str, Any]]]:
    line_path = raw_dir / "kpoem" / "KPoEM_line_dataset_v4.tsv"
    if not line_path.exists():
        raise FileNotFoundError(
            f"Missing {line_path}. Run `py -3 scripts\\prepare_external_data.py` first."
        )

    df = pd.read_csv(line_path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
    annotator_cols = [col for col in df.columns if col.startswith("annotator_")]
    by_poem: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for _, row in df.sort_values(["poem_id", "line_id"]).iterrows():
        metadata = {
            col: row[col]
            for col in annotator_cols
            if col in row and not pd.isna(row[col])
        }
        oracle = collapse_kpoem_emotions(
            metadata,
            mode=mode,
            min_count=min_count,
            require_unique_top=False,
        )
        by_poem[int(row["poem_id"])].append(
            {
                "line_id": int(row["line_id"]),
                "text": str(row["text"]),
                "oracle": oracle,
                "emotion_metadata": metadata,
            }
        )

    return by_poem


def fallback_poem_label(row: dict[str, Any], mode: str) -> str:
    oracle = collapse_kpoem_emotions(
        row.get("emotion_metadata", {}),
        mode=mode,
        min_count=1,
        require_unique_top=False,
    )
    return oracle["label"] if oracle is not None else "sadness"


def prefix_line_trajectory(
    row: dict[str, Any],
    line_metadata: dict[int, list[dict[str, Any]]],
    mode: str,
    num_prefix_lines: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    poem_id = int(row["poem_id"])
    fallback = fallback_poem_label(row, mode)
    lines = line_metadata.get(poem_id, [])[:num_prefix_lines]
    labels = []
    details = []

    for pos in range(num_prefix_lines):
        line = lines[pos] if pos < len(lines) else None
        oracle = None if line is None else line.get("oracle")
        label = oracle["label"] if oracle is not None else fallback
        labels.append(label)
        details.append(
            {
                "position": pos + 1,
                "line_id": None if line is None else line["line_id"],
                "line_text": "" if line is None else line["text"],
                "label": label,
                "label_counts": {} if oracle is None else oracle["label_counts"],
                "used_fallback": oracle is None,
            }
        )

    return labels, details


def build_trajectory_prompt(labels: list[str], language: str = "ko") -> str:
    title = "[감정흐름]" if language == "ko" else "[emotion trajectory]"
    end = "[/감정흐름]" if language == "ko" else "[/emotion trajectory]"
    if language == "ko":
        lines = [f"{idx}: {KO_EMOTION_LABELS.get(label, label)}" for idx, label in enumerate(labels, 1)]
    else:
        lines = [f"{idx}: {label}" for idx, label in enumerate(labels, 1)]
    return "\n".join([title, *lines, end])


def add_conditioning_fields(
    row: dict[str, Any],
    labels: list[str],
    details: list[dict[str, Any]],
    policy: str,
    language: str,
) -> dict[str, Any]:
    prompt = build_trajectory_prompt(labels, language=language)
    model_input = f"{prompt}\n\n{row['prefix']}"
    conditioned_full_text = f"{prompt}\n\n{row['full_text']}"
    updated = dict(row)
    updated.update(
        {
            "trajectory_policy": policy,
            "trajectory_labels": labels,
            "trajectory_label_ids": [EMOTION_LABEL_TO_ID[label] for label in labels],
            "trajectory_labels_ko": [KO_EMOTION_LABELS.get(label, label) for label in labels],
            "trajectory_details": details,
            "trajectory_prompt": prompt,
            "model_input": model_input,
            "conditioned_full_text": conditioned_full_text,
        }
    )
    return updated


def majority_trajectory(train_rows: list[dict[str, Any]], num_prefix_lines: int) -> list[str]:
    counters = [Counter() for _ in range(num_prefix_lines)]
    overall: Counter = Counter()
    for row in train_rows:
        labels = row["trajectory_labels"][:num_prefix_lines]
        overall.update(labels)
        for idx, label in enumerate(labels):
            counters[idx][label] += 1
    fallback = overall.most_common(1)[0][0] if overall else "sadness"
    return [
        counter.most_common(1)[0][0] if counter else fallback
        for counter in counters
    ]


def make_random_labels(seed: int, split: str, row_id: str, num_prefix_lines: int) -> list[str]:
    rng = random.Random(f"{seed}-{split}-{row_id}")
    return [rng.choice(EMOTION_LABELS) for _ in range(num_prefix_lines)]


def resplit_by_ratio(row: dict[str, Any], ratio: float) -> tuple[dict[str, Any], int]:
    """Re-split prefix/target as the first `ratio` fraction of the poem's lines."""
    lines = [line for line in row["full_text"].split("\n") if line.strip()]
    n = len(lines)
    n_prefix = max(1, min(n - 1, int(round(n * ratio))))
    updated = dict(row)
    updated["prefix"] = "\n".join(lines[:n_prefix])
    updated["target"] = "\n".join(lines[n_prefix:])
    updated["num_prefix_lines"] = n_prefix
    return updated, n_prefix


def process(args: argparse.Namespace) -> dict[str, Any]:
    line_metadata = load_line_metadata(args.raw_dir, args.mode, args.line_min_count)
    ratio = args.prefix_ratio
    suffix = f"_r{int(round(ratio * 100))}" if ratio > 0 else ""

    oracle_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "test"]:
        rows = read_jsonl(args.processed_dir / f"poem_ko_{split}.jsonl")
        oracle_rows = []
        plain_rows = []
        for row in rows:
            if ratio > 0:
                row, n_prefix = resplit_by_ratio(row, ratio)
                plain = dict(row)
                plain["trajectory_policy"] = "none"
                plain["model_input"] = row["prefix"]
                plain["conditioned_full_text"] = row["full_text"]
                plain_rows.append(plain)
            else:
                n_prefix = args.num_prefix_lines
            labels, details = prefix_line_trajectory(
                row,
                line_metadata,
                mode=args.mode,
                num_prefix_lines=n_prefix,
            )
            oracle_rows.append(
                add_conditioning_fields(
                    row,
                    labels,
                    details,
                    policy="oracle_prefix_line",
                    language=args.language,
                )
            )
        oracle_by_split[split] = oracle_rows
        write_jsonl(args.processed_dir / f"poem_ko_{split}_line_traj{suffix}.jsonl", oracle_rows)
        if ratio > 0:
            write_jsonl(args.processed_dir / f"poem_ko_{split}_plain{suffix}.jsonl", plain_rows)

    max_prefix_len = max(len(row["trajectory_labels"]) for row in oracle_by_split["train"])
    avg_labels = majority_trajectory(oracle_by_split["train"], max_prefix_len)

    random_by_split: dict[str, list[dict[str, Any]]] = {}
    avg_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, rows in oracle_by_split.items():
        avg_rows = []
        random_rows = []
        for row in rows:
            row_n = len(row["trajectory_labels"])
            # average trajectory truncated/padded (with its last label) to this row's prefix length
            row_avg = (avg_labels + [avg_labels[-1]] * row_n)[:row_n]
            avg_details = [
                {"position": idx + 1, "label": label, "source": "train_majority_trajectory"}
                for idx, label in enumerate(row_avg)
            ]
            avg_rows.append(
                add_conditioning_fields(
                    row,
                    row_avg,
                    avg_details,
                    policy="train_average_prefix_line",
                    language=args.language,
                )
            )

            labels = make_random_labels(args.seed, split, row["id"], row_n)
            details = [
                {"position": idx + 1, "label": label, "source": "deterministic_random"}
                for idx, label in enumerate(labels)
            ]
            random_rows.append(
                add_conditioning_fields(
                    row,
                    labels,
                    details,
                    policy="random_prefix_line",
                    language=args.language,
                )
            )

        avg_by_split[split] = avg_rows
        random_by_split[split] = random_rows
        write_jsonl(args.processed_dir / f"poem_ko_{split}_line_traj_avg{suffix}.jsonl", avg_rows)
        write_jsonl(args.processed_dir / f"poem_ko_{split}_line_traj_random{suffix}.jsonl", random_rows)

    summary = {
        "mode": args.mode,
        "line_min_count": args.line_min_count,
        "num_prefix_lines": args.num_prefix_lines,
        "prefix_ratio": ratio,
        "max_prefix_len": max_prefix_len,
        "language": args.language,
        "average_trajectory": avg_labels,
        "average_trajectory_ko": [KO_EMOTION_LABELS[label] for label in avg_labels],
        "splits": {},
    }
    for split, rows in oracle_by_split.items():
        label_counts = Counter()
        fallback_count = 0
        for row in rows:
            label_counts.update(row["trajectory_labels"])
            fallback_count += sum(1 for detail in row["trajectory_details"] if detail.get("used_fallback"))
        summary["splits"][split] = {
            "rows": len(rows),
            "trajectory_label_counts": dict(label_counts),
            "line_fallback_count": fallback_count,
        }

    output_path = args.processed_dir / f"kpoem_line_trajectory_summary{suffix}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--mode", choices=["strict", "poetic"], default="poetic")
    parser.add_argument("--line_min_count", type=int, default=1)
    parser.add_argument("--num_prefix_lines", type=int, default=3)
    parser.add_argument("--prefix_ratio", type=float, default=0.0,
                        help="if >0, split prefix/target by this fraction of lines instead of a fixed line count")
    parser.add_argument("--language", choices=["ko", "en"], default="ko")
    parser.add_argument("--seed", type=int, default=11711)
    args = parser.parse_args()

    summary = process(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
