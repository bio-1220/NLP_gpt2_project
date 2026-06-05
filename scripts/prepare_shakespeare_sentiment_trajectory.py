#!/usr/bin/env python3
"""Create Shakespeare sonnet sentiment-trajectory inputs for Method A/B.

Method A: Oracle sentiment trajectory guidance.
Method B: Train-average sentiment trajectory guidance.

This does not implement the predicted planner (Method C). It only prepares
conditioned JSONL files that a GPT-2 generation training script can consume.

Outputs:
  data/processed/sonnet_en_train_sent_oracle.jsonl
  data/processed/sonnet_en_dev_sent_oracle.jsonl
  data/processed/sonnet_en_heldout_sent_oracle.jsonl
  data/processed/sonnet_en_train_sent_avg.jsonl
  data/processed/sonnet_en_dev_sent_avg.jsonl
  data/processed/sonnet_en_heldout_sent_avg.jsonl
  data/processed/shakespeare_sentiment_trajectory_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import write_jsonl


ZENODO_SENTIMENT_URL = (
    "https://zenodo.org/api/records/14965925/files/"
    "sentiment_scores_raw.csv/content"
)


def ensure_sentiment_csv(raw_dir: Path) -> Path:
    path = raw_dir / "shakespeare_sentiment_scores_raw.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(ZENODO_SENTIMENT_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def parse_numbered_sonnets(path: Path, require_full: bool) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\n\s*(\d+)\s*\n\n(.*?)(?=\n\s*\d+\s*\n\n|\Z)", re.S)
    rows = []
    for match in pattern.finditer(text):
        sonnet_id = int(match.group(1))
        body = match.group(2).strip()
        lines = [line.rstrip() for line in body.splitlines() if line.strip()]
        if require_full and len(lines) < 14:
            continue
        prefix = "\n".join(lines[:3])
        target = "\n".join(lines[3:14]) if len(lines) >= 14 else ""
        full_text = "\n".join(lines[:14]) if len(lines) >= 14 else "\n".join(lines)
        rows.append(
            {
                "id": f"sonnet-{sonnet_id}",
                "sonnet_id": sonnet_id,
                "prefix": prefix,
                "target": target,
                "full_text": full_text,
                "num_lines": len(lines),
            }
        )
    return rows


def load_sentiment_trajectories(csv_path: Path, score_col: str) -> dict[int, list[float]]:
    df = pd.read_csv(csv_path)
    trajectories: dict[int, list[float]] = {}
    for sonnet_id, group in df.groupby("Sonnet"):
        group = group.sort_values("line")
        values = [float(value) for value in group[score_col].tolist()]
        trajectories[int(sonnet_id)] = values
    return trajectories


def format_value(value: float, discrete: bool) -> str:
    if discrete:
        return str(int(round(value)))
    return f"{value:+.3f}"


def build_sentiment_prompt(values: list[float], discrete: bool) -> str:
    lines = [f"{idx}: {format_value(value, discrete)}" for idx, value in enumerate(values, 1)]
    return "\n".join(["[sentiment trajectory]", *lines, "[/sentiment trajectory]"])


def add_conditioning(
    row: dict[str, Any],
    values: list[float],
    policy: str,
    score_col: str,
    discrete: bool,
) -> dict[str, Any]:
    prompt = build_sentiment_prompt(values, discrete)
    model_input = f"{prompt}\n\n{row['prefix']}"
    conditioned_full_text = f"{prompt}\n\n{row['full_text']}"
    updated = dict(row)
    updated.update(
        {
            "sentiment_policy": policy,
            "sentiment_score_col": score_col,
            "sentiment_trajectory": values,
            "sentiment_prompt": prompt,
            "model_input": model_input,
            "conditioned_full_text": conditioned_full_text,
        }
    )
    return updated


def average_trajectory(train_rows: list[dict[str, Any]], length: int = 14) -> list[float]:
    columns = [[] for _ in range(length)]
    for row in train_rows:
        values = row["sentiment_trajectory"]
        for idx, value in enumerate(values[:length]):
            columns[idx].append(float(value))
    return [sum(values) / len(values) if values else 0.0 for values in columns]


def file_suffix(args: argparse.Namespace) -> str:
    if args.output_suffix:
        return args.output_suffix
    suffix = ""
    if args.score_col != "sentiment_m":
        suffix += f"_{args.score_col}"
    if not args.discrete:
        suffix += "_cont"
    return suffix


def process(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.raw_dir
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_sentiment_csv(raw_dir)
    trajectories = load_sentiment_trajectories(csv_path, args.score_col)

    split_specs = {
        "train": (Path("data/sonnets.txt"), True),
        "dev": (Path("data/TRUE_sonnets_held_out_dev.txt"), True),
        "heldout": (Path("data/sonnets_held_out.txt"), False),
    }

    oracle_by_split: dict[str, list[dict[str, Any]]] = {}
    dropped: dict[str, list[int]] = {}
    skipped_short: dict[str, int] = {}
    suffix = file_suffix(args)
    for split, (path, require_full) in split_specs.items():
        all_rows = parse_numbered_sonnets(path, require_full=False)
        rows = [row for row in all_rows if not require_full or row["num_lines"] >= 14]
        skipped_short[split] = len(all_rows) - len(rows)
        conditioned = []
        dropped_ids = []
        for row in rows:
            values = trajectories.get(row["sonnet_id"])
            if values is None or len(values) < 14:
                dropped_ids.append(row["sonnet_id"])
                continue
            conditioned.append(
                add_conditioning(
                    row,
                    values[:14],
                    policy="oracle_sentiment_trajectory",
                    score_col=args.score_col,
                    discrete=args.discrete,
                )
            )
        oracle_by_split[split] = conditioned
        dropped[split] = dropped_ids
        write_jsonl(processed_dir / f"sonnet_en_{split}_sent_oracle{suffix}.jsonl", conditioned)

    avg_values = average_trajectory(oracle_by_split["train"], length=14)
    avg_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, rows in oracle_by_split.items():
        avg_rows = [
            add_conditioning(
                row,
                avg_values,
                policy="train_average_sentiment_trajectory",
                score_col=args.score_col,
                discrete=args.discrete,
            )
            for row in rows
        ]
        avg_by_split[split] = avg_rows
        write_jsonl(processed_dir / f"sonnet_en_{split}_sent_avg{suffix}.jsonl", avg_rows)

    summary = {
        "score_col": args.score_col,
        "discrete_prompt": args.discrete,
        "average_trajectory": avg_values,
        "average_prompt": build_sentiment_prompt(avg_values, args.discrete),
        "output_suffix": suffix,
        "splits": {},
        "dropped_sonnet_ids": dropped,
        "skipped_short_sonnets": skipped_short,
    }
    for split, rows in oracle_by_split.items():
        ids = [row["sonnet_id"] for row in rows]
        line_values = Counter()
        for row in rows:
            line_values.update(int(round(value)) for value in row["sentiment_trajectory"])
        summary["splits"][split] = {
            "rows": len(rows),
            "sonnet_id_min": min(ids) if ids else None,
            "sonnet_id_max": max(ids) if ids else None,
            "sentiment_value_counts": dict(line_values),
        }

    summary_name = f"shakespeare_sentiment_trajectory_summary{suffix}.json"
    (processed_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--score_col",
        choices=["sentiment_m", "sentiment_v", "sentiment_r"],
        default="sentiment_m",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous prompt values instead of rounded/discrete values.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix for output JSONL files.",
    )
    args = parser.parse_args()
    args.discrete = not args.continuous

    summary = process(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
