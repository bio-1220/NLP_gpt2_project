#!/usr/bin/env python3
"""Round 3 English data builder.

From the per-column oracle/average sentiment-curve files, produce:
  1. dev *_no138.jsonl       — dev with sonnet-138 removed (dev138 == test155 duplicate)
  2. *_sent_shuf{col}_cont   — SHUFFLED control: each sonnet gets ANOTHER sonnet's real
                               curve (cyclic shift within split). Realistic curve form,
                               wrong content. Applied to train AND dev consistently.
  3. sonnet_en_dev_plain_no138.jsonl — unconditioned baseline eval file (prompt = prefix)

Run AFTER prepare_shakespeare_sentiment_trajectory.py --continuous [--score_col ...].
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import read_jsonl, write_jsonl

D = Path("data/processed")
EXCLUDE_ID = "sonnet-138"
COL_SFX = {"m": "", "v": "_sentiment_v", "r": "_sentiment_r"}


def drop_138(rows):
    return [r for r in rows if r["id"] != EXCLUDE_ID]


def reattach_curve(row, donor):
    """Give `row` the curve of `donor` (keeps row's own poem text)."""
    out = dict(row)
    out["sentiment_policy"] = "shuffled_oracle_sentiment_trajectory"
    out["sentiment_trajectory"] = donor["sentiment_trajectory"]
    out["sentiment_prompt"] = donor["sentiment_prompt"]
    out["curve_donor_id"] = donor["id"]
    out["model_input"] = f"{donor['sentiment_prompt']}\n\n{row['prefix']}"
    out["conditioned_full_text"] = f"{donor['sentiment_prompt']}\n\n{row['full_text']}"
    return out


def shuffle_curves(rows):
    """Cyclic shift: sonnet i gets sonnet i+1's curve. Deterministic, no fixed point."""
    rows = sorted(rows, key=lambda r: r["sonnet_id"])
    n = len(rows)
    return [reattach_curve(rows[i], rows[(i + 1) % n]) for i in range(n)]


def main():
    summary = {}
    for col, sfx in COL_SFX.items():
        oracle_train = read_jsonl(D / f"sonnet_en_train_sent_oracle{sfx}_cont.jsonl")
        oracle_dev = drop_138(read_jsonl(D / f"sonnet_en_dev_sent_oracle{sfx}_cont.jsonl"))
        avg_dev = drop_138(read_jsonl(D / f"sonnet_en_dev_sent_avg{sfx}_cont.jsonl"))

        write_jsonl(D / f"sonnet_en_dev_sent_oracle{sfx}_cont_no138.jsonl", oracle_dev)
        write_jsonl(D / f"sonnet_en_dev_sent_avg{sfx}_cont_no138.jsonl", avg_dev)

        shuf_train = shuffle_curves(oracle_train)
        shuf_dev = shuffle_curves(oracle_dev)  # shuffle AFTER dropping 138
        write_jsonl(D / f"sonnet_en_train_sent_shuf{sfx}_cont.jsonl", shuf_train)
        write_jsonl(D / f"sonnet_en_dev_sent_shuf{sfx}_cont_no138.jsonl", shuf_dev)

        summary[col] = {"train": len(oracle_train), "dev_no138": len(oracle_dev)}

    # unconditioned baseline eval file (from col-m oracle dev, curve stripped)
    plain = []
    for r in read_jsonl(D / "sonnet_en_dev_sent_oracle_cont_no138.jsonl"):
        row = {k: r[k] for k in ["id", "sonnet_id", "prefix", "target", "full_text", "num_lines"]}
        row["sentiment_policy"] = "none"
        row["model_input"] = r["prefix"]
        plain.append(row)
    write_jsonl(D / "sonnet_en_dev_plain_no138.jsonl", plain)
    summary["plain_dev"] = len(plain)

    print(summary)


if __name__ == "__main__":
    main()
