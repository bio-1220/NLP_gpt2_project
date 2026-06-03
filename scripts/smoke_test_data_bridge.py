#!/usr/bin/env python3
"""Smoke test for processed JSONL datasets and experiment utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import EmotionJsonlDataset, PoemContinuationJsonlDataset, label_distribution
from experiments.metrics import chrf_corpus_score
from experiments.prompts import build_emotion_prompt


def main() -> None:
    processed_dir = Path("data/processed")
    required = [
        processed_dir / "emotion_en_train.jsonl",
        processed_dir / "emotion_ko_train.jsonl",
        processed_dir / "poem_ko_train.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed files. Run `py -3 scripts\\prepare_external_data.py` first: "
            + ", ".join(missing)
        )

    en_emotion = EmotionJsonlDataset(processed_dir / "emotion_en_train.jsonl")
    ko_emotion = EmotionJsonlDataset(processed_dir / "emotion_ko_train.jsonl")
    ko_poems = PoemContinuationJsonlDataset(processed_dir / "poem_ko_train.jsonl")

    poem_example = ko_poems[0]
    conditioned = build_emotion_prompt(poem_example["prefix"], "sadness", language="ko")
    chrf_identity = chrf_corpus_score([poem_example["target"]], [poem_example["target"]])

    summary = {
        "emotion_en_train": {
            "rows": len(en_emotion),
            "labels": label_distribution(en_emotion.rows),
            "sample": en_emotion[0],
        },
        "emotion_ko_train": {
            "rows": len(ko_emotion),
            "labels": label_distribution(ko_emotion.rows),
            "sample": ko_emotion[0],
        },
        "poem_ko_train": {
            "rows": len(ko_poems),
            "sample": {
                "id": poem_example["id"],
                "title": poem_example["title"],
                "poet": poem_example["poet"],
                "prefix": poem_example["prefix"],
                "target_preview": poem_example["target"][:120],
                "conditioned_prompt": conditioned,
            },
        },
        "chrf_identity_check": chrf_identity,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
