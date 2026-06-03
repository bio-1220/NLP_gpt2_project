#!/usr/bin/env python3
"""Dataset helpers for processed emotion and poem JSONL files.

These classes avoid assumptions about the final model/checkpoint format. They
only define the data contract used by later training and evaluation scripts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_LABEL_TO_ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
EMOTION_ID_TO_LABEL = {i: label for label, i in EMOTION_LABEL_TO_ID.items()}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["label"] for row in rows if "label" in row))


class EmotionJsonlDataset(Dataset):
    """Single-label 6-class emotion classification dataset."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = read_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        label_id = int(row.get("label_id", EMOTION_LABEL_TO_ID[row["label"]]))
        return {
            "id": row["id"],
            "text": row["text"],
            "label": row["label"],
            "label_id": label_id,
            "metadata": {
                key: value
                for key, value in row.items()
                if key not in {"id", "text", "label", "label_id"}
            },
        }

    def collate_fn(self, batch: list[dict[str, Any]], tokenizer: Any | None = None, max_length: int | None = None):
        texts = [item["text"] for item in batch]
        labels = torch.LongTensor([item["label_id"] for item in batch])
        output: dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "texts": texts,
            "labels": labels,
            "label_names": [item["label"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }
        if tokenizer is not None:
            kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
            if max_length is not None:
                kwargs["max_length"] = max_length
            encoded = tokenizer(texts, **kwargs)
            output["token_ids"] = encoded["input_ids"]
            output["attention_mask"] = encoded["attention_mask"]
        return output


class PoemContinuationJsonlDataset(Dataset):
    """Poem continuation dataset: first three lines -> remaining lines."""

    def __init__(
        self,
        path: str | Path,
        input_builder: Callable[[dict[str, Any]], str] | None = None,
    ):
        self.path = Path(path)
        self.rows = read_jsonl(self.path)
        self.input_builder = input_builder or (lambda row: row["prefix"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        model_input = self.input_builder(row)
        return {
            "id": row["id"],
            "poem_id": row.get("poem_id"),
            "title": row.get("title", ""),
            "poet": row.get("poet", ""),
            "prefix": row["prefix"],
            "target": row["target"],
            "full_text": row["full_text"],
            "model_input": model_input,
            "metadata": {
                key: value
                for key, value in row.items()
                if key
                not in {"id", "poem_id", "title", "poet", "prefix", "target", "full_text"}
            },
        }

    def collate_fn(
        self,
        batch: list[dict[str, Any]],
        tokenizer: Any | None = None,
        max_length: int | None = None,
        train_on_full_text: bool = True,
    ):
        model_inputs = [item["model_input"] for item in batch]
        full_texts = [item["full_text"] for item in batch]
        output: dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "poem_ids": [item["poem_id"] for item in batch],
            "titles": [item["title"] for item in batch],
            "poets": [item["poet"] for item in batch],
            "prefixes": [item["prefix"] for item in batch],
            "targets": [item["target"] for item in batch],
            "full_texts": full_texts,
            "model_inputs": model_inputs,
            "metadata": [item["metadata"] for item in batch],
        }
        if tokenizer is not None:
            texts_to_encode = full_texts if train_on_full_text else model_inputs
            kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
            if max_length is not None:
                kwargs["max_length"] = max_length
            encoded = tokenizer(texts_to_encode, **kwargs)
            output["token_ids"] = encoded["input_ids"]
            output["attention_mask"] = encoded["attention_mask"]
        return output

