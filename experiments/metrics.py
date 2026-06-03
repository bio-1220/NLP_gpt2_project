#!/usr/bin/env python3
"""Evaluation helpers for emotion classification and poem generation."""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from sacrebleu.metrics import CHRF
except ModuleNotFoundError:
    CHRF = None


def _char_ngrams(text: str, n: int) -> Counter[str]:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _fallback_sentence_chrf(hypothesis: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    precisions = []
    recalls = []
    for n in range(1, max_order + 1):
        hyp = _char_ngrams(hypothesis, n)
        ref = _char_ngrams(reference, n)
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / max(1, sum(hyp.values())))
        recalls.append(overlap / max(1, sum(ref.values())))
    precision = sum(precisions) / max_order
    recall = sum(recalls) / max_order
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return 100.0 * (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def chrf_corpus_score(hypotheses: list[str], references: list[str]) -> float:
    max_len = min(len(hypotheses), len(references))
    if max_len == 0:
        return 0.0
    if CHRF is None:
        scores = chrf_sentence_scores(hypotheses[:max_len], references[:max_len])
        return sum(scores) / len(scores)
    chrf = CHRF()
    score = chrf.corpus_score(hypotheses[:max_len], [references[:max_len]])
    return float(score.score)


def chrf_sentence_scores(hypotheses: list[str], references: list[str]) -> list[float]:
    if CHRF is None:
        return [
            _fallback_sentence_chrf(hypothesis, reference)
            for hypothesis, reference in zip(hypotheses, references)
        ]
    chrf = CHRF()
    return [
        float(chrf.sentence_score(hypothesis, [reference]).score)
        for hypothesis, reference in zip(hypotheses, references)
    ]


def accuracy(predictions: list[str], labels: list[str]) -> float:
    max_len = min(len(predictions), len(labels))
    if max_len == 0:
        return 0.0
    correct = sum(pred == gold for pred, gold in zip(predictions[:max_len], labels[:max_len]))
    return correct / max_len


def distribution(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


def generation_records(
    examples: list[dict[str, Any]],
    hypotheses: list[str],
    condition_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    sentence_scores = chrf_sentence_scores(hypotheses, [example["target"] for example in examples])
    for i, (example, hypothesis, score) in enumerate(zip(examples, hypotheses, sentence_scores)):
        row = {
            "id": example["id"],
            "prefix": example["prefix"],
            "target": example["target"],
            "hypothesis": hypothesis,
            "chrf": score,
        }
        if condition_labels is not None:
            row["condition_label"] = condition_labels[i]
        for key in ["poem_id", "title", "poet"]:
            if key in example:
                row[key] = example[key]
        records.append(row)
    return records
