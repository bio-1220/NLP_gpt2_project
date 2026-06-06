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


# ---------------------------------------------------------------------------
# Diversity / trajectory metrics (full-tier evaluation)
# ---------------------------------------------------------------------------

# valence map for the shared 6-class emotion space (KO trajectory comparison)
EMOTION_VALENCE = {"joy": 1.0, "love": 1.0, "surprise": 0.0, "sadness": -1.0, "anger": -1.0, "fear": -1.0}


def distinct_n(text: str, n: int = 2) -> float:
    """Unique-n-gram ratio over whitespace tokens (1.0 = no repetition)."""
    tokens = text.split()
    if len(tokens) < n:
        return 1.0 if tokens else 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def trajectory_mse(gen: list[float], ref: list[float]) -> float | None:
    n = min(len(gen), len(ref))
    if n == 0:
        return None
    return sum((g - r) ** 2 for g, r in zip(gen[:n], ref[:n])) / n


def trajectory_correlation(gen: list[float], ref: list[float]) -> float | None:
    """Pearson correlation over aligned positions (None if undefined)."""
    n = min(len(gen), len(ref))
    if n < 2:
        return None
    g, r = gen[:n], ref[:n]
    mg, mr = sum(g) / n, sum(r) / n
    cov = sum((a - mg) * (b - mr) for a, b in zip(g, r))
    vg = sum((a - mg) ** 2 for a in g) ** 0.5
    vr = sum((b - mr) ** 2 for b in r) ** 0.5
    if vg == 0.0 or vr == 0.0:
        return None
    return cov / (vg * vr)


def volta_position(trajectory: list[float]) -> int | None:
    """1-based index of the largest line-to-line sentiment change."""
    if len(trajectory) < 2:
        return None
    deltas = [abs(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)]
    return deltas.index(max(deltas)) + 1


def volta_distance(gen: list[float], ref: list[float]) -> int | None:
    vg, vr = volta_position(gen), volta_position(ref)
    if vg is None or vr is None:
        return None
    return abs(vg - vr)


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
