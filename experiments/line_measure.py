#!/usr/bin/env python3
"""Line-level emotion/sentiment measurers for trajectory evaluation.

The same measurer is applied to BOTH the generated continuation and the
reference continuation, so measurer bias largely cancels in MSE/corr/volta
comparisons.

  EN: pretrained SST-2 DistilBERT (external, no leakage) -> score in [-1, 1]
      score = P(positive) - P(negative)
  KO: our Phase-1 KOTE emotion classifier per line -> 6-class label,
      mapped to valence in {-1, 0, +1} via metrics.EMOTION_VALENCE
"""

from __future__ import annotations

import torch

from experiments.metrics import EMOTION_VALENCE

EN_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


class EnglishLineSentiment:
    """Continuous per-line sentiment scorer (pretrained SST-2, leakage-free)."""

    def __init__(self, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(EN_SENTIMENT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(EN_SENTIMENT_MODEL)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score_lines(self, lines: list[str], batch_size: int = 32) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)  # [neg, pos]
            scores.extend((probs[:, 1] - probs[:, 0]).cpu().tolist())
        return scores

    def trajectory(self, text: str) -> list[float]:
        lines = split_lines(text)
        return self.score_lines(lines) if lines else []


class KoreanLineEmotion:
    """Per-line 6-class emotion labels via the Phase-1 classifier + valence map."""

    def __init__(self, ckpt_path, device):
        from experiments.emotion_predict import EmotionPredictor

        self.predictor = EmotionPredictor(ckpt_path, device)

    def label_lines(self, lines: list[str]) -> list[str]:
        return self.predictor.predict(lines) if lines else []

    def trajectory(self, text: str) -> list[float]:
        labels = self.label_lines(split_lines(text))
        return [EMOTION_VALENCE[label] for label in labels]

    def label_trajectory(self, text: str) -> list[str]:
        return self.label_lines(split_lines(text))
