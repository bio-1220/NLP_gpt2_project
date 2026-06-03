#!/usr/bin/env python3
"""Prompt builders for emotion-conditioned poem generation."""

from __future__ import annotations

import random


EN_EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
KO_EMOTION_LABELS = {
    "sadness": "슬픔",
    "joy": "기쁨",
    "love": "애정",
    "anger": "분노",
    "fear": "공포",
    "surprise": "놀람",
}


def emotion_prefix(label: str, language: str = "en") -> str:
    if language == "ko":
        return f"[감정: {KO_EMOTION_LABELS.get(label, label)}]"
    return f"[emotion: {label}]"


def build_emotion_prompt(prefix: str, label: str, language: str = "en") -> str:
    return f"{emotion_prefix(label, language)}\n{prefix}"


def build_random_emotion_prompt(prefix: str, language: str = "en", seed: int | None = None) -> tuple[str, str]:
    rng = random.Random(seed)
    label = rng.choice(EN_EMOTION_LABELS)
    return build_emotion_prompt(prefix, label, language), label


def strip_prompt_prefix(generated_text: str, prompt: str) -> str:
    """Return the continuation part if generated_text starts with prompt."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].lstrip()
    return generated_text

