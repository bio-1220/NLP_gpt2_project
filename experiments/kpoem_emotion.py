#!/usr/bin/env python3
"""Collapse KPoEM fine-grained emotion metadata into the shared 6-class space."""

from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.data_bridge import EMOTION_LABEL_TO_ID


STRICT_KPOEM_TO_SIX = {
    "sadness": {
        "슬픔",
        "서러움",
        "절망",
    },
    "joy": {
        "기쁨",
        "행복",
        "즐거움/신남",
    },
    "love": {
        "아껴주는",
        "환영/호의",
    },
    "anger": {
        "화남/분노",
        "짜증",
        "증오/혐오",
    },
    "fear": {
        "공포/무서움",
        "불안/걱정",
    },
    "surprise": {
        "놀람",
        "경악",
        "당황/난처",
    },
}


POETIC_KPOEM_TO_SIX = {
    "sadness": {
        "슬픔",
        "서러움",
        "절망",
        "힘듦/지침",
        "안타까움/실망",
        "패배/자기혐오",
        "죄책감",
        "부끄러움",
        "비장함",
    },
    "joy": {
        "기쁨",
        "행복",
        "즐거움/신남",
        "감동/감탄",
        "뿌듯함",
        "고마움",
        "편안/쾌적",
        "기대감",
        "안심/신뢰",
    },
    "love": {
        "아껴주는",
        "환영/호의",
        "흐뭇함(귀여움/예쁨)",
        "존경",
        "불쌍함/연민",
    },
    "anger": {
        "화남/분노",
        "짜증",
        "불평/불만",
        "증오/혐오",
        "지긋지긋",
        "한심함",
        "어이없음",
        "역겨움/징그러움",
    },
    "fear": {
        "공포/무서움",
        "불안/걱정",
        "의심/불신",
        "부담/안_내킴",
    },
    "surprise": {
        "놀람",
        "경악",
        "당황/난처",
        "신기함/관심",
        "깨달음",
    },
}


MAPPINGS = {
    "strict": STRICT_KPOEM_TO_SIX,
    "poetic": POETIC_KPOEM_TO_SIX,
}


def mapping_to_lookup(mode: str = "poetic") -> dict[str, str]:
    if mode not in MAPPINGS:
        raise ValueError(f"Unknown KPoEM emotion mapping mode: {mode}")
    return {source: target for target, sources in MAPPINGS[mode].items() for source in sources}


def parse_emotion_values(metadata: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for value in metadata.values():
        for label in str(value).split(","):
            label = label.strip()
            if label:
                labels.append(label)
    return labels


def collapse_kpoem_emotions(
    metadata: dict[str, Any],
    mode: str = "poetic",
    min_count: int = 2,
    require_unique_top: bool = True,
) -> dict[str, Any] | None:
    """Return the dominant 6-class KPoEM emotion label, or None if ambiguous.

    The returned dictionary keeps the original fine-grained labels and collapsed
    counts so diagnostic experiments can explain how oracle labels were chosen.
    """
    lookup = mapping_to_lookup(mode)
    source_labels = parse_emotion_values(metadata)
    counts = Counter(lookup[label] for label in source_labels if label in lookup)
    if not counts:
        return None
    most_common = counts.most_common()
    if most_common[0][1] < min_count:
        return None
    if require_unique_top and len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return None
    label = most_common[0][0]
    return {
        "label": label,
        "label_id": EMOTION_LABEL_TO_ID[label],
        "label_counts": dict(counts),
        "source_labels": source_labels,
        "mapping_mode": mode,
        "min_count": min_count,
    }

