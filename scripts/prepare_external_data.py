#!/usr/bin/env python3
"""Download and preprocess external datasets for emotion-aware poem generation.

Outputs JSONL files under data/processed by default:
  - emotion_en_{train,dev,test}.jsonl
  - emotion_ko_{train,dev,test}.jsonl
  - poem_ko_{train,dev,test}.jsonl
"""

import argparse
import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SEED = 11711

EN_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EN_LABEL_TO_ID = {label: i for i, label in enumerate(EN_LABELS)}

KO_PROMPT_LABELS = {
    "sadness": "슬픔",
    "joy": "기쁨",
    "love": "애정",
    "anger": "분노",
    "fear": "공포",
    "surprise": "놀람",
}

KOTE_TO_SIX = {
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

KOTE_LABEL_TO_TARGET = {
    source: target for target, sources in KOTE_TO_SIX.items() for source in sources
}

DAIR_HF_SPLITS = {
    "train": "split/train-00000-of-00001.parquet",
    "dev": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}

KOTE_URLS = {
    "hf_raw": "https://huggingface.co/datasets/searle-j/kote/resolve/main/raw.json",
    "github_train": "https://raw.githubusercontent.com/searle-j/KOTE/main/train.tsv",
    "github_test": "https://raw.githubusercontent.com/searle-j/KOTE/main/test.tsv",
}

KPOEM_URL = (
    "https://huggingface.co/datasets/AKS-DHLAB/KPoEM/resolve/main/"
    "KPoEM_poem_dataset_v4.tsv"
)
KPOEM_LINE_URL = (
    "https://huggingface.co/datasets/AKS-DHLAB/KPoEM/resolve/main/"
    "KPoEM_line_dataset_v4.tsv"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    ensure_dir(path.parent)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def fetch_hf_rows(dataset: str, config: str, split: str, batch_size: int = 100) -> list[dict[str, Any]]:
    """Fetch a public HuggingFace dataset through the datasets-server rows API."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": batch_size,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    return " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").split())


def normalize_poem_text(text: Any) -> str:
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.strip().split()) for line in raw.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def preprocess_dair(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    stats = {}
    for split, hf_path in DAIR_HF_SPLITS.items():
        raw_path = raw_dir / "dair-ai-emotion" / f"{split}.parquet"
        rows = []
        download(f"https://huggingface.co/datasets/dair-ai/emotion/resolve/main/{hf_path}", raw_path)
        source_rows = pd.read_parquet(raw_path).to_dict(orient="records")

        for i, item in enumerate(source_rows):
            text = item["text"]
            label_raw = item["label"]
            if isinstance(label_raw, int):
                label_id = label_raw
                label = EN_LABELS[label_id]
            elif str(label_raw).isdigit():
                label_id = int(label_raw)
                label = EN_LABELS[label_id]
            else:
                label = str(label_raw)
                label_id = EN_LABEL_TO_ID[label]

            rows.append(
                {
                    "id": f"en-{split}-{i}",
                    "text": normalize_text(text),
                    "label": label,
                    "label_id": label_id,
                }
            )

        write_jsonl(out_dir / f"emotion_en_{split}.jsonl", rows)
        stats[f"emotion_en_{split}"] = len(rows)
    return stats


def parse_label_value(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        labels = []
        for v in value.values():
            labels.extend(parse_label_value(v))
        return labels

    text = str(value).strip()
    if not text:
        return []

    for sep in ["|", ",", ";"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, dict):
            return [str(k).strip() for k, v in parsed.items() if v and str(k).strip()]
    except json.JSONDecodeError:
        pass

    return [text]


def parse_rater_labels(value: Any) -> list[list[str]]:
    """Return labels grouped by annotator/rater when available."""
    if isinstance(value, dict):
        groups = []
        for labels in value.values():
            parsed = parse_label_value(labels)
            if parsed:
                groups.append(parsed)
        return groups
    parsed = parse_label_value(value)
    return [parsed] if parsed else []


def extract_text_and_labels(record: dict[str, Any]) -> tuple[str | None, list[str]]:
    text_keys = [
        "text",
        "sentence",
        "comment",
        "content",
        "utterance",
        "문장",
        "댓글",
    ]
    label_keys = [
        "labels",
        "label",
        "emotion",
        "emotions",
        "class",
        "감정",
        "labels_str",
    ]

    text = None
    for key in text_keys:
        if key in record and record[key] is not None:
            text = normalize_text(record[key])
            break

    labels = []
    for key in label_keys:
        if key in record:
            labels = parse_label_value(record[key])
            if labels:
                break

    if not labels:
        labels = [
            str(k).strip()
            for k, v in record.items()
            if str(k).strip() in KOTE_LABEL_TO_TARGET and v in (1, True, "1", "true", "True")
        ]

    return text, labels


def extract_text_and_rater_labels(record: dict[str, Any]) -> tuple[str | None, list[list[str]]]:
    text, _ = extract_text_and_labels(record)
    label_keys = [
        "labels",
        "label",
        "emotion",
        "emotions",
        "class",
        "감정",
        "labels_str",
    ]
    for key in label_keys:
        if key in record:
            groups = parse_rater_labels(record[key])
            if groups:
                return text, groups
    labels = [
        str(k).strip()
        for k, v in record.items()
        if str(k).strip() in KOTE_LABEL_TO_TARGET and v in (1, True, "1", "true", "True")
    ]
    return text, [labels] if labels else []


def load_kote_records(raw_dir: Path) -> list[dict[str, Any]]:
    raw_path = raw_dir / "kote" / "raw.json"
    try:
        download(KOTE_URLS["hf_raw"], raw_path)
        content = raw_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ["data", "train", "rows"]:
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            return [
                {"id": record_id, **record}
                for record_id, record in parsed.items()
                if isinstance(record, dict)
            ]
    except Exception as exc:
        print(f"[warn] Could not load KOTE raw.json from HuggingFace: {exc}")

    records = []
    for split, url in [("train", KOTE_URLS["github_train"]), ("test", KOTE_URLS["github_test"])]:
        tsv_path = raw_dir / "kote" / f"{split}.tsv"
        download(url, tsv_path)
        with tsv_path.open("r", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = any(name in sample.splitlines()[0].lower() for name in ["text", "label"])
            if has_header:
                reader = csv.DictReader(f, delimiter="\t")
                records.extend(dict(row) for row in reader)
            else:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        records.append({"text": row[0], "labels": row[1]})
    return records


def preprocess_kote(raw_dir: Path, out_dir: Path, seed: int) -> dict[str, int]:
    records = load_kote_records(raw_dir)
    kept = []
    dropped_no_match = 0
    dropped_multi_target = 0
    dropped_low_agreement = 0
    source_label_counts = Counter()

    for idx, record in enumerate(records):
        text, rater_label_groups = extract_text_and_rater_labels(record)
        if not text:
            continue
        labels = [label for group in rater_label_groups for label in group]
        source_label_counts.update(labels)

        target_votes = Counter()
        for group in rater_label_groups:
            group_targets = {
                KOTE_LABEL_TO_TARGET[label]
                for label in group
                if label in KOTE_LABEL_TO_TARGET
            }
            if len(group_targets) == 1:
                target_votes.update(group_targets)

        target_counts = Counter(
            KOTE_LABEL_TO_TARGET[label] for label in labels if label in KOTE_LABEL_TO_TARGET
        )
        if not target_counts:
            dropped_no_match += 1
            continue
        if not target_votes:
            dropped_multi_target += 1
            continue

        most_common = target_votes.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            dropped_multi_target += 1
            continue
        if most_common[0][1] < 3:
            dropped_low_agreement += 1
            continue
        label = most_common[0][0]
        kept.append(
            {
                "id": str(record.get("id", f"ko-emotion-{idx}")),
                "text": text,
                "label": label,
                "label_id": EN_LABEL_TO_ID[label],
                "source_labels": labels,
                "target_label_counts": dict(target_counts),
                "target_rater_votes": dict(target_votes),
                "prompt_label_ko": KO_PROMPT_LABELS[label],
            }
        )

    rng = random.Random(seed)
    rng.shuffle(kept)

    n = len(kept)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    splits = {
        "train": kept[:n_train],
        "dev": kept[n_train : n_train + n_dev],
        "test": kept[n_train + n_dev :],
    }

    for split, rows in splits.items():
        write_jsonl(out_dir / f"emotion_ko_{split}.jsonl", rows)

    stats = {f"emotion_ko_{split}": len(rows) for split, rows in splits.items()}
    stats["emotion_ko_dropped_no_match"] = dropped_no_match
    stats["emotion_ko_dropped_multi_target"] = dropped_multi_target
    stats["emotion_ko_dropped_low_agreement"] = dropped_low_agreement

    mapping_path = out_dir / "emotion_ko_label_mapping.json"
    write_jsonl(
        mapping_path,
        [
            {
                "target_label": target,
                "source_labels": sorted(sources),
                "label_id": EN_LABEL_TO_ID[target],
                "prompt_label_ko": KO_PROMPT_LABELS[target],
            }
            for target, sources in KOTE_TO_SIX.items()
        ],
    )
    return stats


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {column.lower().replace(" ", "").replace("_", ""): column for column in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    for column in columns:
        low = column.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return column
    return None


def split_lines(text: str) -> list[str]:
    return [line for line in normalize_poem_text(text).split("\n") if line.strip()]


def preprocess_kpoem(raw_dir: Path, out_dir: Path, seed: int) -> dict[str, int]:
    poem_raw_path = raw_dir / "kpoem" / "KPoEM_poem_dataset_v4.tsv"
    line_raw_path = raw_dir / "kpoem" / "KPoEM_line_dataset_v4.tsv"
    download(KPOEM_URL, poem_raw_path)
    download(KPOEM_LINE_URL, line_raw_path)
    poem_df = pd.read_csv(poem_raw_path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
    line_df = pd.read_csv(line_raw_path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)

    columns = list(line_df.columns)
    text_col = "text"
    title_col = "title"
    poet_col = "poet"
    line_id_col = "line_id"
    poem_id_col = "poem_id"
    emotion_cols = [col for col in poem_df.columns if col.startswith("annotator_")]
    poem_meta = {}
    for _, record in poem_df.iterrows():
        poem_id = record.get(poem_id_col)
        if pd.isna(poem_id):
            continue
        poem_meta[int(poem_id)] = {
            col: record[col]
            for col in emotion_cols
            if col in record and not pd.isna(record[col])
        }

    rows = []
    dropped_short = 0
    for poem_id, group in line_df.groupby(poem_id_col):
        group = group.sort_values(line_id_col)
        lines = [normalize_text(text) for text in group[text_col].tolist()]
        lines = [line for line in lines if line]
        if len(lines) < 4:
            dropped_short += 1
            continue
        prefix = "\n".join(lines[:3])
        target = "\n".join(lines[3:])
        full_text = "\n".join(lines)
        first = group.iloc[0]
        rows.append(
            {
                "id": f"kpoem-{int(poem_id)}",
                "poem_id": int(poem_id),
                "title": "" if pd.isna(first.get(title_col)) else str(first[title_col]),
                "poet": "" if pd.isna(first.get(poet_col)) else str(first[poet_col]),
                "prefix": prefix,
                "target": target,
                "full_text": full_text,
                "num_lines": len(lines),
                "emotion_metadata": poem_meta.get(int(poem_id), {}),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    splits = {
        "train": rows[:n_train],
        "dev": rows[n_train : n_train + n_dev],
        "test": rows[n_train + n_dev :],
    }
    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"poem_ko_{split}.jsonl", split_rows)

    schema = {
        "line_columns": columns,
        "poem_columns": list(poem_df.columns),
        "text_col": text_col,
        "title_col": title_col,
        "poet_col": poet_col,
        "emotion_cols": emotion_cols,
        "dropped_short": dropped_short,
    }
    (out_dir / "poem_ko_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {f"poem_ko_{split}": len(split_rows) for split, split_rows in splits.items()}
    stats["poem_ko_dropped_short"] = dropped_short
    return stats


def summarize_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    labels = Counter(row.get("label") for row in rows if "label" in row)
    return {
        "file": str(path),
        "rows": len(rows),
        "labels": dict(labels),
        "sample": rows[0] if rows else None,
    }


def write_summary(out_dir: Path, stats: dict[str, int]) -> None:
    summaries = []
    for path in sorted(out_dir.glob("*.jsonl")):
        if path.name.endswith("label_mapping.json"):
            continue
        summaries.append(summarize_jsonl(path))
    summary = {"stats": stats, "files": summaries}
    (out_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--only",
        choices=["all", "dair", "kote", "kpoem"],
        default="all",
    )
    args = parser.parse_args()

    ensure_dir(args.raw_dir)
    ensure_dir(args.out_dir)

    stats: dict[str, int] = {}
    if args.only in ("all", "dair"):
        stats.update(preprocess_dair(args.raw_dir, args.out_dir))
    if args.only in ("all", "kote"):
        stats.update(preprocess_kote(args.raw_dir, args.out_dir, args.seed))
    if args.only in ("all", "kpoem"):
        stats.update(preprocess_kpoem(args.raw_dir, args.out_dir, args.seed))

    write_summary(args.out_dir, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
