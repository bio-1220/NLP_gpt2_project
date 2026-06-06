#!/usr/bin/env python3
"""Phase 1: 6-class emotion classification fine-tuning (SFT).

Two separate code paths (see phase1 design decision: split en/ko):
  - English: our custom GPT2Model backbone (models/gpt2.py) + linear head.
  - Korean : HuggingFace skt/kogpt2-base-v2 loaded as-is + linear head.

The saved checkpoint is reused by:
  - Pipeline A (E1/K1): predict the emotion of a poem's first 3 lines.
  - Pipeline B (E3/K3): transfer the fine-tuned backbone into the poem generator.

Run:
  python train_emotion.py --lang en --use_gpu
  python train_emotion.py --lang ko --use_gpu
Smoke test on CPU:
  python train_emotion.py --lang en --limit 200 --epochs 1
"""

import argparse
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_bridge import (
    EMOTION_LABELS,
    EMOTION_LABEL_TO_ID,
    EmotionJsonlDataset,
)
from optimizer import AdamW

NUM_LABELS = len(EMOTION_LABELS)  # 6
KO_MODEL_NAME = "skt/kogpt2-base-v2"
TQDM_DISABLE = False


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# Models (one per language; deliberately not abstracted into a shared class).
# ----------------------------------------------------------------------------
class GPT2EmotionClassifier(nn.Module):
    """English path: custom GPT2Model backbone + linear head."""

    def __init__(self, hidden_dropout_prob=0.3):
        super().__init__()
        from models.gpt2 import GPT2Model

        self.gpt = GPT2Model.from_pretrained()  # standard gpt2 weights
        for param in self.gpt.parameters():
            param.requires_grad = True  # full-model fine-tuning
        self.hidden_size = 768
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs["last_token"])  # last non-pad token
        return self.classifier(pooled)


class KoGPT2EmotionClassifier(nn.Module):
    """Korean path: HuggingFace skt/kogpt2-base-v2 backbone + linear head."""

    def __init__(self, hidden_dropout_prob=0.3):
        super().__init__()
        from transformers import AutoModel

        self.gpt = AutoModel.from_pretrained(KO_MODEL_NAME)
        for param in self.gpt.parameters():
            param.requires_grad = True  # full-model fine-tuning
        self.hidden_size = self.gpt.config.hidden_size  # 768
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        lhs = outputs.last_hidden_state  # (B, T, H)
        last_idx = attention_mask.sum(dim=1) - 1  # last non-pad index (right padding)
        pooled = lhs[torch.arange(lhs.shape[0], device=lhs.device), last_idx]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_tokenizer(lang):
    if lang == "en":
        from transformers import GPT2Tokenizer

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return tok, "gpt2"
    else:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast.from_pretrained(
            KO_MODEL_NAME,
            bos_token="</s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        tok.padding_side = "right"
        return tok, KO_MODEL_NAME


def build_model(lang, hidden_dropout_prob):
    if lang == "en":
        return GPT2EmotionClassifier(hidden_dropout_prob)
    return KoGPT2EmotionClassifier(hidden_dropout_prob)


# ----------------------------------------------------------------------------
# Class weights: mild (sqrt-inverse-frequency), normalized to mean 1, capped.
# Goal is to avoid majority-class collapse, not to chase peak accuracy.
# ----------------------------------------------------------------------------
def compute_class_weights(dataset, mode="sqrt", cap=5.0):
    counts = torch.zeros(NUM_LABELS)
    for row in dataset.rows:
        label_id = int(row.get("label_id", EMOTION_LABEL_TO_ID[row["label"]]))
        counts[label_id] += 1
    counts = counts.clamp(min=1.0)
    if mode == "none":
        return None, counts
    freq = counts / counts.sum()
    weights = 1.0 / torch.sqrt(freq) if mode == "sqrt" else 1.0 / freq
    weights = weights / weights.mean()  # center around 1
    weights = weights.clamp(max=cap)
    return weights, counts


def make_loader(path, tokenizer, args, shuffle, limit=0):
    dataset = EmotionJsonlDataset(path)
    if limit and limit > 0:
        dataset.rows = dataset.rows[:limit]
    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=args.batch_size,
        collate_fn=lambda b: dataset.collate_fn(b, tokenizer=tokenizer, max_length=args.max_length),
    )
    return dataset, loader


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval", disable=TQDM_DISABLE):
        logits = model(batch["token_ids"].to(device), batch["attention_mask"].to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(batch["labels"].numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    return acc, macro_f1, per_class.tolist(), cm.tolist()


def save_model(model, optimizer, args, model_config, filepath):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "args": args,
            "model_config": model_config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        },
        filepath,
    )
    print(f"saved model to {filepath}")


def format_per_class(per_class):
    return ", ".join(f"{lab}={v:.3f}" for lab, v in zip(EMOTION_LABELS, per_class))


def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    tokenizer, tokenizer_name = build_tokenizer(args.lang)

    base = f"{args.data_dir}/emotion_{args.lang}"
    train_dataset, train_loader = make_loader(f"{base}_train.jsonl", tokenizer, args, True, args.limit)
    _, dev_loader = make_loader(f"{base}_dev.jsonl", tokenizer, args, False, args.limit)

    class_weights, counts = compute_class_weights(train_dataset, args.weight_mode, args.weight_cap)
    print(f"[{args.lang}] train n={len(train_dataset)}  counts={counts.int().tolist()}")
    print(f"  labels = {EMOTION_LABELS}")
    if class_weights is not None:
        print(f"  class_weights = {[round(w, 3) for w in class_weights.tolist()]}")
    if class_weights is not None:
        class_weights = class_weights.to(device)

    model = build_model(args.lang, args.hidden_dropout_prob).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model_config = SimpleNamespace(
        lang=args.lang,
        num_labels=NUM_LABELS,
        labels=EMOTION_LABELS,
        hidden_size=model.hidden_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        backbone="custom-gpt2" if args.lang == "en" else KO_MODEL_NAME,
        tokenizer_name=tokenizer_name,
        max_length=args.max_length,
    )

    best_macro_f1 = -1.0
    step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        running, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            optimizer.zero_grad()
            logits = model(batch["token_ids"].to(device), batch["attention_mask"].to(device))
            loss = F.cross_entropy(logits, batch["labels"].to(device), weight=class_weights)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        acc, macro_f1, per_class, cm = evaluate(dev_loader, model, device)
        print(
            f"Epoch {epoch}: train_loss={running / max(1, n_batches):.3f}  "
            f"dev_acc={acc:.3f}  dev_macroF1={macro_f1:.3f}"
        )
        print(f"  per-class F1: {format_per_class(per_class)}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_model(model, optimizer, args, model_config, args.out)
            with open(args.out + ".eval.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lang": args.lang,
                        "epoch": epoch,
                        "dev_acc": acc,
                        "dev_macro_f1": macro_f1,
                        "per_class_f1": dict(zip(EMOTION_LABELS, per_class)),
                        "confusion_matrix": cm,
                        "labels": EMOTION_LABELS,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"[{args.lang}] best dev macro-F1 = {best_macro_f1:.3f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (0=use epochs)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--weight_mode", choices=["sqrt", "inverse", "none"], default="sqrt")
    parser.add_argument("--weight_cap", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="truncate splits for smoke testing")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = f"checkpoints/emotion/emotion_{args.lang}_classifier.pt"
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
