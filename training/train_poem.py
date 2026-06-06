#!/usr/bin/env python3
"""Phase 2: poem-continuation SFT trainer (baseline E0/K0, plus hooks for E3/K3).

Objective: full-text causal-LM fine-tuning (loss on every non-pad token).
Generation + CHRF live in Phase 3 (evaluate_generation.py); this script only
produces the fine-tuned generator checkpoint.

Split code paths (same philosophy as train_emotion.py):
  - English: SonnetGPT (our custom GPT2Model + tied LM head) on data/sonnets.txt
  - Korean : skt/kogpt2-base-v2 (AutoModelForCausalLM) on poem_ko_*.jsonl

Experiment coverage via flags:
  (no flag)               -> E0 / K0  baseline (pretrained init, no emotion)
  --init_from_emotion CKPT -> E3 / K3  sequential SFT (start from emotion backbone)

Runs:
  python train_poem.py --lang en --use_gpu
  python train_poem.py --lang ko --use_gpu
  python train_poem.py --lang en --init_from_emotion emotion_en_classifier.pt --use_gpu --setup_id E3
Smoke test (CPU):
  python train_poem.py --lang en --limit 8 --epochs 1
"""

import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer import AdamW

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
# Model + data construction (one path per language).
# ----------------------------------------------------------------------------
def _emotion_labels_for(texts, args, device):
    """Return one emotion label per text (predicted or random) for prefixing."""
    import random as _random

    from experiments.data_bridge import EMOTION_LABELS

    if args.emotion_prefix == "pred":
        from experiments.emotion_predict import EmotionPredictor

        return EmotionPredictor(args.emotion_ckpt, device).predict(texts)
    rng = _random.Random(args.seed)
    return [EMOTION_LABELS[rng.randrange(len(EMOTION_LABELS))] for _ in texts]


def _apply_conditioned_text(*datasets):
    """Train on conditioned_full_text (trajectory prompt + poem) when present."""
    for ds in datasets:
        for row in ds.rows:
            if row.get("conditioned_full_text"):
                row["full_text"] = row["conditioned_full_text"]


def build_english(args, device):
    """Return (model, train_loader, dev_loader). Reuses SonnetGPT + SonnetsDataset."""
    from datasets import SonnetsDataset
    from experiments.prompts import emotion_prefix
    from sonnet_generation import SonnetGPT, add_arguments

    model_args = add_arguments(SimpleNamespace(model_size=args.model_size))
    model = SonnetGPT(model_args)

    if args.train_file:  # conditioned JSONL training (sentiment trajectory experiments)
        from experiments.data_bridge import PoemContinuationJsonlDataset

        tokenizer = model.tokenizer  # gpt2 tokenizer with pad=eos
        train_ds = PoemContinuationJsonlDataset(args.train_file)
        dev_ds = PoemContinuationJsonlDataset(args.dev_file or args.train_file.replace("train", "dev"))
        _apply_conditioned_text(train_ds, dev_ds)
        if args.limit:
            train_ds.rows = train_ds.rows[: args.limit]
            dev_ds.rows = dev_ds.rows[: max(1, args.limit // 2)]

        def collate(ds):
            return lambda b: ds.collate_fn(b, tokenizer=tokenizer, max_length=args.max_length, train_on_full_text=True)

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collate(train_ds))
        dev_loader = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate(dev_ds))
        return model, train_loader, dev_loader, model_args

    train_ds = SonnetsDataset(args.en_train_path)
    dev_ds = SonnetsDataset(args.en_dev_path)
    if args.limit:
        train_ds.sonnets = train_ds.sonnets[: args.limit]
        dev_ds.sonnets = dev_ds.sonnets[: max(1, args.limit // 2)]

    if args.emotion_prefix != "none":
        for ds in (train_ds, dev_ds):
            first3 = ["\n".join(s.splitlines()[:3]) for s in ds.sonnets]
            labels = _emotion_labels_for(first3, args, device)
            ds.sonnets = [f"{emotion_prefix(lab, 'en')}\n{s}" for lab, s in zip(labels, ds.sonnets)]

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=train_ds.collate_fn)
    dev_loader = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)
    return model, train_loader, dev_loader, model_args


def build_korean(args, device):
    from transformers import AutoModelForCausalLM

    from experiments.data_bridge import PoemContinuationJsonlDataset
    from experiments.prompts import emotion_prefix
    from train_emotion import build_tokenizer

    tokenizer, _ = build_tokenizer("ko")
    model = AutoModelForCausalLM.from_pretrained(KO_MODEL_NAME)

    train_path = args.train_file or f"{args.data_dir}/poem_ko_train.jsonl"
    dev_path = args.dev_file or (train_path.replace("train", "dev") if args.train_file else f"{args.data_dir}/poem_ko_dev.jsonl")
    train_ds = PoemContinuationJsonlDataset(train_path)
    dev_ds = PoemContinuationJsonlDataset(dev_path)
    _apply_conditioned_text(train_ds, dev_ds)
    if args.limit:
        train_ds.rows = train_ds.rows[: args.limit]
        dev_ds.rows = dev_ds.rows[: max(1, args.limit // 2)]

    if args.emotion_prefix != "none":
        for ds in (train_ds, dev_ds):
            labels = _emotion_labels_for([r["prefix"] for r in ds.rows], args, device)
            for r, lab in zip(ds.rows, labels):
                r["full_text"] = f"{emotion_prefix(lab, 'ko')}\n{r['full_text']}"

    def collate(ds):
        return lambda b: ds.collate_fn(b, tokenizer=tokenizer, max_length=args.max_length, train_on_full_text=True)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collate(train_ds))
    dev_loader = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate(dev_ds))
    return model, train_loader, dev_loader, SimpleNamespace(backbone=KO_MODEL_NAME)


def load_emotion_backbone(model, ckpt_path, lang):
    """Pipeline B: load the emotion classifier's gpt.* backbone into the generator."""
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")["model"]
    backbone_sd = {k[len("gpt.") :]: v for k, v in sd.items() if k.startswith("gpt.")}
    target = model.gpt if lang == "en" else model.transformer
    missing, unexpected = target.load_state_dict(backbone_sd, strict=False)
    print(
        f"init_from_emotion: loaded {len(backbone_sd)} backbone tensors from {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


# ----------------------------------------------------------------------------
# Full-text LM loss (pad positions ignored).
# ----------------------------------------------------------------------------
def lm_loss(model, token_ids, attention_mask, lang):
    if lang == "en":
        logits = model(token_ids, attention_mask)  # (B, T, V)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = token_ids[:, 1:].contiguous().clone()
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
        )
    labels = token_ids.clone()
    labels[attention_mask == 0] = -100
    return model(input_ids=token_ids, attention_mask=attention_mask, labels=labels).loss


@torch.no_grad()
def dev_loss(model, loader, device, lang):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        loss = lm_loss(model, batch["token_ids"].to(device), batch["attention_mask"].to(device), lang)
        total += loss.item()
        n += 1
    return total / max(1, n)


def save_model(model, args, model_args, filepath):
    torch.save(
        {
            "model": model.state_dict(),
            "config": SimpleNamespace(
                lang=args.lang,
                setup_id=args.setup_id,
                objective="full_text_lm",
                init_from_emotion=args.init_from_emotion,
                emotion_prefix=args.emotion_prefix,
                model_size=args.model_size if args.lang == "en" else None,
                backbone="custom-gpt2" if args.lang == "en" else KO_MODEL_NAME,
            ),
            "model_args": model_args,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        },
        filepath,
    )
    print(f"saved model to {filepath}")


def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    if args.lang == "en":
        model, train_loader, dev_loader, model_args = build_english(args, device)
    else:
        model, train_loader, dev_loader, model_args = build_korean(args, device)

    if args.init_from_emotion:
        load_emotion_backbone(model, args.init_from_emotion, args.lang)

    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"[{args.lang}] setup={args.setup_id}  train_batches={len(train_loader)}  -> {args.out}")

    best_dev = float("inf")
    step, stop = 0, False
    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        running, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = lm_loss(model, batch["token_ids"].to(device), batch["attention_mask"].to(device), args.lang)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        dl = dev_loss(model, dev_loader, device, args.lang)
        print(f"Epoch {epoch}: train_loss={running / max(1, n):.3f}  dev_loss={dl:.3f}")
        if dl < best_dev:
            best_dev = dl
            save_model(model, args, model_args, args.out)

    print(f"[{args.lang}] best dev_loss = {best_dev:.3f}  (saved to {args.out})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--setup_id", type=str, default=None, help="experiment id, e.g. E0/K0/E3/K3")
    parser.add_argument("--init_from_emotion", type=str, default=None, help="emotion classifier ckpt for E3/K3")
    parser.add_argument("--emotion_prefix", choices=["none", "pred", "random"], default="none",
                        help="prepend [emotion: X] to training text (E1/E2/K1/K2)")
    parser.add_argument("--emotion_ckpt", type=str, default=None, help="classifier ckpt for --emotion_prefix pred")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--train_file", type=str, default=None,
                        help="conditioned JSONL for training (uses conditioned_full_text when present)")
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--en_train_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--en_dev_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (0=use epochs)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=256, help="Korean full_text truncation length")
    parser.add_argument("--model_size", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large"])
    parser.add_argument("--limit", type=int, default=0, help="truncate splits for smoke testing")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    if args.setup_id is None:
        args.setup_id = "E0" if args.lang == "en" else "K0"
    if args.out is None:
        args.out = f"checkpoints/poem_{args.lang}_{args.setup_id}.pt"
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
