#!/usr/bin/env python3
"""Phase 3: generate poem continuations and score them with CHRF.

Loads a poem checkpoint (from train_poem.py), generates the continuation of each
eval poem's first 3 lines, and computes CHRF against the reference continuation.

Eval sets (original CS224n / proposal splits, kept as-is):
  EN: data/sonnets_held_out_dev.txt (12, first-3-lines) vs TRUE_sonnets_held_out_dev.txt (full)
  KO: data/processed/poem_ko_test.jsonl (47; prefix/target/full_text)

Metrics:
  - continuation CHRF (primary): generated tail vs reference tail (lines 4+)
  - full-poem CHRF (EN extra): (prefix + generated) vs full reference

Decoding: fixed-seed nucleus sampling (temp/top_p), N samples per prompt, CHRF
averaged. ALL setups must share identical decoding for fair comparison.

Runs:
  python evaluate_generation.py --lang en --ckpt poem_en_E0.pt --setup_id E0 --use_gpu
  python evaluate_generation.py --lang ko --ckpt poem_ko_K0.pt --setup_id K0 --use_gpu
"""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import chrf_corpus_score, chrf_sentence_scores
from experiments.prompts import build_emotion_prompt, build_random_emotion_prompt

KO_MODEL_NAME = "skt/kogpt2-base-v2"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Prompt building (baseline = none; emotion modes are the Phase 4 hook).
# ----------------------------------------------------------------------------
def build_prompt(prefix, lang, mode, label=None, seed=0):
    if mode == "none":
        return prefix, None
    if mode in ("fixed", "pred"):
        # 'pred' labels are precomputed per example and passed in as `label`.
        return build_emotion_prompt(prefix, label, language=lang), label
    if mode == "random":
        prompt, chosen = build_random_emotion_prompt(prefix, language=lang, seed=seed)
        return prompt, chosen
    raise ValueError(f"unknown emotion_prefix mode: {mode}")


# ----------------------------------------------------------------------------
# Eval data loading.
# ----------------------------------------------------------------------------
def load_eval_examples(lang, args):
    examples = []
    if args.eval_file:  # conditioned JSONL (trajectory experiments); lang-agnostic
        import json as _json

        with open(args.eval_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = _json.loads(line)
                ex = {
                    "id": r["id"],
                    "prefix": r["prefix"],
                    "target": r["target"],
                    "full_text": r["full_text"],
                }
                if r.get("model_input"):
                    ex["model_input"] = r["model_input"]
                for k in ["poem_id", "title", "poet", "sonnet_id", "trajectory_policy", "sentiment_policy"]:
                    if k in r:
                        ex[k] = r[k]
                examples.append(ex)
        if args.limit:
            examples = examples[: args.limit]
        return examples
    if lang == "en":
        from datasets import SonnetsDataset

        prefixes = [t for _, t in SonnetsDataset(args.en_input_path)]
        golds = [t for _, t in SonnetsDataset(args.en_gold_path)]
        for i, (prefix, gold_full) in enumerate(zip(prefixes, golds)):
            cont = "\n".join(gold_full.splitlines()[3:])
            examples.append(
                {"id": f"en-dev-{i}", "prefix": prefix.strip(), "target": cont.strip(), "full_text": gold_full.strip()}
            )
    else:
        from experiments.data_bridge import PoemContinuationJsonlDataset

        ds = PoemContinuationJsonlDataset(args.ko_eval_path)
        for ex in ds:
            examples.append(
                {
                    "id": ex["id"],
                    "prefix": ex["prefix"],
                    "target": ex["target"],
                    "full_text": ex["full_text"],
                    "poem_id": ex.get("poem_id"),
                    "title": ex.get("title", ""),
                    "poet": ex.get("poet", ""),
                }
            )
    if args.limit:
        examples = examples[: args.limit]
    return examples


# ----------------------------------------------------------------------------
# Model loading + generation (one path per language).
# ----------------------------------------------------------------------------
def load_model(lang, ckpt_path, device):
    saved = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if lang == "en":
        from sonnet_generation import SonnetGPT

        model = SonnetGPT(saved["model_args"])
        model.load_state_dict(saved["model"])
        model.to(device).eval()
        return model, model.tokenizer
    else:
        from transformers import AutoModelForCausalLM

        from train_emotion import build_tokenizer

        tokenizer, _ = build_tokenizer("ko")
        model = AutoModelForCausalLM.from_pretrained(KO_MODEL_NAME)
        model.load_state_dict(saved["model"])
        model.to(device).eval()
        return model, tokenizer


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt, lang, device, args):
    if lang == "en":
        enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
        prompt_len = enc["input_ids"].shape[1]
        token_ids, _ = model.generate(
            enc["input_ids"], temperature=args.temperature, top_p=args.top_p, max_length=args.max_new_tokens
        )
        new_tokens = token_ids[0][prompt_len:].cpu().tolist()
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][prompt_len:].cpu().tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model, tokenizer = load_model(args.lang, args.ckpt, device)
    examples = load_eval_examples(args.lang, args)
    print(f"[{args.lang}] setup={args.setup_id}  eval_n={len(examples)}  n_samples={args.n_samples}")

    # For 'pred', predict each prefix's emotion once with the Phase 1 classifier.
    pred_labels = None
    if args.emotion_prefix == "pred":
        from experiments.emotion_predict import EmotionPredictor

        predictor = EmotionPredictor(args.emotion_ckpt, device)
        pred_labels = predictor.predict([ex["prefix"] for ex in examples])
        del predictor

    records = []
    for i, ex in enumerate(tqdm(examples, desc="generate")):
        cont_scores, full_scores, hyps, used_label = [], [], [], None
        fixed_label = pred_labels[i] if pred_labels is not None else args.emotion_label
        base_input = ex.get("model_input") or ex["prefix"]  # conditioned prompt when provided
        for s in range(args.n_samples):
            seed_everything(args.seed + 1000 * s + i)
            prompt, label = build_prompt(base_input, args.lang, args.emotion_prefix, fixed_label, seed=args.seed + s)
            used_label = label
            cont = generate_continuation(model, tokenizer, prompt, args.lang, device, args)
            hyps.append(cont)
            cont_scores.append(chrf_sentence_scores([cont], [ex["target"]])[0])
            full_hyp = ex["prefix"] + "\n" + cont
            full_scores.append(chrf_sentence_scores([full_hyp], [ex["full_text"]])[0])

        rec = {
            "id": ex["id"],
            "setup_id": args.setup_id,
            "prefix": ex["prefix"],
            "target": ex["target"],
            "hypothesis": hyps[0],
            "hypotheses": hyps,
            "chrf_continuation_mean": sum(cont_scores) / len(cont_scores),
            "chrf_full_mean": sum(full_scores) / len(full_scores),
            "n_samples": args.n_samples,
        }
        if used_label is not None:
            rec["condition_label"] = used_label
        for k in ["poem_id", "title", "poet", "sonnet_id", "trajectory_policy", "sentiment_policy"]:
            if k in ex:
                rec[k] = ex[k]
        records.append(rec)

    cont_corpus = sum(r["chrf_continuation_mean"] for r in records) / len(records)
    full_corpus = sum(r["chrf_full_mean"] for r in records) / len(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "setup_id": args.setup_id,
        "lang": args.lang,
        "ckpt": args.ckpt,
        "eval_n": len(records),
        "n_samples": args.n_samples,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
        "emotion_prefix": args.emotion_prefix,
        "chrf_continuation": cont_corpus,
        "chrf_full_poem": full_corpus,
    }
    with out_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{args.setup_id}] CHRF continuation = {cont_corpus:.2f}   full-poem = {full_corpus:.2f}")
    print(f"  records -> {out_path}")
    return summary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--setup_id", type=str, required=True)
    parser.add_argument("--emotion_prefix", choices=["none", "random", "fixed", "pred"], default="none")
    parser.add_argument("--emotion_label", type=str, default=None, help="label for --emotion_prefix fixed")
    parser.add_argument("--emotion_ckpt", type=str, default=None, help="classifier ckpt for --emotion_prefix pred")
    parser.add_argument("--en_input_path", type=str, default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--en_gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
    parser.add_argument("--ko_eval_path", type=str, default="data/processed/poem_ko_test.jsonl")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="conditioned eval JSONL; prompts with model_input when present")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Korean only (HF generate)")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="Korean only (HF generate)")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = f"results/gen_{args.setup_id}.jsonl"
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
