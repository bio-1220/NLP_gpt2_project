#!/usr/bin/env python3
"""Full-tier scoring of generation outputs (re-scorable, decoupled from generation).

Reads a gen_*.jsonl produced by evaluate_generation.py and computes per record:
  - CHRF (continuation)
  - Distinct-1 / Distinct-2
  - BERTScore F1 (semantic similarity; lang-appropriate model)
  - Trajectory metrics via line measurers (same measurer on gen & ref):
      valence/sentiment MSE, correlation (continuation lines),
      volta distance (prefix+continuation curve),
      [KO] per-line emotion label agreement
If the record stores all samples in "hypotheses", metrics are averaged over
samples; otherwise the single "hypothesis" is scored.

Run:
  python score_generations.py --gen_file results/gen_K0.jsonl --lang ko --use_gpu
  python score_generations.py --gen_file results/gen_E0.jsonl --lang en --use_gpu
"""

import argparse
import json
import statistics as st
from pathlib import Path

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.line_measure import EnglishLineSentiment, KoreanLineEmotion, split_lines
from experiments.metrics import (
    EMOTION_VALENCE,
    chrf_sentence_scores,
    distinct_n,
    trajectory_correlation,
    trajectory_mse,
    volta_distance,
)


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def record_hypotheses(rec):
    hyps = rec.get("hypotheses")
    if hyps:
        return list(hyps)
    return [rec["hypothesis"]]


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--emotion_ckpt", type=str, default="checkpoints/emotion/emotion_ko_classifier.pt",
                        help="KO line-emotion measurer checkpoint")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--skip_trajectory", action="store_true")
    parser.add_argument("--bertscore_batch", type=int, default=64)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    records = read_jsonl(args.gen_file)
    out_base = Path(args.out) if args.out else Path(args.gen_file).with_suffix("").with_name(
        Path(args.gen_file).stem.replace("gen_", "scored_"))

    # ---- flat lists over (record, sample) for batch metrics ----
    flat_hyps, flat_refs, flat_idx = [], [], []
    for i, rec in enumerate(records):
        for hyp in record_hypotheses(rec):
            flat_hyps.append(hyp)
            flat_refs.append(rec["target"])
            flat_idx.append(i)

    # CHRF + distinct (cheap, per sample)
    chrf_flat = chrf_sentence_scores(flat_hyps, flat_refs)
    d1_flat = [distinct_n(h, 1) for h in flat_hyps]
    d2_flat = [distinct_n(h, 2) for h in flat_hyps]

    # BERTScore (batched once)
    bs_flat = [None] * len(flat_hyps)
    if not args.skip_bertscore:
        from bert_score import score as bertscore

        _, _, F = bertscore(flat_hyps, flat_refs, lang=args.lang,
                            device=str(device), batch_size=args.bertscore_batch, verbose=False)
        bs_flat = [float(f) for f in F]

    # Trajectory metrics
    traj_flat = [dict() for _ in flat_hyps]
    if not args.skip_trajectory:
        measurer = EnglishLineSentiment(device) if args.lang == "en" else KoreanLineEmotion(args.emotion_ckpt, device)
        ref_cont_traj, ref_full_traj, ref_labels = {}, {}, {}
        for i, rec in enumerate(records):
            ref_cont_traj[i] = measurer.trajectory(rec["target"])
            ref_full_traj[i] = measurer.trajectory(rec["prefix"] + "\n" + rec["target"])
            if args.lang == "ko":
                ref_labels[i] = measurer.label_trajectory(rec["target"])
        for j, (hyp, i) in enumerate(zip(flat_hyps, flat_idx)):
            gen_cont = measurer.trajectory(hyp)
            gen_full = measurer.trajectory(records[i]["prefix"] + "\n" + hyp)
            entry = {
                "traj_mse": trajectory_mse(gen_cont, ref_cont_traj[i]),
                "traj_corr": trajectory_correlation(gen_cont, ref_cont_traj[i]),
                "volta_dist": volta_distance(gen_full, ref_full_traj[i]),
                "gen_lines": len(gen_cont),
                "ref_lines": len(ref_cont_traj[i]),
            }
            if args.lang == "ko":
                gen_labels = measurer.label_trajectory(hyp)
                ref_l = ref_labels[i]
                n = min(len(gen_labels), len(ref_l))
                entry["line_emotion_agreement"] = (
                    sum(g == r for g, r in zip(gen_labels[:n], ref_l[:n])) / n if n else None
                )
            traj_flat[j] = entry

    # ---- aggregate back per record (mean over samples) ----
    scored = []
    per_rec = {}
    for j, i in enumerate(flat_idx):
        per_rec.setdefault(i, []).append(j)
    for i, rec in enumerate(records):
        js = per_rec[i]
        row = {
            "id": rec["id"],
            "n_samples": len(js),
            "chrf": st.mean(chrf_flat[j] for j in js),
            "distinct1": st.mean(d1_flat[j] for j in js),
            "distinct2": st.mean(d2_flat[j] for j in js),
            "bertscore_f1": mean_or_none([bs_flat[j] for j in js]),
            "traj_mse": mean_or_none([traj_flat[j].get("traj_mse") for j in js]),
            "traj_corr": mean_or_none([traj_flat[j].get("traj_corr") for j in js]),
            "volta_dist": mean_or_none([traj_flat[j].get("volta_dist") for j in js]),
        }
        if args.lang == "ko":
            row["line_emotion_agreement"] = mean_or_none(
                [traj_flat[j].get("line_emotion_agreement") for j in js])
        for key in ["poem_id", "title", "poet", "setup_id", "condition_label"]:
            if key in rec:
                row[key] = rec[key]
        scored.append(row)

    metric_keys = ["chrf", "distinct1", "distinct2", "bertscore_f1", "traj_mse", "traj_corr", "volta_dist"]
    if args.lang == "ko":
        metric_keys.append("line_emotion_agreement")
    summary = {
        "gen_file": args.gen_file,
        "lang": args.lang,
        "n_records": len(scored),
        "metrics": {k: mean_or_none([r.get(k) for r in scored]) for k in metric_keys},
    }

    with open(str(out_base) + ".jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(str(out_base) + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["metrics"], ensure_ascii=False,
                     indent=2, default=lambda x: round(x, 4)))
    print(f"-> {out_base}.jsonl / .summary.json")


if __name__ == "__main__":
    main()
