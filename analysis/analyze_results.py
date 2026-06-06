#!/usr/bin/env python3
"""Phase 6: aggregate all emotion-poem experiment results into results/analysis.json.

Collects: CHRF matrix, paired comparisons, emotion-classifier metrics, label
distributions, generated/reference length, and tokenizer token-length stats.
"""

import json
import statistics as st
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(1, str(ROOT / "training"))

EN_SETUPS = ["E0", "E1", "E2", "E3"]
KO_SETUPS = ["K0", "K1", "K2", "K3"]
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_records(setup):
    return {r["id"]: r for r in read_jsonl(f"results/gen_{setup}.jsonl")}


def load_summary(setup):
    return json.load(open(f"results/gen_{setup}.summary.json", encoding="utf-8"))


def paired(a_recs, b_recs):
    d = [b_recs[k]["chrf_continuation_mean"] - a_recs[k]["chrf_continuation_mean"] for k in a_recs]
    better = sum(1 for x in d if x > 0)
    worse = sum(1 for x in d if x < 0)
    return {"n": len(d), "mean_delta": st.mean(d), "median_delta": st.median(d),
            "b_better": better, "b_worse": worse}


def gen_length_stats(setup):
    recs = read_jsonl(f"results/gen_{setup}.jsonl")
    hyp_chars = [len(r["hypothesis"]) for r in recs]
    hyp_lines = [r["hypothesis"].count("\n") + 1 for r in recs]
    ref_chars = [len(r["target"]) for r in recs]
    return {"hyp_chars_mean": st.mean(hyp_chars), "hyp_lines_mean": st.mean(hyp_lines),
            "ref_chars_mean": st.mean(ref_chars)}


def main():
    analysis = {}

    # 1. CHRF matrix
    matrix = {}
    for s in EN_SETUPS + KO_SETUPS:
        summ = load_summary(s)
        matrix[s] = {"continuation": summ["chrf_continuation"], "full_poem": summ["chrf_full_poem"],
                     "n": summ["eval_n"]}
    analysis["chrf_matrix"] = matrix

    # 2. paired comparisons (within language)
    recs = {s: load_records(s) for s in EN_SETUPS + KO_SETUPS}
    analysis["paired"] = {
        "E1_vs_E0": paired(recs["E0"], recs["E1"]),
        "E2_vs_E0": paired(recs["E0"], recs["E2"]),
        "E1_vs_E2": paired(recs["E2"], recs["E1"]),  # predicted vs random (Pipeline A test)
        "E3_vs_E0": paired(recs["E0"], recs["E3"]),
        "K1_vs_K0": paired(recs["K0"], recs["K1"]),
        "K1_vs_K2": paired(recs["K2"], recs["K1"]),
        "K3_vs_K0": paired(recs["K0"], recs["K3"]),
    }

    # 3. emotion classifier metrics
    clf = {}
    for lang in ["en", "ko"]:
        d = json.load(open(f"checkpoints/emotion/emotion_{lang}_classifier.pt.eval.json", encoding="utf-8"))
        clf[lang] = {"dev_acc": d["dev_acc"], "dev_macro_f1": d["dev_macro_f1"],
                     "per_class_f1": d["per_class_f1"], "confusion_matrix": d["confusion_matrix"]}
    analysis["classifier"] = clf

    # 4. emotion label distribution (train)
    dist = {}
    for lang in ["en", "ko"]:
        rows = read_jsonl(f"data/processed/emotion_{lang}_train.jsonl")
        c = Counter(r["label"] for r in rows)
        dist[lang] = {"n": len(rows), "counts": {lab: c.get(lab, 0) for lab in LABELS}}
    analysis["emotion_label_distribution"] = dist

    # 5. generated / reference length per setup
    analysis["length"] = {s: gen_length_stats(s) for s in EN_SETUPS + KO_SETUPS}

    # 6. tokenizer token-length of poems (en gpt2 vs ko kogpt2, each on its own corpus)
    from train_emotion import build_tokenizer

    tok_stats = {}
    # English sonnets
    from datasets import SonnetsDataset
    en_tok, _ = build_tokenizer("en")
    en_poems = [t for _, t in SonnetsDataset("data/sonnets.txt")]
    en_lens = [len(en_tok(p)["input_ids"]) for p in en_poems]
    tok_stats["en_gpt2"] = {"n_poems": len(en_poems), "tokens_per_poem_mean": st.mean(en_lens),
                            "chars_per_poem_mean": st.mean(len(p) for p in en_poems)}
    # Korean poems
    ko_tok, _ = build_tokenizer("ko")
    ko_poems = [r["full_text"] for r in read_jsonl("data/processed/poem_ko_train.jsonl")]
    ko_lens = [len(ko_tok(p)["input_ids"]) for p in ko_poems]
    tok_stats["ko_kogpt2"] = {"n_poems": len(ko_poems), "tokens_per_poem_mean": st.mean(ko_lens),
                              "chars_per_poem_mean": st.mean(len(p) for p in ko_poems)}
    analysis["tokenizer_length"] = tok_stats

    Path("results").mkdir(exist_ok=True)
    with open("results/analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # readable print
    print("=== CHRF matrix (continuation) ===")
    for s in EN_SETUPS + KO_SETUPS:
        print(f"  {s}: {matrix[s]['continuation']:.2f}  (full {matrix[s]['full_poem']:.2f}, n={matrix[s]['n']})")
    print("\n=== key paired ===")
    for k in ["E1_vs_E0", "E1_vs_E2", "E3_vs_E0"]:
        p = analysis["paired"][k]
        print(f"  {k}: meanΔ={p['mean_delta']:+.2f}  better {p['b_better']}/{p['n']}, worse {p['b_worse']}/{p['n']}")
    print("\n=== classifier ===")
    for lang in ["en", "ko"]:
        print(f"  {lang}: acc={clf[lang]['dev_acc']:.3f}  macroF1={clf[lang]['dev_macro_f1']:.3f}")
    print("\n=== tokenizer length ===")
    for k, v in tok_stats.items():
        print(f"  {k}: {v['tokens_per_poem_mean']:.1f} tok/poem, {v['chars_per_poem_mean']:.1f} chars/poem")
    print("\nsaved -> results/analysis.json")


if __name__ == "__main__":
    main()
