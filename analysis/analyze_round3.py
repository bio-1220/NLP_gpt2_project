#!/usr/bin/env python3
"""Round 3 aggregation: full-curve conditioning (m/v/r) vs base/avg/shuffled controls.

Primary metrics per user spec: BERTScore, chrF, Volta distance.
Reads results/scored_R3_*.{jsonl,summary.json} -> results/round3_analysis.json
"""

import json
import statistics as st
from pathlib import Path

SETUPS = ["R3_base"] + [f"R3_{p}_{c}" for c in ["m", "v", "r"] for p in ["orc", "avg", "shuf"]]
METRICS = ["chrf", "bertscore_f1", "volta_dist", "traj_corr", "distinct2"]
HIGHER = {"chrf": True, "bertscore_f1": True, "volta_dist": False, "traj_corr": True, "distinct2": True}
PAIRINGS = [(f"R3_orc_{c}", b) for c in ["m", "v", "r"]
            for b in ("R3_base", f"R3_shuf_{c}", f"R3_avg_{c}")]


def rows(s):
    p = Path(f"results/scored_{s}.jsonl")
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))} if p.exists() else {}


def summ(s):
    p = Path(f"results/scored_{s}.summary.json")
    return json.load(open(p, encoding="utf-8"))["metrics"] if p.exists() else {}


def mean_or_none(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def paired(a, b):
    ra, rb = rows(a), rows(b)
    shared = sorted(set(ra) & set(rb))
    out = {"a": a, "b": b, "n": len(shared), "metrics": {}}
    for m in METRICS:
        d = [(ra[k][m], rb[k][m]) for k in shared if ra[k].get(m) is not None and rb[k].get(m) is not None]
        if not d:
            continue
        deltas = [x - y for x, y in d]
        sign = 1 if HIGHER[m] else -1
        out["metrics"][m] = {"mean_delta": st.mean(deltas),
                             "a_better": sum(1 for x in deltas if sign * x > 0),
                             "a_worse": sum(1 for x in deltas if sign * x < 0),
                             "n": len(deltas)}
    return out


def main():
    table = {s: {m: summ(s).get(m) for m in METRICS} for s in SETUPS}
    pairs = [paired(a, b) for a, b in PAIRINGS]
    out = {"table": table, "paired": pairs}
    Path("results/round3_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = lambda v: "  --  " if v is None else f"{v:7.3f}"
    print(f"{'setup':12s} " + " ".join(f"{m[:11]:>11s}" for m in METRICS))
    for s in SETUPS:
        print(f"{s:12s} " + " ".join(f"{fmt(table[s].get(m)):>11s}" for m in METRICS))
    print("\n=== paired (a vs b; chrF / BERTScore / Volta) ===")
    for p in pairs:
        cells = []
        for m in ["chrf", "bertscore_f1", "volta_dist"]:
            d = p["metrics"].get(m, {})
            cells.append(f"{m[:5]} Δ{d.get('mean_delta', 0):+.3f}({d.get('a_better','-')}/{d.get('n','-')})")
        print(f"{p['a']:11s} vs {p['b']:11s}  " + "  ".join(cells))
    print("\nsaved -> results/round3_analysis.json")


if __name__ == "__main__":
    main()
