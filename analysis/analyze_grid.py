#!/usr/bin/env python3
"""Aggregate the trajectory-guidance grid into comparison tables + paired stats.

Reads results/scored_{setup}.{jsonl,summary.json} and writes results/grid_analysis.json.
"""

import json
import statistics as st
from pathlib import Path

GROUPS = {
    "KO_first3": ["K_base", "K_orc", "K_avg", "K_rnd"],
    "KO_r30": ["K_r30_base", "K_r30_orc", "K_r30_avg", "K_r30_rnd"],
    "KO_r50": ["K_r50_base", "K_r50_orc", "K_r50_avg", "K_r50_rnd"],
    "KO_r70": ["K_r70_base", "K_r70_orc", "K_r70_avg", "K_r70_rnd"],
    "EN_sent": ["E_base", "E_orc", "E_avg"],
    "SOWOL": ["S_base", "S_orc"],
}
PAIRINGS = [
    ("K_orc", "K_base"), ("K_orc", "K_rnd"), ("K_orc", "K_avg"),
    ("K_r30_orc", "K_r30_base"), ("K_r30_orc", "K_r30_rnd"),
    ("K_r50_orc", "K_r50_base"), ("K_r50_orc", "K_r50_rnd"),
    ("K_r70_orc", "K_r70_base"), ("K_r70_orc", "K_r70_rnd"),
    ("E_orc", "E_base"), ("E_orc", "E_avg"),
    ("S_orc", "S_base"),
]
METRICS = ["chrf", "bertscore_f1", "traj_mse", "traj_corr", "volta_dist", "distinct2", "line_emotion_agreement"]
HIGHER_BETTER = {"chrf": True, "bertscore_f1": True, "traj_mse": False, "traj_corr": True,
                 "volta_dist": False, "distinct2": True, "line_emotion_agreement": True}


def load_summary(setup):
    p = Path(f"results/scored_{setup}.summary.json")
    return json.load(open(p, encoding="utf-8"))["metrics"] if p.exists() else None


def norm_id(rid):
    """Align EN baseline ids (en-dev-i) with conditioned-file ids (sonnet-{132+i})."""
    if isinstance(rid, str) and rid.startswith("en-dev-"):
        return f"sonnet-{132 + int(rid.split('-')[-1])}"
    return rid


def load_rows(setup):
    p = Path(f"results/scored_{setup}.jsonl")
    if not p.exists():
        return {}
    return {norm_id(r["id"]): r for r in (json.loads(l) for l in open(p, encoding="utf-8")) if r}


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def paired(a_setup, b_setup):
    """Per-poem deltas of a vs b (a - b) on shared ids."""
    a, b = load_rows(a_setup), load_rows(b_setup)
    shared = sorted(set(a) & set(b))
    out = {"a": a_setup, "b": b_setup, "n": len(shared), "metrics": {}}
    for m in METRICS:
        deltas = []
        for k in shared:
            va, vb = a[k].get(m), b[k].get(m)
            if va is not None and vb is not None:
                deltas.append(va - vb)
        if not deltas:
            continue
        sign = 1 if HIGHER_BETTER.get(m, True) else -1
        better = sum(1 for d in deltas if sign * d > 0)
        worse = sum(1 for d in deltas if sign * d < 0)
        out["metrics"][m] = {
            "mean_delta": st.mean(deltas),
            "a_better": better,
            "a_worse": worse,
            "n": len(deltas),
        }
    return out


def sowol_subset_means(setup):
    """Means over only Kim Sowol poems inside a full-KPoEM run."""
    rows = [r for r in load_rows(setup).values() if "김소월" in str(r.get("poet", ""))]
    if not rows:
        return None
    return {"n": len(rows), **{m: mean_or_none([r.get(m) for r in rows]) for m in METRICS}}


def main():
    analysis = {"groups": {}, "paired": [], "sowol_cross": {}}

    for group, setups in GROUPS.items():
        table = {}
        for s in setups:
            summ = load_summary(s)
            if summ:
                table[s] = {m: summ.get(m) for m in METRICS}
        analysis["groups"][group] = table

    for a, b in PAIRINGS:
        analysis["paired"].append(paired(a, b))

    # Sowol cross-comparison: poet-restricted means of the full-KPoEM models
    for s in ["K_base", "K_orc", "K_avg", "K_rnd"]:
        sub = sowol_subset_means(s)
        if sub:
            analysis["sowol_cross"][f"{s}_on_sowol_test"] = sub

    Path("results/grid_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # readable print
    fmt = lambda v: ("  --  " if v is None else f"{v:6.3f}")
    for group, table in analysis["groups"].items():
        print(f"\n==== {group} ====")
        print(f"{'setup':14s} " + " ".join(f"{m[:10]:>10s}" for m in METRICS))
        for s, row in table.items():
            print(f"{s:14s} " + " ".join(f"{fmt(row.get(m)):>10s}" for m in METRICS))
    print("\n==== key paired (a vs b) ====")
    for p in analysis["paired"]:
        tc = p["metrics"].get("traj_corr") or {}
        bs = p["metrics"].get("bertscore_f1") or {}
        print(f"{p['a']:14s} vs {p['b']:14s}  traj_corr Δ={tc.get('mean_delta', 0):+.3f} "
              f"({tc.get('a_better','-')}/{tc.get('n','-')})   bertscore Δ={bs.get('mean_delta', 0):+.4f} "
              f"({bs.get('a_better','-')}/{bs.get('n','-')})")
    print("\n==== Sowol cross ====")
    for k, v in analysis["sowol_cross"].items():
        print(k, json.dumps({m: round(v[m], 3) for m in METRICS if v.get(m) is not None}, ensure_ascii=False))
    print("\nsaved -> results/grid_analysis.json")


if __name__ == "__main__":
    main()
