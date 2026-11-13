"""Stage 28: summarise the Phase 3 LOSO / seen-source runs (scripts/26) across extractor arms.

    python scripts/28_summarise_phase3.py --run F:/NUST/ML/runs/p4_multisource

Merges results/loso/metrics*.csv. Primary metrics: bin -> balanced accuracy (also AUC, healthy recall); six -> macro-F1
over the classes present in the scored source (also balanced accuracy). Every source counts once (sources are not
weighted by size). Six-class rows of held-out sources with < 2 classes present are dropped.
Outputs (results/phase3_summary/): overview.csv (task x set x model: LOSO mean/sd over held-out sources, seen-source
mean over sources, gap), per_source.csv, paired_vs_hand.csv (Wilcoxon over held-out sources vs the hand set, Holm
within task x model), best_per_arm.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARM = {"hand": "formula", "lesion": "formula", "hand+lesion": "formula", "colour_words": "kmeans",
       "textons": "kmeans", "kmeans": "kmeans", "hand+lesion+kmeans": "formula+kmeans", "concepts": "deep",
       "clip": "deep", "resnet": "deep"}
PRIMARY = {"bin": "balanced_accuracy", "six": "macro_f1"}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    res = Path(args.run) / "results"
    out = res / "phase3_summary"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.concat([pd.read_csv(f) for f in sorted((res / "loso").glob("metrics*.csv"))], ignore_index=True)
    # A held-out source with one class (Brahmanbaria healthy only in six-class) gives no meaningful macro-F1; the
    # design lists it as binary-only. Its healthy images still take part in six-class training.
    m = m[~((m.task == "six") & (m.n_classes_present < 2))].copy()
    m["arm"] = m["set"].map(ARM)
    m["primary"] = np.where(m["task"] == "bin", m["balanced_accuracy"], m["macro_f1"])

    lo = m[m.protocol == "loso"]
    se = m[m.protocol == "seen"].groupby(["task", "set", "model", "source"], as_index=False)[
        ["primary", "balanced_accuracy", "macro_f1", "auc", "healthy_recall"]].mean()
    per = lo[["task", "set", "arm", "model", "source", "n", "primary", "balanced_accuracy", "macro_f1", "auc",
              "healthy_recall", "disease_recall"]].merge(
        se[["task", "set", "model", "source", "primary"]].rename(columns={"primary": "seen_primary"}),
        on=["task", "set", "model", "source"], how="left")
    per["gap"] = per["seen_primary"] - per["primary"]
    per.to_csv(out / "per_source.csv", index=False)

    ov = per.groupby(["task", "arm", "set", "model"]).agg(
        loso_mean=("primary", "mean"), loso_sd=("primary", "std"), loso_min=("primary", "min"),
        seen_mean=("seen_primary", "mean"), gap_mean=("gap", "mean"), auc_mean=("auc", "mean"),
        healthy_recall_mean=("healthy_recall", "mean"), n_sources=("source", "nunique")).reset_index()
    ov.to_csv(out / "overview.csv", index=False)

    rows = []
    for (task, model), g in per.groupby(["task", "model"]):
        base = g[g.set == "hand"].set_index("source")["primary"]
        tmp = []
        for s, h in g[g.set != "hand"].groupby("set"):
            v = h.set_index("source")["primary"].reindex(base.index)
            d = (v - base).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            tmp.append({"task": task, "model": model, "set": s, "arm": ARM[s], "n_sources": len(d),
                        "mean_diff_vs_hand": d.mean(), "wins": int((d > 0).sum()), "p_wilcoxon": p})
        t = pd.DataFrame(tmp)
        t["p_holm"] = holm(t["p_wilcoxon"].fillna(1.0))
        rows.append(t)
    pd.concat(rows).to_csv(out / "paired_vs_hand.csv", index=False)

    best = ov.sort_values("loso_mean", ascending=False).groupby(["task", "arm"]).head(1)
    best.sort_values(["task", "loso_mean"], ascending=[True, False]).to_csv(out / "best_per_arm.csv", index=False)
    pd.set_option("display.width", 200)
    for task in ov.task.unique():
        print(f"\n== {task} ({PRIMARY[task]}; LOSO = unseen source, mean over held-out sources)")
        print(ov[ov.task == task].sort_values("loso_mean", ascending=False)[
            ["arm", "set", "model", "loso_mean", "loso_sd", "loso_min", "seen_mean", "gap_mean", "auc_mean",
             "healthy_recall_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
