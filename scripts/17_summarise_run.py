"""Stage 17 (Phase 2): summarise protocol runs and compare each with a reference run on the same splits.

    python scripts/17_summarise_run.py \
        --base base=~/Desktop/MLPROJECT/runs/p2_base:main \
        --arm A0=~/Desktop/MLPROJECT/runs/p2_augmentation:aug_none \
        --arm A1=~/Desktop/MLPROJECT/runs/p2_augmentation:aug_geo \
        --out ~/Desktop/MLPROJECT/runs/p2_augmentation/summary

A run is NAME=ROOT:TAG, read from ROOT/results/protocol_TAG/ and ROOT/work/protocol_TAG/predictions.csv.gz (the layout
scripts/13_protocol.py writes). Nothing is re-trained: everything comes from the stored metrics and per-image predictions.

Integrity checks first (written to checks.csv; the script stops if one fails): same splits, same metric keys, same test
rows and labels per split as the reference, finite metrics, and, for augmented runs, a clean leakage check and
n_aug_train = copies x n_train on every augmented row.
Outputs (--out): checks.csv, metrics_summary.csv (mean, sd over splits), per_class_prf_per_split.csv.gz and
per_class_prf_summary.csv (precision / recall / F1 per class), confusion_per_split.csv.gz, confusion_pooled.csv,
confusion_<run>_<model>_all_clean.png, comparison_clean.csv, comparison_conditions.csv, comparison_kinds.csv
(paired over splits: Wilcoxon signed-rank, Holm within each family), summary.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import plot_confusion  # noqa: E402
from src.stats import holm, paired  # noqa: E402

KEY = ["split", "feature_set", "model", "condition"]
METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "mcc", "ece", "brier"]
# perturbation kinds related to each augmentation setting (reported separately from unseen kinds)
RELATED = {"rotate_deg": {"rotate"}, "brightness": {"brightness"}, "contrast": {"contrast"}}


def parse(spec):
    name, rest = spec.split("=", 1)
    root, tag = rest.rsplit(":", 1)
    root = Path(root).expanduser()
    res = root / "results" / f"protocol_{tag}"
    run = {"name": name, "res": res, "metrics": pd.read_csv(res / "metrics_per_split.csv"),
           "splits": pd.read_csv(res / "splits.csv"), "params": pd.read_csv(res / "chosen_params.csv"),
           "pred": pd.read_csv(root / "work" / f"protocol_{tag}" / "predictions.csv.gz")}
    s = res / "augmentation_settings.json"
    run["aug"] = json.load(open(s))["augmentation"] if s.exists() else None
    return run


def checks(base, runs):
    rows = []
    def add(run, check, ok, detail=""):
        rows.append({"run": run, "check": check, "ok": bool(ok), "detail": detail})
    bk = set(map(tuple, base["metrics"][KEY].values))
    bt = base["pred"][base["pred"].condition == "clean"].groupby(["split", "feature_set", "model"])
    for r in [base] + runs:
        m, n = r["metrics"], r["name"]
        add(n, "finite metrics", np.isfinite(m[METRICS].values).all())
        add(n, "no duplicate metric keys", not m.duplicated(KEY).any())
        if r is base:
            continue
        add(n, "splits.csv equals reference", r["splits"].equals(base["splits"]))
        add(n, "metric keys equal reference", set(map(tuple, m[KEY].values)) == bk)
        same = True
        for k, g in r["pred"][r["pred"].condition == "clean"].groupby(["split", "feature_set", "model"]):
            b = bt.get_group(k)
            same &= np.array_equal(np.sort(g.row.values), np.sort(b.row.values)) and \
                np.array_equal(g.sort_values("row").y_true.values, b.sort_values("row").y_true.values)
        add(n, "test rows and labels per split equal reference", same)
        if r["aug"] is not None:
            leak = pd.read_csv(r["res"] / "augmentation_leakage_check.csv")
            cols = ["sources_not_in_train", "sources_in_val_or_test", "source_groups_in_val_or_test"]
            add(n, "leakage check all zero", not leak[cols].values.any(), f"{len(leak)} splits")
            p = r["params"].merge(r["splits"].rename(columns={"n_train": "n_tr"})[["split", "n_tr"]], on="split")
            k = int(r["aug"]["copies"])
            exp = np.where(p.feature_set == "meta", 0, k * p.n_tr)
            add(n, "n_aug_train = copies x n_train (0 for meta)", np.array_equal(p.n_aug_train.values, exp), f"copies={k}")
    out = pd.DataFrame(rows)
    return out


def per_class(runs, classes):
    """Per-split confusion counts and per-class precision / recall / F1 from the stored predictions."""
    k = len(classes)
    conf, prf = [], []
    for r in runs:
        p = r["pred"]
        g = p.groupby(["split", "feature_set", "model", "condition", "y_true", "y_pred"]).size().rename("count").reset_index()
        g.insert(0, "run", r["name"])
        conf.append(g)
        for key, sub in g.groupby(["split", "feature_set", "model", "condition"]):
            cm = np.zeros((k, k))
            cm[sub.y_true.values, sub.y_pred.values] = sub["count"].values
            tp, col, row = np.diag(cm), cm.sum(0), cm.sum(1)
            prec = np.divide(tp, col, out=np.zeros(k), where=col > 0)
            rec = np.divide(tp, row, out=np.zeros(k), where=row > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
            for i, c in enumerate(classes):
                prf.append((r["name"], *key, c, prec[i], rec[i], f1[i], int(row[i])))
    conf = pd.concat(conf, ignore_index=True)
    conf["true"] = np.array(classes)[conf.y_true]
    conf["predicted"] = np.array(classes)[conf.y_pred]
    prf = pd.DataFrame(prf, columns=["run", *KEY, "class", "precision", "recall", "f1", "support"])
    return conf, prf


def compare(base, run, related):
    """Paired comparisons of macro-F1 over the matched splits (run minus reference)."""
    m, b = run["metrics"], base["metrics"]
    j = m.merge(b, on=KEY, suffixes=("", "_ref"))
    clean, conds, kinds = [], [], []
    for (fs, mo), g in j[j.condition == "clean"].groupby(["feature_set", "model"]):
        clean.append({"feature_set": fs, "model": mo, **paired(g.macro_f1.values, g.macro_f1_ref.values)})
    pert = j[j.condition != "clean"]
    for (fs, mo, c), g in pert.groupby(["feature_set", "model", "condition"]):
        conds.append({"feature_set": fs, "model": mo, "condition": c, "kind": g.kind.iloc[0], "level": g.level.iloc[0],
                      **paired(g.macro_f1.values, g.macro_f1_ref.values)})
    cl = j[j.condition == "clean"].set_index(["split", "feature_set", "model"])
    pk = pert.groupby(["split", "feature_set", "model", "kind"])[["macro_f1", "macro_f1_ref"]].mean().reset_index()
    pk = pk.join(cl[["macro_f1", "macro_f1_ref"]].rename(columns=lambda c: c + "_clean"), on=["split", "feature_set", "model"])
    pk["ratio"], pk["ratio_ref"] = pk.macro_f1 / pk.macro_f1_clean, pk.macro_f1_ref / pk.macro_f1_ref_clean
    for (fs, mo, kd), g in pk.groupby(["feature_set", "model", "kind"]):
        a = paired(g.macro_f1.values, g.macro_f1_ref.values)
        rr = paired(g.ratio.values, g.ratio_ref.values)
        kinds.append({"feature_set": fs, "model": mo, "kind": kd, "related_to_augmentation": kd in related, **a,
                      "ratio_mean_a": rr["mean_a"], "ratio_mean_b": rr["mean_b"], "ratio_mean_diff": rr["mean_diff"],
                      "ratio_p_wilcoxon": rr["p_wilcoxon"]})
    clean, conds, kinds = map(pd.DataFrame, (clean, conds, kinds))
    clean["p_holm"] = holm(clean.p_wilcoxon.values)                                   # family: all clean set x model
    for d in (conds, kinds):
        d["p_holm"] = d.groupby(["feature_set", "model"]).p_wilcoxon.transform(lambda s: holm(s.values))
    kinds["ratio_p_holm"] = kinds.groupby(["feature_set", "model"]).ratio_p_wilcoxon.transform(lambda s: holm(s.values))
    names = {"mean_a": "mean_run", "mean_b": "mean_ref", "a_wins": "run_wins", "b_wins": "ref_wins"}
    return tuple(d.rename(columns=names).assign(run=run["name"])[["run"] + [names.get(c, c) for c in d.columns]]
                 for d in (clean, conds, kinds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="reference run NAME=ROOT:TAG")
    ap.add_argument("--arm", action="append", required=True, help="run to compare, NAME=ROOT:TAG (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    base, runs = parse(args.base), [parse(a) for a in args.arm]
    classes = [c[2:] for c in base["pred"].columns if c.startswith("p_")]
    ck = checks(base, runs)
    ck.to_csv(out / "checks.csv", index=False)
    print(ck.to_string(index=False))
    if not ck.ok.all():
        raise SystemExit(f"integrity check failed, see {out / 'checks.csv'}")

    allr = [base] + runs
    ms = pd.concat([r["metrics"].assign(run=r["name"]) for r in allr])
    s = ms.groupby(["run", "feature_set", "model", "condition"])[METRICS].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    s.reset_index().assign(n_splits=ms.groupby(["run", "feature_set", "model", "condition"]).size().values).to_csv(
        out / "metrics_summary.csv", index=False)

    conf, prf = per_class(allr, classes)
    conf.to_csv(out / "confusion_per_split.csv.gz", index=False)
    pooled = conf.groupby(["run", "feature_set", "model", "condition", "true", "predicted"])["count"].sum().reset_index()
    pooled.to_csv(out / "confusion_pooled.csv", index=False)
    prf.to_csv(out / "per_class_prf_per_split.csv.gz", index=False)
    (prf.groupby(["run", "feature_set", "model", "condition", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
        .pipe(lambda d: d.set_axis(["_".join(c) for c in d.columns], axis=1)).reset_index()
        .to_csv(out / "per_class_prf_summary.csv", index=False))
    for r in allr:
        for mo in sorted(r["metrics"].model.unique()):
            sub = conf[(conf.run == r["name"]) & (conf.feature_set == "all") & (conf.model == mo) & (conf.condition == "clean")]
            cm = np.zeros((len(classes), len(classes)))
            np.add.at(cm, (sub.y_true.values, sub.y_pred.values), sub["count"].values)
            plot_confusion(cm, classes, out / f"confusion_{r['name']}_{mo}_all_clean.png",
                           f"{r['name']} {mo.upper()} all features, clean, pooled over {r['splits'].shape[0]} splits")

    comps = [compare(base, r, set().union(*[v for k, v in RELATED.items() if (r["aug"] or {}).get(k)]))
             for r in runs]
    cc, co, ki = (pd.concat([c[i] for c in comps], ignore_index=True) for i in range(3))
    cc.to_csv(out / "comparison_clean.csv", index=False)
    co.to_csv(out / "comparison_conditions.csv", index=False)
    ki.to_csv(out / "comparison_kinds.csv", index=False)

    a = args.alpha
    lines = [f"# Comparison against `{base['name']}` (paired over {base['splits'].shape[0]} splits, Wilcoxon, Holm)", "",
             "## Clean macro-F1 (mean ± sd; Δ = run − reference)", "",
             "| run | features | model | run | reference | Δ | run wins | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    sd = ms[ms.condition == "clean"].groupby(["run", "feature_set", "model"]).macro_f1.std()
    for _, r in cc[cc.feature_set.isin(["all", "color+texture"])].iterrows():
        lines.append(f"| {r.run} | {r.feature_set} | {r.model} | {r.mean_run:.3f} ± {sd[(r.run, r.feature_set, r.model)]:.3f} | "
                     f"{r.mean_ref:.3f} ± {sd[(base['name'], r.feature_set, r.model)]:.3f} | {r.mean_diff:+.3f} | "
                     f"{r.run_wins}/{r.n} | {r.p_holm:.3f} |")
    lines += ["", f"## Perturbation kinds, macro-F1 averaged over levels (Holm within run × features × model; α = {a})", ""]
    for (rn, fs, mo), g in ki[ki.feature_set.isin(["all", "color+texture"])].groupby(["run", "feature_set", "model"]):
        up = g[(g.p_holm < a) & (g.mean_diff > 0)]
        dn = g[(g.p_holm < a) & (g.mean_diff < 0)]
        fmt = lambda d: ", ".join(f"{k}{'*' if rel else ''} {v:+.3f}" for k, v, rel in zip(d.kind, d.mean_diff, d.related_to_augmentation)) or "none"
        lines.append(f"- **{rn} {fs} {mo}** — better: {fmt(up)}; worse: {fmt(dn)}")
    lines += ["", "`*` = perturbation family related to the training augmentation (not an unseen condition)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
