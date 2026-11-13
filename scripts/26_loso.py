"""Stage 26: leave-one-source-out (LOSO) and seen-source evaluation (docs/phase3_design_2026-11-12.md).

    python scripts/26_loso.py --run F:/NUST/ML/runs/p4_multisource [--tasks bin six] [--sets ...] [--models ...]

Tasks: bin = healthy vs diseased (all sources); six = six classes (rows with a six-class label).
Protocols:
  loso : hold out one source; validation = 15% of every training source (grouped, per class); tune on it, refit on
         train + validation, score the held-out source once.
  seen : 70/15/15 grouped split inside every source, pooled (seeds 42-46); scored per source on its test part.
Feature sets (three extractor arms; the classifiers are always classical ML):
  formula : hand (105), lesion (20), hand+lesion
  kmeans  : colour_words (24), textons (32), kmeans (both), hand+lesion+kmeans -- k-means codebooks (src/codebook.py)
            re-learned in EVERY fold from the training rows only (label-free), then applied to all rows
  deep    : concepts (24 CLIP concepts), clip (512), resnet (2,048 -> PCA 256 fitted on the training rows of each
            fold) -- frozen pretrained extractors (scripts/25b); used only if their feature files exist
Models: rf, svm (RBF), lr; class_weight balanced; grids below.
Outputs (<run>/results/loso/): metrics.csv (one row per protocol x fold x task x set x model x scored source),
per_class.csv, chosen_params.csv; predictions in <run>/work/loso_predictions.csv.gz
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import ParameterGrid  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src.codebook import K_COLOUR, K_TEXTON, fit_codebook, histograms  # noqa: E402
from src.data import group_split  # noqa: E402
from src.protocol import group_holdout  # noqa: E402

SEED = 42
GRIDS = {"rf": {"min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.2]},
         "svm": {"C": [1, 10, 100]},
         "lr": {"C": [0.01, 0.1, 1]}}
SETS = ["hand", "lesion", "hand+lesion", "colour_words", "textons", "kmeans", "hand+lesion+kmeans",
        "concepts", "clip", "resnet"]
KMEANS_SETS = {"colour_words", "textons", "kmeans", "hand+lesion+kmeans"}
CLASSES = ["early_blight", "late_blight", "septoria_leaf_spot", "leaf_mold", "yellow_leaf_curl_virus", "healthy"]


def load_sets(work: Path):
    X = {"hand": np.load(work / "features_hand.npy"), "lesion": np.load(work / "features_lesion.npy")}
    X["hand+lesion"] = np.hstack([X["hand"], X["lesion"]])
    for name, f in (("concepts", "features_concepts.npy"), ("clip", "features_clip.npy"),
                    ("resnet", "features_resnet.npy")):
        if (work / f).exists():
            X[name] = np.load(work / f)
    desc = {"colour": np.load(work / "desc_colour.npy", mmap_mode="r"),
            "texture": np.load(work / "desc_texture.npy", mmap_mode="r")}
    return X, desc


def kmeans_sets(X, desc, fit_rows):
    """Codebooks learned on fit_rows only; histograms for every image."""
    cw = histograms(fit_codebook(desc["colour"], fit_rows, K_COLOUR), desc["colour"])
    tx = histograms(fit_codebook(desc["texture"], fit_rows, K_TEXTON), desc["texture"])
    km = np.hstack([cw, tx])
    return {"colour_words": cw, "textons": tx, "kmeans": km, "hand+lesion+kmeans": np.hstack([X["hand+lesion"], km])}


def make(name, p, n_feat, pca=False):
    steps = [StandardScaler()] + ([PCA(256, random_state=SEED)] if pca and n_feat > 256 else [])
    if name == "rf":
        return make_pipeline(*steps, RandomForestClassifier(300, random_state=SEED, n_jobs=-1,
                                                            class_weight="balanced", **p))
    if name == "svm":
        return make_pipeline(*steps, SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED, **p))
    return make_pipeline(*steps, LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED, **p))


def score_fn(y, pred):
    return f1_score(y, pred, average="macro", labels=sorted(set(y.tolist())))


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X)
    return d if d.ndim > 1 else np.c_[-d, d]


def metrics(task, y, pred, S, classes_fit):
    present = sorted(set(y.tolist()))
    m = {"n": len(y), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro", labels=present), "n_classes_present": len(present)}
    if task == "bin":
        m["healthy_recall"] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else np.nan
        m["disease_recall"] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else np.nan
        col = list(classes_fit).index(1)
        m["auc"] = roc_auc_score(y, S[:, col]) if len(present) == 2 else np.nan
    return m


def val_mask(df, rows, ylab, seed):
    """15% of each training source as validation, grouped and per class."""
    va = np.zeros(len(rows), bool)
    sub = df.iloc[rows]
    for s in sub["source"].unique():
        k = np.where(sub["source"].values == s)[0]
        va[k] = group_holdout(ylab[rows][k], sub["group"].values[k], 0.15, seed)
    return va


def run_fold(proto, fold, task, tr, te_rows_by_source, va, df, y, X, desc, sets, models, out, preds):
    t0 = time.time()
    trn, val = tr[~va], tr[va]
    sets = [s for s in sets if s in X or s in KMEANS_SETS]
    if KMEANS_SETS & set(sets):
        X = {**X, **kmeans_sets(X, desc, tr)}
    for sname in sets:
        Xs = X[sname]
        for mname in models:
            best, best_s = None, -1
            for p in ParameterGrid(GRIDS[mname]):
                mdl = make(mname, p, Xs.shape[1], pca=sname == "resnet").fit(Xs[trn], y[trn])
                s = score_fn(y[val], mdl.predict(Xs[val]))
                if s > best_s:
                    best, best_s = p, s
            mdl = make(mname, best, Xs.shape[1], pca=sname == "resnet").fit(Xs[tr], y[tr])
            out["params"].append({"protocol": proto, "fold": fold, "task": task, "set": sname, "model": mname,
                                  "params": json.dumps(best), "val_macro_f1": best_s})
            classes_fit = mdl.classes_ if hasattr(mdl, "classes_") else mdl[-1].classes_
            for src, rows in te_rows_by_source.items():
                pred = mdl.predict(Xs[rows])
                S = scores(mdl, Xs[rows])
                m = metrics(task, y[rows], pred, S, classes_fit)
                out["metrics"].append({"protocol": proto, "fold": fold, "task": task, "set": sname, "model": mname,
                                       "source": src, **m})
                if task == "six":
                    labs = list(range(6))
                    p_, r_, f_, n_ = precision_recall_fscore_support(y[rows], pred, labels=labs, zero_division=0)
                    for i in labs:
                        if n_[i]:
                            out["per_class"].append({"protocol": proto, "fold": fold, "set": sname, "model": mname,
                                                     "source": src, "class": CLASSES[i], "support": int(n_[i]),
                                                     "recall": r_[i], "f1": f_[i]})
                preds.append(pd.DataFrame({"protocol": proto, "fold": fold, "task": task, "set": sname,
                                           "model": mname, "row": rows, "y": y[rows], "pred": pred}))
    print(f"  {proto} {fold} {task}: {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", nargs="*", default=["bin", "six"])
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--models", nargs="*", default=["rf", "svm", "lr"])
    ap.add_argument("--protocols", nargs="*", default=["loso", "seen"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for the output files (runs of different arms are merged later)")
    args = ap.parse_args()
    run = Path(args.run)
    res = run / "results" / "loso"
    res.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(run / "sources" / "manifest_sources.csv")
    X, desc = load_sets(run / "work")
    missing = [s for s in args.sets if s not in X and s not in KMEANS_SETS]
    if missing:
        print(f"feature sets not available yet (skipped): {missing}")
    out = {"metrics": [], "per_class": [], "params": []}
    preds = []
    for task in args.tasks:
        rows_all = np.arange(len(df)) if task == "bin" else np.where(df["class_id6"].values >= 0)[0]
        y = df["y_bin"].values if task == "bin" else df["class_id6"].values
        srcs = sorted(df.iloc[rows_all]["source"].unique())
        print(f"task {task}: {len(rows_all)} images, sources {srcs}")
        if "loso" in args.protocols:
            for s in srcs:
                te = rows_all[df.iloc[rows_all]["source"].values == s]
                tr = rows_all[df.iloc[rows_all]["source"].values != s]
                run_fold("loso", s, task, tr, {s: te}, val_mask(df, tr, y, SEED), df, y, X, desc, args.sets,
                         args.models, out, preds)
        if "seen" in args.protocols:
            lab = df["binary"].values if task == "bin" else df["label6"].values
            for k in range(args.seeds):
                part = np.empty(len(df), object)
                for s in srcs:
                    r = rows_all[df.iloc[rows_all]["source"].values == s]
                    sub = pd.DataFrame({"label": lab[r], "group": df["group"].values[r]})
                    part[r] = group_split(sub, {"train": 0.7, "val": 0.15, "test": 0.15}, SEED + k).values
                trv = rows_all[np.isin(part[rows_all], ["train", "val"])]
                va = part[trv] == "val"
                te = {s: rows_all[(part[rows_all] == "test") & (df.iloc[rows_all]["source"].values == s)]
                      for s in srcs}
                run_fold("seen", f"seed{SEED + k}", task, trv, {s: r for s, r in te.items() if len(r)}, va, df, y,
                         X, desc, args.sets, args.models, out, preds)
            pd.DataFrame(out["metrics"]).to_csv(res / f"metrics{args.tag}.csv", index=False)     # checkpoint
    pd.DataFrame(out["metrics"]).to_csv(res / f"metrics{args.tag}.csv", index=False)
    pd.DataFrame(out["per_class"]).to_csv(res / f"per_class{args.tag}.csv", index=False)
    pd.DataFrame(out["params"]).to_csv(res / f"chosen_params{args.tag}.csv", index=False)
    pd.concat(preds).to_csv(run / "work" / f"loso_predictions{args.tag}.csv.gz", index=False)
    print("done")


if __name__ == "__main__":
    main()
