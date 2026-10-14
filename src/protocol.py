"""Repeated, nested evaluation protocol.  (Owner: Member B)

Section 15, stage 1. A split is a dict of manifest row indices {"id", "seed", "train", "val", "test"}; every split keeps
whole source groups together. For each split, feature set and model: tune on train -> val (models.tune), refit on
train+val, then score the test rows clean and under every perturbation. Every prediction is kept, so all metrics,
paired tests and robustness summaries are computed afterwards without refitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import group_split
from .evaluate import full_metrics, selective_curve
from .models import make_model, tune
from .utils import Timer


# --------------------------------------------------------------------------- splits
def group_holdout(labels, groups, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of a per-class grouped holdout of about `frac` of the rows (used for the validation part)."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        sizes = pd.Series(groups[rows]).value_counts()
        g = sizes.index.to_numpy().copy()
        rng.shuffle(g)
        target, cum, take = frac * len(rows), 0, []
        for gi in g:
            if cum >= target:
                break
            take.append(gi)
            cum += sizes[gi]
        hold[rows[np.isin(groups[rows], take)]] = True
    return hold


def make_splits(df: pd.DataFrame, cfg, rows=None, mode: str | None = None, n: int | None = None) -> list[dict]:
    """Splits over the manifest rows `rows` (default all). Indices refer to the full manifest."""
    p = cfg["protocol"]
    mode = mode or p["mode"]
    rows = np.arange(len(df)) if rows is None else np.asarray(rows)
    sub = df.iloc[rows].reset_index(drop=True)
    out = []
    if mode == "repeated_split":
        for k in range(n or p["n_repeats"]):
            seed = cfg["seed"] + k
            s = group_split(sub, cfg["dataset"]["group_split"], seed).values
            out.append({"id": k, "seed": seed, **{part: rows[s == part] for part in ("train", "val", "test")}})
    elif mode == "repeated_cv":
        for r in range(n or p["cv_repeats"]):
            seed = cfg["seed"] + r
            cv = StratifiedGroupKFold(p["n_folds"], shuffle=True, random_state=seed)
            for f, (trv, te) in enumerate(cv.split(sub, sub["class_id"], groups=sub["group"])):
                va = group_holdout(sub["class_id"].values[trv], sub["group"].values[trv], p["val_frac"], seed + f)
                out.append({"id": len(out), "seed": seed, "fold": f,
                            "train": rows[trv[~va]], "val": rows[trv[va]], "test": rows[te]})
    else:
        raise ValueError(f"unknown protocol mode: {mode}")
    return out


def cluster_splits(df: pd.DataFrame, clusters, cfg) -> list[dict]:
    """Leave-one-cluster-out: each source cluster is the test set once; tuning uses a grouped holdout of the rest."""
    clusters = np.asarray(clusters)
    out = []
    for k, c in enumerate(sorted(np.unique(clusters))):
        te = np.where(clusters == c)[0]
        rest = np.where(clusters != c)[0]
        va = group_holdout(df["class_id"].values[rest], df["group"].values[rest], cfg["protocol"]["val_frac"],
                           cfg["seed"] + k)
        out.append({"id": k, "seed": cfg["seed"], "cluster": int(c),
                    "train": rest[~va], "val": rest[va], "test": te})
    return out


# --------------------------------------------------------------------------- feature sets
@dataclass
class FeatureSet:
    name: str
    X: np.ndarray                                   # (n_manifest_rows, n_columns)
    cols: np.ndarray | Callable[[np.ndarray], np.ndarray]   # fixed columns, or fn(train+val rows) -> columns
    perturbable: bool = True                        # False for metadata: pixel perturbations do not change it

    def columns(self, rows) -> np.ndarray:
        return self.cols(rows) if callable(self.cols) else self.cols


# --------------------------------------------------------------------------- run
def _proba_full(model, X, k: int) -> np.ndarray:
    P = np.zeros((len(X), k))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def run_protocol(splits, y, fsets: list[FeatureSet], models, cfg, perturbed: dict | None = None,
                 perturbed_sets=(), conditions_meta: dict | None = None, keep_predictions: bool = True):
    """Returns dict of DataFrames: metrics (one row per split x set x model x condition), params, selective,
    predictions (long format, optional). perturbed: {condition: X_all} for the main feature matrix."""
    classes = cfg["dataset"]["classes"]
    k = len(classes)
    perturbed = perturbed or {}
    conditions_meta = conditions_meta or {}
    metrics, params, selective, preds = [], [], [], []
    for s in splits:
        tr, va, te = s["train"], s["val"], s["test"]
        tv = np.concatenate([tr, va])
        info = {key: s[key] for key in ("id", "seed", "fold", "cluster") if key in s}
        info["split"] = info.pop("id")
        with Timer(f"split {info['split']}: train {len(tr)}, val {len(va)}, test {len(te)}"):
            for fs in fsets:
                cols = fs.columns(tv)
                for mname in models:
                    best, table = tune(mname, fs.X[tr][:, cols], y[tr], fs.X[va][:, cols], y[va], cfg)
                    model = make_model(mname, best, cfg, proba=True).fit(fs.X[tv][:, cols], y[tv])
                    params.append({**info, "feature_set": fs.name, "model": mname, "n_features": len(cols),
                                   "params": str(best), "val_macro_f1": float(table.val_macro_f1.max())})
                    conds = [("clean", fs.X)]
                    if fs.perturbable and fs.name in perturbed_sets:
                        conds += list(perturbed.items())
                    for cname, Xc in conds:
                        Xt = np.asarray(Xc[te])[:, cols]
                        pred = model.predict(Xt)
                        P = _proba_full(model, Xt, k)
                        kind, level = conditions_meta.get(cname, ("clean", "none"))
                        metrics.append({**info, "feature_set": fs.name, "model": mname, "condition": cname,
                                        "kind": kind, "level": str(level), "n_features": len(cols),
                                        "n_test": len(te), **full_metrics(y[te], pred, P, classes)})
                        if cname == "clean":
                            for r in selective_curve(y[te], pred, P.max(1)):
                                selective.append({**info, "feature_set": fs.name, "model": mname, **r})
                        if keep_predictions:
                            d = pd.DataFrame(P.astype(np.float32), columns=[f"p_{c}" for c in classes])
                            d.insert(0, "y_pred", pred)
                            d.insert(0, "y_true", y[te])
                            d.insert(0, "row", te)
                            d.insert(0, "condition", cname)
                            d.insert(0, "model", mname)
                            d.insert(0, "feature_set", fs.name)
                            d.insert(0, "split", info["split"])
                            preds.append(d)
    out = {"metrics": pd.DataFrame(metrics), "params": pd.DataFrame(params), "selective": pd.DataFrame(selective)}
    if keep_predictions:
        out["predictions"] = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return out


def summarise(m: pd.DataFrame, by=("feature_set", "model", "condition"),
              cols=("macro_f1", "accuracy", "balanced_accuracy", "mcc", "ece", "brier")) -> pd.DataFrame:
    """Mean and std over splits."""
    by, cols = list(by), [c for c in cols if c in m.columns]
    s = m.groupby(by)[cols].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    return s.reset_index().assign(n_splits=m.groupby(by).size().values)
