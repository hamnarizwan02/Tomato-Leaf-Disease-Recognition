"""Metrics and plots.  (Owner: Member C)"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import warnings  # noqa: E402

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             matthews_corrcoef, precision_recall_fscore_support)


def pretty(c: str) -> str:
    return c.replace("_", " ").title()


def compute_metrics(y_true, y_pred, classes) -> dict:
    labels = list(range(len(classes)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": float(f.mean()),
           "macro_precision": float(p.mean()), "macro_recall": float(r.mean())}
    for i, c in enumerate(classes):
        out[f"f1_{c}"] = float(f[i])
    return out


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Top-label ECE: |accuracy - confidence| per confidence bin, weighted by bin size."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    conf, correct = proba.max(1), proba.argmax(1) == y_true
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(correct[bins == b].mean() - conf[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def brier_score(y_true, proba) -> float:
    """Multi-class Brier score: mean over images of the squared distance to the one-hot label (0 best, 2 worst)."""
    proba = np.asarray(proba)
    return float(((proba - np.eye(proba.shape[1])[np.asarray(y_true)]) ** 2).sum(1).mean())


def full_metrics(y_true, y_pred, proba, classes) -> dict:
    """compute_metrics plus balanced accuracy, MCC, ECE and Brier score (proba: (n, n_classes))."""
    out = compute_metrics(y_true, y_pred, classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.update(balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                   mcc=float(matthews_corrcoef(y_true, y_pred)),
                   ece=expected_calibration_error(y_true, proba), brier=brier_score(y_true, proba))
    return out


def selective_curve(y_true, y_pred, confidence, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> list[dict]:
    """Accuracy on the most confident `coverage` share of the images (selective prediction)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    order = np.argsort(-np.asarray(confidence), kind="stable")
    rows = []
    for c in coverages:
        n = max(1, int(np.ceil(c * len(order))))
        rows.append({"coverage": c, "accuracy": float((y_true[order[:n]] == y_pred[order[:n]]).mean())})
    return rows


def robustness_table(m: pd.DataFrame, identity: dict, metric: str = "macro_f1",
                     keys=("split", "feature_set", "model")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per split: perturbed/clean ratio per kind and direction (mean, worst, area under the severity curve) and the
    overall worst case. m needs columns keys + condition, kind, level, metric ("clean" rows give the reference)."""
    keys = list(keys)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    clean = m[m["condition"] == "clean"].set_index(keys)[metric].rename("clean")
    p = m[m["condition"] != "clean"].join(clean, on=keys)
    p = p.assign(ratio=p[metric] / p["clean"].clip(lower=1e-9))
    rows = []
    for gk, g in p.groupby(keys + ["kind"]):
        base = dict(zip(keys + ["kind"], gk))
        rows.append({**base, "direction": "all", "n_levels": len(g), "mean_ratio": g["ratio"].mean(),
                     "worst_ratio": g["ratio"].min(), "auc": np.nan})
        if base["kind"] not in identity:
            continue
        d = g["level"].astype(float).values - identity[base["kind"]]
        for sign, name in ((-1, "down"), (1, "up")):
            sel = np.sign(d) == sign
            if not sel.any():
                continue
            sev, r = np.abs(d[sel]), g["ratio"].values[sel]
            o = np.argsort(sev)
            auc = float(trapz(np.r_[1.0, r[o]], np.r_[0.0, sev[o]] / sev.max()))
            rows.append({**base, "direction": name, "n_levels": int(sel.sum()), "mean_ratio": float(r.mean()),
                         "worst_ratio": float(r.min()), "auc": auc})
    per_kind = pd.DataFrame(rows)
    worst = (p.loc[p.groupby(keys)["ratio"].idxmin(), keys + ["condition", "ratio"]]
             .rename(columns={"condition": "worst_condition", "ratio": "worst_ratio"}).reset_index(drop=True))
    return per_kind, worst


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})


def bootstrap_ci(y_true, y_pred, n: int = 500, seed: int = 0, alpha: float = 0.05):
    """95% bootstrap CI of macro-F1 (resampling test images)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[ix], y_pred[ix], average="macro"))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def confusion(y_true, y_pred, k: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(k)))


def plot_confusion(cm, classes, path, title="", normalize=True):
    cm = np.asarray(cm, float)
    show = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), [pretty(c) for c in classes], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), [pretty(c) for c in classes], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{show[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if show[i, j] > 0.5 * show.max() else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
