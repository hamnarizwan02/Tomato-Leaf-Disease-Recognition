"""Paired comparisons over matched splits, with Holm correction.  (Owner: Member C)

Every protocol split is shared by all models and feature sets, so two configurations are compared on the SAME test
images: the per-split differences are paired. With 10 splits the smallest possible two-sided Wilcoxon p is 0.002.
Splits overlap in their training data, so the tests are approximate (they ignore that dependence) and are read
together with the size of the mean difference.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    run = 0.0
    for r, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (len(ok) - r) * p[i]))
        out[i] = run
    return out


def paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pw = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            pw = np.nan
        pt = float(ttest_rel(a, b).pvalue) if np.std(d) > 0 else (1.0 if np.all(d == 0) else 0.0)
    return {"n": len(d), "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(), "std_diff": d.std(ddof=1),
            "a_wins": int((d > 0).sum()), "b_wins": int((d < 0).sum()), "p_wilcoxon": pw, "p_ttest": pt}


def paired_table(m: pd.DataFrame, comparisons, metric: str = "macro_f1") -> pd.DataFrame:
    """comparisons: iterable of (family, (set_a, model_a), (set_b, model_b)). m: per-split metrics on one condition.
    Holm correction is applied within each family."""
    rows = []
    for fam, (sa, ma), (sb, mb) in comparisons:
        A = m[(m.feature_set == sa) & (m.model == ma)].set_index("split")[metric]
        B = m[(m.feature_set == sb) & (m.model == mb)].set_index("split")[metric]
        common = A.index.intersection(B.index)
        if len(common) < 2:
            continue
        rows.append({"family": fam, "a": f"{ma}/{sa}", "b": f"{mb}/{sb}", "metric": metric,
                     **paired(A[common].values, B[common].values)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("family")["p_wilcoxon"].transform(lambda s: holm(s.values))
    return out
