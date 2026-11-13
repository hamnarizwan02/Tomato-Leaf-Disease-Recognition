"""Stage 23: model-free, feature-level explanation and its stability across datasets (run after scripts/20).

    python scripts/23_feature_explainability.py

No classifier is trained. Each of the 105 hand-crafted features is examined directly:
  A. healthy vs diseased, in every dataset that has both (our raw set; PlantDoc independent subset; Nazirpur healthy vs
     septoria + leaf curl, where the leaf-curl mapping does not matter because only "diseased" is used; Brahmanbaria):
     signed effect = ROC AUC of the feature for "diseased" minus 0.5 (Mann-Whitney; +: higher in diseased leaves).
     A feature is "consistent" if the sign is the same in all datasets and |effect| >= 0.1 in every dataset.
  B. six classes, our raw set vs PlantDoc independent: Kruskal-Wallis effect size eta^2 = (H - k + 1) / (n - k) per
     feature, and the rank agreement of the two eta^2 profiles.
  C. per-feature distribution shift between our raw set and each external set: two-sample KS statistic.
Outputs: results/feature_explain/ (healthy_vs_diseased.csv, consistency_summary.csv, sixclass_eta2.csv,
feature_shift_ks.csv, summary.json)
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal, ks_2samp, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.features import load_all_features  # noqa: E402
from src.utils import save_json  # noqa: E402

EFFECT = 0.1


def ext(cfg, name, independent_only=True):
    X = np.load(cfg["paths"]["work_dir"] / "cross_dataset" / name / "features_clean.npy")
    o = pd.read_csv(cfg["paths"]["results_dir"] / "cross_dataset" / name / "overlap_with_raw.csv")
    keep = ~o["overlap"].values if independent_only else np.ones(len(o), bool)
    return X[keep], o["label"].values[keep]


def eta2(X, y):
    k, n = len(np.unique(y)), len(y)
    out = []
    for j in range(X.shape[1]):
        groups = [X[y == c, j] for c in np.unique(y)]
        try:
            h = kruskal(*groups).statistic
        except ValueError:          # all values identical
            h = 0.0
        out.append((h - k + 1) / (n - k))
    return np.array(out)


def main():
    cfg = load_config()
    out = cfg["paths"]["results_dir"] / "feature_explain"
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(cfg)
    Xr, names = load_all_features(cfg, df)
    yr = df["label"].values
    group = np.array([n.split("_")[0] for n in names])

    sets = {"raw": (Xr, yr)}
    sets["plantdoc"] = ext(cfg, "plantdoc")
    Xn, yn = ext(cfg, "nazirpur")
    Xl, yl = ext(cfg, "nazirpur_leafcurl")
    sets["nazirpur"] = (np.vstack([Xn, Xl]), np.r_[yn, yl])
    sets["brahmanbaria"] = ext(cfg, "brahmanbaria")

    # A. healthy vs diseased
    eff, counts = {}, {}
    for s, (X, y) in sets.items():
        dis = (y != "healthy").astype(int)
        counts[s] = {"healthy": int((dis == 0).sum()), "diseased": int(dis.sum())}
        eff[s] = np.array([roc_auc_score(dis, X[:, j]) - 0.5 if X[:, j].std() > 0 else 0.0
                           for j in range(X.shape[1])])
    E = pd.DataFrame(eff, index=names)
    sign_same = (np.sign(E).nunique(axis=1) == 1) & (E != 0).all(axis=1)
    strong_all = (E.abs() >= EFFECT).all(axis=1)
    E["group"] = group
    E["consistent"] = sign_same & strong_all
    E["same_sign_all"] = sign_same
    E["min_abs_effect"] = E[list(sets)].abs().min(axis=1)
    E.sort_values("min_abs_effect", ascending=False).to_csv(out / "healthy_vs_diseased.csv", index_label="feature")
    agree = [{"a": a, "b": b, "spearman_signed_effect": float(spearmanr(E[a], E[b])[0]),
              "sign_agreement": float((np.sign(E[a]) == np.sign(E[b])).mean())}
             for a, b in combinations(list(sets), 2)]
    pd.DataFrame(agree).to_csv(out / "healthy_vs_diseased_agreement.csv", index=False)
    cs = E.groupby("group").agg(n=("consistent", "size"), consistent=("consistent", "sum"),
                                same_sign_all=("same_sign_all", "sum")).reset_index()
    cs.to_csv(out / "consistency_summary.csv", index=False)

    # B. six classes, raw vs PlantDoc independent
    er, ep = eta2(*sets["raw"]), eta2(*sets["plantdoc"])
    B = pd.DataFrame({"group": group, "eta2_raw": er, "eta2_plantdoc": ep}, index=names)
    B.sort_values("eta2_raw", ascending=False).to_csv(out / "sixclass_eta2.csv", index_label="feature")
    top_r, top_p = set(B["eta2_raw"].nlargest(10).index), set(B["eta2_plantdoc"].nlargest(10).index)

    # C. per-feature distribution shift vs our raw set
    ks = pd.DataFrame({s: [ks_2samp(Xr[:, j], X[:, j]).statistic for j in range(Xr.shape[1])]
                       for s, (X, _) in sets.items() if s != "raw"}, index=names)
    ks["group"] = group
    ks.to_csv(out / "feature_shift_ks.csv", index_label="feature")

    summary = {"counts": counts, "effect_threshold": EFFECT,
               "n_consistent": int(E["consistent"].sum()),
               "consistent_features": E[E["consistent"]].sort_values("min_abs_effect", ascending=False)
               .index.tolist(),
               "n_same_sign_all": int(sign_same.sum()),
               "agreement_healthy_vs_diseased": agree,
               "sixclass_spearman_raw_vs_plantdoc": float(spearmanr(er, ep)[0]),
               "sixclass_top10_jaccard": len(top_r & top_p) / len(top_r | top_p),
               "sixclass_top10_raw": B["eta2_raw"].nlargest(10).index.tolist(),
               "sixclass_top10_plantdoc": B["eta2_plantdoc"].nlargest(10).index.tolist(),
               "sixclass_eta2_by_group": B.groupby("group")[["eta2_raw", "eta2_plantdoc"]].mean().round(4).to_dict(),
               "ks_mean_by_group": ks.groupby("group").mean().round(3).to_dict(),
               "ks_top10_shift": {s: ks[s].nlargest(10).index.tolist() for s in ks.columns if s != "group"}}
    save_json(summary, out / "summary.json")
    print(pd.Series({k: v for k, v in summary.items() if k not in ("ks_top10_shift",)}).to_string())
    print(cs.to_string(index=False))
    print(E[E["consistent"]].sort_values("min_abs_effect", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
