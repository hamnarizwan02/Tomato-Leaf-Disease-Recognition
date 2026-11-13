"""Stage 22: how different are the external sets from our raw set, in feature space? (run after scripts/20)

    python scripts/22_domain_shift.py --ext plantdoc nazirpur nazirpur_leafcurl brahmanbaria

Per external set (all images, and the independent subset that does not overlap our raw images):
  - leaf-mask area (shape_leaf_area_frac) mean and whole-image fallback rate (area > 0.999), vs our raw set;
  - "dataset classifier": RF (300 trees, seed 0) separating our 813 images from the external images on the
    105 features, 5-fold stratified CV ROC AUC. 0.5 = indistinguishable, 1.0 = perfectly separable (strong shift).
Output: results/cross_dataset/domain_shift.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.features import load_all_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config()
    own, names = load_all_features(cfg, load_manifest(cfg))
    i = names.index("shape_leaf_area_frac")
    rows = []
    for n in args.ext:
        X = np.load(cfg["paths"]["work_dir"] / "cross_dataset" / n / "features_clean.npy")
        ov = pd.read_csv(cfg["paths"]["results_dir"] / "cross_dataset" / n / "overlap_with_raw.csv")["overlap"].values
        for subset, mask in (("all", np.ones(len(X), bool)), ("independent", ~ov)):
            Xe = X[mask]
            Z, y = np.vstack([own, Xe]), np.r_[np.zeros(len(own)), np.ones(len(Xe))]
            auc = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), Z, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
            rows.append({"dataset": n, "subset": subset, "n": int(mask.sum()),
                         "leaf_area_mean": float(Xe[:, i].mean()), "own_leaf_area_mean": float(own[:, i].mean()),
                         "fallback_rate": float((Xe[:, i] > 0.999).mean()),
                         "own_fallback_rate": float((own[:, i] > 0.999).mean()),
                         "dataset_auc_mean": float(auc.mean()), "dataset_auc_std": float(auc.std())})
    out = pd.DataFrame(rows)
    out.to_csv(cfg["paths"]["results_dir"] / "cross_dataset" / "domain_shift.csv", index=False)
    print(out.round(3).to_string(index=False))

    # reference for importance_agreement.csv: how well per-feature permutation importances on OUR test split agree
    # between different splits (same dataset), per model. Pairwise Spearman / top-10 Jaccard over the 10 splits.
    from itertools import combinations
    from src.explain import agreement
    f = pd.read_csv(cfg["paths"]["results_dir"] / "cross_dataset" / args.ext[0] / "importance_feature.csv")
    ref = []
    for m, g in f.groupby("model"):
        P = g.pivot(index="split", columns="feature", values="own_test").values
        ag = [agreement(P[a], P[b]) for a, b in combinations(range(len(P)), 2)]
        ref.append({"model": m, "n_pairs": len(ag), **{k: float(np.mean([x[k] for x in ag])) for k in ag[0]}})
    ref = pd.DataFrame(ref)
    ref.to_csv(cfg["paths"]["results_dir"] / "cross_dataset" / "importance_own_split_agreement.csv", index=False)
    print("\nown-test importance agreement between splits:\n" + ref.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
