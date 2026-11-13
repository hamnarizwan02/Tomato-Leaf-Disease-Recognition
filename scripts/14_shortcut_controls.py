"""Stage 14 (Member A): source-shortcut controls that need source clusters (section 15, stage 2 b and c).

    python scripts/14_shortcut_controls.py
    python scripts/14_shortcut_controls.py --models rf --sets all color+texture meta

(b) Clusters images by size, aspect ratio, compression and file-name pattern (one cluster per source group), then
    leave-one-cluster-out: train on the other clusters, test on the held-out one. Also accuracy per cluster under the
    main protocol, if scripts/13 has been run.
(c) Spearman correlation of every feature with log pixel count and bytes per pixel (all rows, for the report table;
    the pruned feature set in scripts/13 recomputes it on training rows only).
The other controls are run elsewhere: (a) V1 and (d) as config variants (configs/res_matched.yaml,
configs/reencode.yaml, see run_pipeline.py variants), (a) V2 with scripts/13 --min-side 500.

Outputs (results/shortcut/): feature_correlations.csv, cluster_k_selection.csv, clusters.csv, cluster_summary.csv,
  loco_metrics.csv, accuracy_by_cluster.csv
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.features import FEATURE_SETS, columns_for_set, load_all_features  # noqa: E402
from src.metadata import META_FEATURES, image_metadata, shortcut_correlations, source_clusters  # noqa: E402
from src.protocol import FeatureSet, cluster_splits, run_protocol  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["rf", "svm", "lr"])
    ap.add_argument("--sets", nargs="*", default=["all", "color+texture", "meta"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    sc = cfg["shortcut"]
    res = cfg["paths"]["results_dir"] / "shortcut"
    res.mkdir(parents=True, exist_ok=True)
    classes = cfg["dataset"]["classes"]

    df = load_manifest(cfg)
    y = df["class_id"].values
    X, names = load_all_features(cfg, df)
    md = image_metadata(cfg, df)

    # (c) feature <-> metadata correlations
    rho = shortcut_correlations(X, md[sc["corr_targets"]].values)
    corr = pd.DataFrame(rho, columns=[f"spearman_{t}" for t in sc["corr_targets"]])
    corr.insert(0, "feature", names)
    corr["max_abs"] = np.abs(rho).max(1)
    corr["above_threshold"] = corr["max_abs"] > sc["corr_threshold"]
    corr.sort_values("max_abs", ascending=False).to_csv(res / "feature_correlations.csv", index=False)
    print(f"(c) {int(corr.above_threshold.sum())}/{len(names)} features have |Spearman| > {sc['corr_threshold']} "
          f"with {sc['corr_targets']}. Top 5:")
    print(corr.sort_values("max_abs", ascending=False).head(5)[["feature", "max_abs"]].round(3).to_string(index=False))

    # (b) source clusters
    clusters, ksel = source_clusters(md, df["group"].values, sc["cluster_k"], cfg["seed"])
    ksel.to_csv(res / "cluster_k_selection.csv", index=False)
    pd.DataFrame({"path": df["path"], "label": df["label"], "cluster": clusters}).to_csv(res / "clusters.csv", index=False)
    summ = pd.crosstab(clusters, df["label"]).reindex(columns=classes, fill_value=0)
    summ.index.name = "cluster"
    summ = summ.join(md.groupby(clusters)[["megapixels", "aspect", "bytes_per_pixel"]].median().add_prefix("median_"))
    summ.join(pd.crosstab(clusters, md["name_pattern"]).add_prefix("name_")).to_csv(res / "cluster_summary.csv")
    print(f"\n(b) {len(summ)} source clusters (k by silhouette):\n{summ.round(2).to_string()}")
    alone = [c for c in classes if (summ[c] > 0).sum() == 1]
    if alone:
        print(f"WARNING: classes found in a single cluster: {alone}. Holding that cluster out removes the class from "
              "training, so its leave-one-cluster-out score is 0 by construction (reported, not interpreted).")

    fsets = []
    for s in args.sets:
        if s == "meta":
            fsets.append(FeatureSet("meta", md[META_FEATURES].values.astype(float), np.arange(len(META_FEATURES)), False))
        else:
            assert s in FEATURE_SETS, s
            fsets.append(FeatureSet(s, X, columns_for_set(names, s)))
    out = run_protocol(cluster_splits(df, clusters, cfg), y, fsets, args.models, cfg)
    pr = out["predictions"]
    rows = []
    for (c, fs, mo), g in pr.groupby(["split", "feature_set", "model"]):
        present = np.unique(g.y_true)
        trained = np.unique(y[clusters != c])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"cluster": c, "feature_set": fs, "model": mo, "n_test": len(g),
                         "classes_in_cluster": len(present),
                         "classes_unseen_in_training": int(len(set(present) - set(trained))),
                         "accuracy": accuracy_score(g.y_true, g.y_pred),
                         "macro_f1_present_classes": f1_score(g.y_true, g.y_pred, labels=present, average="macro")})
    loco = pd.DataFrame(rows)
    loco.to_csv(res / "loco_metrics.csv", index=False)
    print("\nleave-one-cluster-out accuracy:")
    print(loco.pivot_table(index="cluster", columns=["model", "feature_set"], values="accuracy").round(3).to_string())

    # accuracy per cluster under the main protocol (clean, every split)
    pf = cfg["paths"]["work_dir"] / "protocol_main" / "predictions.csv.gz"
    if pf.exists():
        p = pd.read_csv(pf, usecols=["split", "feature_set", "model", "condition", "row", "y_true", "y_pred"])
        p = p[p.condition == "clean"].assign(cluster=lambda d: clusters[d.row.values])
        acc = (p.assign(correct=p.y_true == p.y_pred)
                .groupby(["feature_set", "model", "cluster", "split"])["correct"].mean()
                .groupby(["feature_set", "model", "cluster"]).agg(["mean", "std", "count"]).reset_index())
        acc.to_csv(res / "accuracy_by_cluster.csv", index=False)
    else:
        print(f"note: {pf} missing, run scripts/13_protocol.py for accuracy per cluster under the main protocol")
    save_json({"n_clusters": int(len(summ)), "silhouette": float(ksel.silhouette.max()),
               "n_features_above_threshold": int(corr.above_threshold.sum())}, res / "summary.json")


if __name__ == "__main__":
    main()
