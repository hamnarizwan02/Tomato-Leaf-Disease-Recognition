"""Stage 13 (Member B, tables with C): repeated nested evaluation protocol (section 15, stages 1, 2c and 11).

    python scripts/13_protocol.py                       # protocol from config.yaml (10 grouped splits), all sets
    python scripts/13_protocol.py --mode repeated_cv    # 5x3 grouped CV instead
    python scripts/13_protocol.py --min-side 500        # shortcut control (a) V2: only images >= 500 px per side
    python scripts/13_protocol.py --no-perturbed --sets all color+texture --models rf    # quick run

Per split: tune on validation, refit on train+val, score test clean and under every perturbation from scripts/12.
The metadata-only model ("meta") and the shortcut-pruned feature set ("pruned") are ordinary feature sets, so they
appear as reference rows in every table. Split 0 of repeated_split is the committed seed-42 split.

Outputs (results/protocol_<tag>/): metrics_per_split.csv, summary_clean.csv, summary_all_conditions.csv,
  per_class_f1.csv, paired_tests.csv, robustness_by_kind.csv, robustness_worst.csv, selective.csv, chosen_params.csv,
  pruned_features.csv, splits.csv
Predictions: work/protocol_<tag>/predictions.csv.gz (one row per split x set x model x condition x test image)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest  # noqa: E402
from src.evaluate import robustness_table  # noqa: E402
from src.features import FEATURE_SETS, columns_for_set, load_all_features, load_perturbed_all  # noqa: E402
from src.metadata import META_FEATURES, image_metadata, pruned_columns, shortcut_correlations  # noqa: E402
from src.perturb import IDENTITY, all_conditions, condition_name  # noqa: E402
from src.protocol import FeatureSet, make_splits, run_protocol, summarise  # noqa: E402
from src.stats import paired_table  # noqa: E402
from src.augment import load_augmented  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402


def comparisons(sets, models):
    feat = [s for s in sets if s != "meta"]
    out = []
    for s in feat:                                       # models against each other on the same features
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                out.append(("models", (s, a), (s, b)))
    for m in models:                                     # each feature set against all features
        for s in feat:
            if s != "all" and "all" in feat:
                out.append(("feature_sets", (s, m), ("all", m)))
    if "meta" in sets:                                   # everything against the metadata-only reference
        for m in models:
            for s in feat:
                out.append(("vs_metadata", (s, m), ("meta", "rf" if "rf" in models else models[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repeated_split", "repeated_cv"])
    ap.add_argument("--n", type=int, help="number of repeats (splits, or CV repeats)")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--no-perturbed", action="store_true", help="clean test only")
    ap.add_argument("--min-side", type=int, default=0, help="keep only images with both sides >= this (V2)")
    ap.add_argument("--tag", help="output folder suffix (default from the options)")
    ap.add_argument("--no-predictions", action="store_true", help="do not store per-image predictions")
    ap.add_argument("--augment", action="store_true",
                    help="train-only augmentation from config['augmentation'] (needs scripts/16 with the same config)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    p, sc = cfg["protocol"], cfg["shortcut"]
    mode = args.mode or p["mode"]
    sets = args.sets or p["feature_sets"]
    models = args.models or p["models"]
    if args.augment and not cfg.get("augmentation"):
        raise SystemExit("--augment needs an augmentation block, e.g. TOMATO_CONFIG=configs/aug_geo.yaml")
    tag = args.tag or ((f"aug_{cfg['augmentation']['name']}" if args.augment else
                        f"min_side{args.min_side}" if args.min_side else "main")
                       + ("_cv" if mode == "repeated_cv" else ""))
    res = cfg["paths"]["results_dir"] / f"protocol_{tag}"
    work = cfg["paths"]["work_dir"] / f"protocol_{tag}"
    res.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    df = load_manifest(cfg)
    y = df["class_id"].values
    X, names = load_all_features(cfg, df)
    md = image_metadata(cfg, df)
    T = md[sc["corr_targets"]].values

    rows = None
    if args.min_side:
        keep = np.minimum(md["width"], md["height"]).values >= args.min_side
        rows = np.where(keep)[0]
        counts = df.iloc[rows]["label"].value_counts().reindex(cfg["dataset"]["classes"], fill_value=0)
        print(f"min side >= {args.min_side}: {len(rows)}/{len(df)} images\n{counts.to_string()}")
        if (counts < 10).any():
            print("WARNING: some classes have fewer than 10 images: their scores are not interpretable")
    splits = make_splits(df, cfg, rows=rows, mode=mode, n=args.n)
    pd.DataFrame([{"split": s["id"], "seed": s["seed"], "fold": s.get("fold"), "n_train": len(s["train"]),
                   "n_val": len(s["val"]), "n_test": len(s["test"])} for s in splits]).to_csv(
        res / "splits.csv", index=False)

    fsets, pruned_log = [], []
    for s in sets:
        if s == "meta":
            fsets.append(FeatureSet("meta", md[META_FEATURES].values.astype(float), np.arange(len(META_FEATURES)),
                                    perturbable=False))
        elif s == "pruned":
            def cols_fn(r, _log=pruned_log):
                c = pruned_columns(X, T, r, sc["corr_threshold"])
                _log.append(c)
                return c
            fsets.append(FeatureSet("pruned", X, cols_fn))
        elif s in FEATURE_SETS:
            fsets.append(FeatureSet(s, X, columns_for_set(names, s)))
        else:
            raise ValueError(f"unknown feature set {s!r}")

    perturbed, cmeta = {}, {}
    if not args.no_perturbed and set(p["perturbed_sets"]) & set(sets):
        conds = all_conditions(cfg, extended=True)
        cmeta = {condition_name(k, v): (k, v) for k, v in conds}
        perturbed = load_perturbed_all(cfg, df, list(cmeta))
        print(f"{len(perturbed)}/{len(cmeta)} perturbation conditions available")

    augment = None
    if args.augment:
        augment = load_augmented(cfg, df)
        groups = df["group"].to_numpy(dtype=object)
        check = []
        for s in splits:
            sel = np.isin(augment["src"], s["train"])
            held = np.concatenate([s["val"], s["test"]])
            check.append({"split": s["id"], "n_train": len(s["train"]), "n_aug_selected": int(sel.sum()),
                          "n_sources": int(np.unique(augment["src"][sel]).size),
                          "sources_not_in_train": int((~np.isin(augment["src"][sel], s["train"])).sum()),
                          "sources_in_val_or_test": int(np.isin(augment["src"][sel], held).sum()),
                          "source_groups_in_val_or_test": int(np.isin(groups[augment["src"][sel]], groups[held]).sum())})
        check = pd.DataFrame(check)
        check.to_csv(res / "augmentation_leakage_check.csv", index=False)
        if check[["sources_not_in_train", "sources_in_val_or_test", "source_groups_in_val_or_test"]].values.any():
            raise SystemExit(f"augmentation leakage detected, see {res / 'augmentation_leakage_check.csv'}")
        save_json({"augmentation": cfg["augmentation"], "seed": cfg["seed"], "n_aug_total": len(augment["src"])},
                  res / "augmentation_settings.json")
        print(f"augmentation '{cfg['augmentation']['name']}': {len(augment['src'])} copies stored; per split "
              f"{check.n_aug_selected.min()}-{check.n_aug_selected.max()} added to training; leakage check passed")

    out = run_protocol(splits, y, fsets, models, cfg, perturbed, p["perturbed_sets"], cmeta,
                       keep_predictions=not args.no_predictions, augment=augment)
    m = out["metrics"]
    m.to_csv(res / "metrics_per_split.csv", index=False)
    out["params"].to_csv(res / "chosen_params.csv", index=False)
    if not args.no_predictions:
        out["predictions"].to_csv(work / "predictions.csv.gz", index=False)

    clean = m[m.condition == "clean"]
    summ = summarise(clean, by=("feature_set", "model"))
    summ.to_csv(res / "summary_clean.csv", index=False)
    summarise(m).to_csv(res / "summary_all_conditions.csv", index=False)
    f1c = [c for c in clean.columns if c.startswith("f1_")]
    pc = clean.groupby(["feature_set", "model"])[f1c].agg(["mean", "std"]).round(4)
    pc.set_axis(["_".join(c) for c in pc.columns], axis=1).reset_index().to_csv(res / "per_class_f1.csv", index=False)
    sel = out["selective"].groupby(["feature_set", "model", "coverage"])["accuracy"].agg(["mean", "std"])
    sel.reset_index().to_csv(res / "selective.csv", index=False)
    paired_table(clean, comparisons(sets, models)).to_csv(res / "paired_tests.csv", index=False)

    if pruned_log:
        kept = np.zeros(len(names))
        for c in pruned_log:
            kept[c] += 1
        rho = np.abs(shortcut_correlations(X, T)).max(1)
        pd.DataFrame({"feature": names, "max_abs_spearman_all_rows": rho, "kept_share": kept / len(pruned_log)}
                     ).sort_values("max_abs_spearman_all_rows", ascending=False).to_csv(
            res / "pruned_features.csv", index=False)
        print(f"pruned set keeps {np.mean([len(c) for c in pruned_log]):.0f} of {len(names)} features on average")

    if (m.condition != "clean").any():
        per_kind, worst = robustness_table(m, IDENTITY)
        g = per_kind.groupby(["feature_set", "model", "kind", "direction"])
        (g[["mean_ratio", "worst_ratio", "auc"]].agg(["mean", "std"]).round(4)
         .pipe(lambda d: d.set_axis(["_".join(c) for c in d.columns], axis=1)).reset_index()
         .to_csv(res / "robustness_by_kind.csv", index=False))
        worst.to_csv(res / "robustness_worst.csv", index=False)

    view = summ.assign(macro_f1=summ.macro_f1_mean.map("{:.3f}".format) + " ± " + summ.macro_f1_std.map("{:.3f}".format))
    print(f"\nclean test macro-F1, mean ± std over {len(splits)} splits ({tag}):")
    print(view.pivot(index="feature_set", columns="model", values="macro_f1").to_string())
    print(f"\nresults -> {res}")


if __name__ == "__main__":
    main()
