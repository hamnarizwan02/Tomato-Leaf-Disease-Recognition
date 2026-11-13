"""Stage 20: cross-dataset generalisation and explanation transfer (train on our raw set, test on external sets).

Run in the main environment (TOMATO_DATA_ROOT = Dataset (raw), work dir holding clean_all.npy from stage 03):

    python scripts/20_cross_dataset.py --ext configs/external/plantdoc.yaml [--ext ...] [--models rf svm lr]

For each of the 10 grouped splits of the main protocol (same seeds, same grids): tune on validation (fit on train),
refit on train+val, then score (a) our own test split and (b) every image of each external set with the SAME fitted
model. Nothing is tuned, refitted or thresholded on external data.

Overlap with our data (decided before scoring): an external image overlaps if its file bytes equal a raw image
(sha256) or its 16x16 grey thumbnail correlates >= 0.95 with a raw image under one of the 8 rotations/flips (the frozen
content rule of scripts/18). Raw images = every file in Dataset (raw), including the 17 dropped ones. External scores
are reported for the independent subset (headline) and the overlapping subset separately.

Explanations (same fitted models): grouped permutation importance (colour / texture / shape, drop in true-class
probability) and per-feature permutation importance on our test split and on the independent external subset, and
their agreement (Spearman, top-10/20 Jaccard). RF impurity importance is saved too.

Outputs: results/cross_dataset/<name>/ (overlap_with_raw.csv, overlap_summary.json, metrics_per_split.csv,
summary.csv, per_class_f1.csv, confusion_pooled.csv, importance_group.csv, importance_feature.csv,
importance_agreement.csv, indomain_check.csv); predictions in work/cross_dataset/<name>/predictions.csv.gz
"""
import argparse
import copy
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import _read_rgb, build_image_cache, cache_file, load_manifest  # noqa: E402
from src.evaluate import full_metrics  # noqa: E402
from src.explain import agreement, permutation_importance  # noqa: E402
from src.features import (columns_for_set, extract_matrix, feature_schema, group_columns,  # noqa: E402
                          load_all_features, worker_cfg)
from src.models import make_model, tune  # noqa: E402
from src.protocol import make_splits  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402

THR, THUMB = 0.95, 16
ANY_DISEASE = "any_disease"     # binary external sets (healthy vs diseased): scored as healthy / not healthy only
EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".webp"}


def dihedral(path) -> np.ndarray:
    img = _read_rgb(path)
    if img is None:
        raise RuntimeError(f"unreadable image: {path}")
    t = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    out = []
    for k in range(4):
        r = np.rot90(t.astype(np.float32), k)
        for v in (r, r[:, ::-1]):
            v = v - v.mean()
            out.append((v / (np.linalg.norm(v) or 1.0)).ravel())
    return np.stack(out)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_external(spec: dict, classes: list[str]) -> pd.DataFrame:
    root = Path(spec["root"])
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            folder = p.parent.name
            if folder in spec["class_map"]:
                rows.append({"path": p.relative_to(root).as_posix(), "folder": folder,
                             "label": spec["class_map"][folder]})
    df = pd.DataFrame(rows)
    bad = set(df["label"]) - set(classes) - {ANY_DISEASE}
    if bad:
        raise SystemExit(f"class_map maps to unknown classes: {bad}")
    df["class_id"] = df["label"].map({**{c: i for i, c in enumerate(classes)}, ANY_DISEASE: -1})
    return df


def overlap_with_raw(ext: pd.DataFrame, ext_root: Path, raw_root: Path, out: Path) -> pd.DataFrame:
    raw = sorted(p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    with ThreadPoolExecutor(8) as ex:
        R = np.stack(list(ex.map(dihedral, raw)))[:, 0, :]
        E = np.stack(list(ex.map(dihedral, [ext_root / p for p in ext["path"]])))
        raw_sha = dict(zip(ex.map(sha, raw), raw))
        ext_sha = list(ex.map(sha, [ext_root / p for p in ext["path"]]))
    sim = np.einsum("nkd,md->nkm", E, R).max(1)                       # (n_ext, n_raw)
    j = sim.argmax(1)
    raw_names = {p.name.lower() for p in raw}
    o = ext.copy()
    o["sha_match"] = [s in raw_sha for s in ext_sha]
    o["best_corr"] = sim.max(1)
    o["best_raw_path"] = [raw[k].relative_to(raw_root).as_posix() for k in j]
    o["best_raw_folder"] = [raw[k].parent.name for k in j]
    o["content_match"] = o["best_corr"] >= THR
    o["filename_match"] = [Path(p).name.lower() in raw_names for p in o["path"]]
    o["overlap"] = o["sha_match"] | o["content_match"]
    o.to_csv(out / "overlap_with_raw.csv", index=False)
    s = {"n_external": len(o), "n_overlap": int(o["overlap"].sum()), "n_sha_match": int(o["sha_match"].sum()),
         "n_content_match": int(o["content_match"].sum()), "n_filename_match": int(o["filename_match"].sum()),
         "n_filename_match_confirmed_by_content_or_sha": int((o["filename_match"] & o["overlap"]).sum()),
         "n_independent": int((~o["overlap"]).sum()),
         "independent_per_class": o[~o["overlap"]]["label"].value_counts().to_dict(),
         "overlap_per_class": o[o["overlap"]]["label"].value_counts().to_dict()}
    save_json(s, out / "overlap_summary.json")
    print("  overlap:", s)
    return o


def ext_features(cfg, spec, ext: pd.DataFrame, work: Path) -> np.ndarray:
    c = copy.deepcopy(cfg)
    c["paths"]["data_root"] = Path(spec["root"])
    c["paths"]["work_dir"] = work
    f = work / "features_clean.npy"
    key = hashlib.md5(("|".join(ext["path"]) + str(cfg["preprocess"]) + str(cfg["segmentation"])
                       + str(cfg["features"])).encode()).hexdigest()
    kf = work / "features_clean.key"
    if f.exists() and kf.exists() and kf.read_text() == key:
        return np.load(f)
    build_image_cache(c, ext)
    X = extract_matrix(cache_file(c), np.arange(len(ext)), worker_cfg(cfg), cfg["n_jobs"])
    np.save(f, X)
    kf.write_text(key)
    return X


def metrics_present(y, pred, P, classes) -> dict:
    present = sorted(set(y.tolist()))
    _, _, f, _ = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    m = full_metrics(y, pred, P, classes)
    m["macro_f1_present"] = float(f.mean())
    m["n_classes_present"] = len(present)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", action="append", required=True, help="configs/external/<name>.yaml")
    ap.add_argument("--models", nargs="*", default=["rf", "svm", "lr"])
    ap.add_argument("--n", type=int, help="number of splits (default: protocol n_repeats)")
    ap.add_argument("--perm-repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    classes = cfg["dataset"]["classes"]
    df = load_manifest(cfg)
    y = df["class_id"].values
    X, names = load_all_features(cfg, df)
    cols = columns_for_set(names, "all")
    gcols = group_columns(names)
    assert feature_schema(worker_cfg(cfg)) == names

    exts = []
    for f in args.ext:
        spec = yaml.safe_load(open(f))
        name = spec["name"]
        out = cfg["paths"]["results_dir"] / "cross_dataset" / name
        work = cfg["paths"]["work_dir"] / "cross_dataset" / name
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        print(f"== {name}")
        e = scan_external(spec, classes)
        print("  images per class:", e["label"].value_counts().to_dict())
        e = overlap_with_raw(e, Path(spec["root"]), cfg["paths"]["data_root"], out)
        Xe = ext_features(cfg, spec, e, work)
        exts.append({"name": name, "df": e, "X": Xe, "out": out, "work": work})

    splits = make_splits(df, cfg, n=args.n)
    rng = np.random.default_rng(cfg["seed"])
    res = {x["name"]: {"m": [], "pc": [], "ig": [], "if": [], "ag": [], "pred": [], "cm": {}} for x in exts}
    indomain = []
    for s in splits:
        tr, va, te = s["train"], s["val"], s["test"]
        tv = np.concatenate([tr, va])
        for mname in args.models:
            best, _ = tune(mname, X[tr][:, cols], y[tr], X[va][:, cols], y[va], cfg)
            model = make_model(mname, best, cfg, proba=True).fit(X[tv][:, cols], y[tv])
            Xt = X[te][:, cols]
            pt, Pt = model.predict(Xt), model.predict_proba(Xt)
            m_in = full_metrics(y[te], pt, Pt, classes)
            indomain.append({"split": s["id"], "model": mname, "params": str(best), "macro_f1": m_in["macro_f1"],
                             "accuracy": m_in["accuracy"]})
            print(f"split {s['id']} {mname}: in-domain macro-F1 {m_in['macro_f1']:.3f}", flush=True)
            imp_in_g = permutation_importance(model, Xt, y[te], [gcols[g] for g in gcols], args.perm_repeats, rng)
            imp_in_f = permutation_importance(model, Xt, y[te], [[i] for i in range(len(cols))], args.perm_repeats,
                                              rng)
            for x in exts:
                r, e = res[x["name"]], x["df"]
                Xe = x["X"][:, cols]
                pe, Pe = model.predict(Xe), model.predict_proba(Xe)
                ye = e["class_id"].values
                ind = ~e["overlap"].values
                base = {"split": s["id"], "seed": s["seed"], "model": mname, "params": str(best)}
                r["m"].append({**base, "subset": "own_test", "n": len(te), **metrics_present(y[te], pt, Pt, classes)})
                d = e[["path", "label", "overlap"]].copy()
                d["y_true"], d["y_pred"], d["confidence"] = ye, pe, Pe.max(1)
                d.insert(0, "model", mname)
                d.insert(0, "split", s["id"])
                r["pred"].append(d)
                if (ye < 0).any():          # binary set: healthy vs any disease
                    h = classes.index("healthy")
                    for sub, mk in (("external_independent", ind), ("external_overlap", ~ind),
                                    ("external_all", np.ones(len(e), bool))):
                        hm, dm = mk & (ye == h), mk & (ye < 0)
                        if mk.sum():
                            hr = float((pe[hm] == h).mean()) if hm.any() else np.nan
                            dr = float((pe[dm] != h).mean()) if dm.any() else np.nan
                            r["m"].append({**base, "subset": sub, "n": int(mk.sum()), "n_healthy": int(hm.sum()),
                                           "n_diseased": int(dm.sum()), "healthy_recall": hr,
                                           "disease_detection": dr, "binary_balanced_accuracy": np.nanmean([hr, dr]),
                                           **{f"pred_frac_{c}": float((pe[dm] == i).mean()) if dm.any() else np.nan
                                              for i, c in enumerate(classes)}})
                    continue
                for sub, mask in (("external_independent", ind), ("external_overlap", ~ind), ("external_all", None)):
                    mk = np.ones(len(e), bool) if mask is None else mask
                    if mk.sum():
                        r["m"].append({**base, "subset": sub, "n": int(mk.sum()),
                                       **metrics_present(ye[mk], pe[mk], Pe[mk], classes)})
                present = sorted(set(ye[ind].tolist()))
                p_, r_, f_, n_ = precision_recall_fscore_support(ye[ind], pe[ind], labels=list(range(len(classes))),
                                                                 zero_division=0)
                for i, c in enumerate(classes):
                    r["pc"].append({**base, "class": c, "support": int(n_[i]), "precision": p_[i], "recall": r_[i],
                                    "f1": f_[i]})
                cm = confusion_matrix(ye[ind], pe[ind], labels=list(range(len(classes))))
                r["cm"][mname] = r["cm"].get(mname, 0) + cm
                # explanations on the independent external subset (only classes the model was trained on)
                if ind.sum() >= 20:
                    ex_g = permutation_importance(model, Xe[ind], ye[ind], [gcols[g] for g in gcols],
                                                  args.perm_repeats, rng)
                    ex_f = permutation_importance(model, Xe[ind], ye[ind], [[i] for i in range(len(cols))],
                                                  args.perm_repeats, rng)
                    for gi, g in enumerate(gcols):
                        r["ig"].append({**base, "group": g, "own_test": imp_in_g[gi], "external": ex_g[gi]})
                    imp_rf = model.feature_importances_ if mname == "rf" else np.full(len(cols), np.nan)
                    for i in range(len(cols)):
                        r["if"].append({**base, "feature": names[cols[i]], "own_test": imp_in_f[i],
                                        "external": ex_f[i], "rf_impurity": imp_rf[i]})
                    r["ag"].append({**base, **agreement(imp_in_f, ex_f)})

    ind_df = pd.DataFrame(indomain)
    ref = Path(__file__).resolve().parents[1] / "results" / "protocol_main" / "metrics_per_split.csv"   # committed
    for x in exts:
        r, out = res[x["name"]], x["out"]
        ind_df.to_csv(out / "indomain_check.csv", index=False)
        m = pd.DataFrame(r["m"])
        m.to_csv(out / "metrics_per_split.csv", index=False)
        cols_s = [c for c in ["macro_f1", "macro_f1_present", "accuracy", "balanced_accuracy", "mcc", "ece",
                              "healthy_recall", "disease_detection", "binary_balanced_accuracy"]
                  + [f"pred_frac_{c}" for c in classes] if c in m.columns]
        sm = m.groupby(["subset", "model"])[cols_s].agg(["mean", "std"])
        sm.columns = ["_".join(c) for c in sm.columns]
        sm = sm.reset_index().merge(m.groupby(["subset", "model"])["n"].first().reset_index())
        sm.to_csv(out / "summary.csv", index=False)
        if r["pc"]:
            pd.DataFrame(r["pc"]).groupby(["model", "class"])[["support", "precision", "recall", "f1"]].mean().to_csv(
                out / "per_class_f1.csv")
        rows = []
        for mname, cm in r["cm"].items():
            for i, a in enumerate(classes):
                for j, b in enumerate(classes):
                    rows.append({"model": mname, "true": a, "pred": b, "count": int(cm[i, j])})
        pd.DataFrame(rows).to_csv(out / "confusion_pooled.csv", index=False)
        for key, fn in (("ig", "importance_group.csv"), ("if", "importance_feature.csv"),
                        ("ag", "importance_agreement.csv")):
            if r[key]:
                pd.DataFrame(r[key]).to_csv(out / fn, index=False)
        pd.concat(r["pred"]).to_csv(x["work"] / "predictions.csv.gz", index=False)
        print(f"\n== {x['name']} (mean over {len(splits)} splits)")
        show = [c for c in ["subset", "model", "n", "macro_f1_present_mean", "macro_f1_present_std", "accuracy_mean",
                            "healthy_recall_mean", "disease_detection_mean"] if c in sm.columns]
        print(sm[show].to_string(index=False))
    if ref.exists():
        p1 = pd.read_csv(ref)
        p1 = p1[(p1.feature_set == "all") & (p1.condition == "clean")][["split", "model", "macro_f1"]]
        chk = ind_df.merge(p1, on=["split", "model"], suffixes=("", "_protocol_main"))
        chk["abs_diff"] = (chk["macro_f1"] - chk["macro_f1_protocol_main"]).abs()
        chk.to_csv(exts[0]["out"].parent / "indomain_vs_protocol_main.csv", index=False)
        print(f"\nin-domain vs committed protocol_main: max |diff| = {chk['abs_diff'].max():.2e} over {len(chk)} rows")


if __name__ == "__main__":
    main()
