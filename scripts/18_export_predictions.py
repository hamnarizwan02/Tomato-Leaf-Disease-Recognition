"""Stage 18: export stage-4 predictions and probabilities, and break test metrics down by detected siblings.

Made for the supplied-split benchmark (dataset.split_strategy: official), where the supplied train/val/test folders
hold augmented copies of the same photos. Stage 4 itself is not changed: this script reuses its saved models and
best parameters.

    python scripts/18_export_predictions.py --flags-only   # 1. freeze the sibling flags (before stage 4 is read)
    python scripts/04_baseline.py                          # 2. tune on val, refit on train+val, test once
    python scripts/18_export_predictions.py                # 3. export + checks + sibling breakdown

Sibling rules (frozen; a test image with no detected sibling is NOT guaranteed leakage-free):
    name    : same class and same source name in train or val, after lower-casing and stripping both augmentation
              naming schemes ("...aug414", "..._aug_5aug62", "_mirror", "_change_270", "_hight", ...) and the
              Roboflow export id ("..._jpg.rf.<32 hex>")
    content : any train or val image (any class) whose 16x16 grey thumbnail correlates >= 0.95 with one of the
              8 rotations/flips of the test image's thumbnail (read from the original files)

Outputs (results/):
    sibling_flags_test.csv, sibling_rules.json                 (--flags-only)
    predictions_test.csv.gz   path, label, y_true, model, y_pred, prob_<class>, sibling flags
    predictions_val.csv.gz    the same for validation, from the tuned train-only fits
    export_metrics_test.csv   accuracy, macro-F1 (+ bootstrap CI), balanced accuracy, MCC, ECE, Brier
    export_per_class_test.csv, confusion_<model>_test_counts.csv
    sibling_breakdown_test.csv, sibling_per_class_test.csv
    export_checks.csv          every check; the script exits 1 if one fails
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_manifest, split_indices  # noqa: E402
from src.evaluate import bootstrap_ci, confusion, full_metrics, per_class_table  # noqa: E402
from src.features import load_clean_features  # noqa: E402
from src.models import MODEL_NAMES, make_model  # noqa: E402
from src.utils import load_json, save_json, set_seed  # noqa: E402

CONTENT_THRESHOLD = 0.95
THUMB = 16
_AUG = re.compile(r"(_?aug_?\d+)+$")
_EXT = re.compile(r"\.(jpe?g|png)$")
_ROBOFLOW = re.compile(r"_(jpe?g|png)\.rf\.[0-9a-f]{32}$")
_SCHEME2 = re.compile(r"(_(mirror(_vertical)?|change_\d+|hight|high|lower|low|flip\w*|rot\w*|bright\w*|dark\w*))+$")


# --------------------------------------------------------------------------- sibling flags
def source_name(filename: str) -> str:
    s = _AUG.sub("", Path(filename).stem.lower())
    s = _ROBOFLOW.sub("", _EXT.sub("", s))
    return _SCHEME2.sub("", s)


def _thumb(img: np.ndarray) -> np.ndarray:
    t = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    return t / (np.linalg.norm(t) or 1.0)


def _dihedral(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"unreadable image: {path}")
    t = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for k in range(4):
        r = np.rot90(t, k)
        for v in (r, r[:, ::-1]):
            v = v - v.mean()
            out.append((v / (np.linalg.norm(v) or 1.0)).ravel())
    return np.stack(out)


def sibling_flags(cfg, df, idx) -> pd.DataFrame:
    root = cfg["paths"]["data_root"]
    names = df["path"].map(lambda p: source_name(Path(p).name)).values
    labels = df["label"].values
    ref = np.concatenate([idx["train"], idx["val"]])
    ref_names = set(zip(labels[ref], names[ref]))
    te = idx["test"]
    by_name = np.array([(labels[i], names[i]) in ref_names for i in te])

    ref_thumbs = np.stack([_thumb(cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE)).ravel() for p in df["path"].values[ref]])
    te_thumbs = np.stack([_dihedral(root / p) for p in df["path"].values[te]])          # (n_test, 8, 256)
    sim = np.einsum("nkd,md->nkm", te_thumbs, ref_thumbs).max(1)                         # (n_test, n_ref)
    best = sim.max(1)
    j = ref[sim.argmax(1)]
    return pd.DataFrame({"path": df["path"].values[te], "label": labels[te], "source_name": names[te],
                         "sibling_name": by_name, "content_best_corr": best,
                         "content_best_path": df["path"].values[j], "content_best_label": labels[j],
                         "sibling_content": best >= CONTENT_THRESHOLD,
                         "sibling": by_name | (best >= CONTENT_THRESHOLD)})


def freeze_flags(cfg, df, idx, res) -> None:
    out, rules = res / "sibling_flags_test.csv", res / "sibling_rules.json"
    if out.exists():
        raise SystemExit(f"{out} already exists: the flags are frozen, delete it by hand only to redefine them")
    if (res / "baseline_test_metrics.csv").exists():
        raise SystemExit("stage-4 results already exist: freeze the flags before stage 4 is run")
    fl = sibling_flags(cfg, df, idx)
    fl.to_csv(out, index=False)
    save_json({"rules": __doc__.split("Sibling rules")[1].split("Outputs")[0].strip(),
               "content_threshold": CONTENT_THRESHOLD, "thumb": THUMB,
               "regex": {"aug": _AUG.pattern, "ext": _EXT.pattern, "roboflow": _ROBOFLOW.pattern,
                         "scheme2": _SCHEME2.pattern},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "flags_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
               "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_test": len(fl), "n_sibling": int(fl.sibling.sum()), "n_sibling_name": int(fl.sibling_name.sum()),
               "n_sibling_content": int(fl.sibling_content.sum())}, rules)
    print(f"froze {out.name}: {int(fl.sibling.sum())}/{len(fl)} test images have a detected sibling "
          f"(name {int(fl.sibling_name.sum())}, content {int(fl.sibling_content.sum())})")


# --------------------------------------------------------------------------- export
def _diff_ci(y, p, mask, n: int = 2000, seed: int = 0):
    """Bootstrap 95% CI of (accuracy with sibling - accuracy without), resampling within each group."""
    rng = np.random.default_rng(seed)
    a, b = np.where(mask)[0], np.where(~mask)[0]
    ok = (y == p).astype(float)
    d = [ok[rng.choice(a, len(a))].mean() - ok[rng.choice(b, len(b))].mean() for _ in range(n)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags-only", action="store_true", help="compute and freeze the sibling flags, then stop")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    df = load_manifest(cfg)
    idx = split_indices(df)
    if args.flags_only:
        freeze_flags(cfg, df, idx, res)
        return

    t0 = time.time()
    rules = load_json(res / "sibling_rules.json")
    flag_file = res / "sibling_flags_test.csv"
    if hashlib.sha256(flag_file.read_bytes()).hexdigest() != rules["flags_sha256"]:
        raise SystemExit(f"{flag_file.name} changed after it was frozen")
    flags = pd.read_csv(flag_file)
    if list(flags["path"]) != list(df["path"].values[idx["test"]]):
        raise SystemExit("sibling flags do not match the manifest's test rows")

    y = df["class_id"].values
    X, _ = load_clean_features(cfg)
    ytr, yva, yte = y[idx["train"]], y[idx["val"]], y[idx["test"]]
    Xtv, ytv = np.vstack([X["train"], X["val"]]), np.concatenate([ytr, yva])
    best = load_json(res / "best_params.json")
    saved = pd.read_csv(res / "baseline_test_metrics.csv").set_index("model")
    saved_pc = pd.read_csv(res / "baseline_per_class_test.csv")
    models_dir = cfg["paths"]["work_dir"] / "models"
    sib = flags["sibling"].values.astype(bool)

    checks, preds, val_preds, metrics, per_class, breakdown, pc_sib = [], [], [], [], [], [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")

    for name in MODEL_NAMES:
        print(f"{name}:")
        stage4 = joblib.load(models_dir / f"{name}_all.joblib")
        pred_saved = stage4.predict(X["test"])
        if hasattr(stage4, "predict_proba"):
            model, pred, proba = stage4, pred_saved, stage4.predict_proba(X["test"])
        else:       # stage 4 saves a plain SVC: refit the PlattSVC variant on the same rows and parameters
            model = make_model(name, best[name], cfg, proba=True).fit(Xtv, ytv)
            pred, proba = model.predict(X["test"]), model.predict_proba(X["test"])
            check(f"{name}: probability refit predicts exactly like the saved model",
                  np.array_equal(pred, pred_saved), f"{int((pred != pred_saved).sum())} differ")
        check(f"{name}: probabilities are finite and sum to 1", np.isfinite(proba).all()
              and np.allclose(proba.sum(1), 1, atol=1e-6))
        check(f"{name}: probability column order is the class order",
              list(model.classes_) == list(range(len(classes))), list(model.classes_))

        m = full_metrics(yte, pred, proba, classes)
        lo, hi = bootstrap_ci(yte, pred)
        metrics.append({"model": name, "n_test": len(yte), **m, "macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi,
                        "best_params": json.dumps(best[name])})
        for k, col in (("accuracy", "test_accuracy"), ("macro_f1", "test_macro_f1")):
            check(f"{name}: recomputed {k} equals stage 4", abs(m[k] - saved.loc[name, col]) < 1e-12,
                  f"{m[k]:.6f} vs {saved.loc[name, col]:.6f}")
        check(f"{name}: recomputed bootstrap CI equals stage 4",
              abs(lo - saved.loc[name, "test_macro_f1_ci_lo"]) < 1e-12 and abs(hi - saved.loc[name, "test_macro_f1_ci_hi"]) < 1e-12)
        pc = per_class_table(yte, pred, classes)
        ref_pc = saved_pc[saved_pc["model"] == name].reset_index(drop=True)
        check(f"{name}: per-class table equals stage 4",
              np.allclose(pc[["precision", "recall", "f1", "support"]].values,
                          ref_pc[["precision", "recall", "f1", "support"]].values, atol=1e-12))
        pc.insert(0, "model", name)
        per_class.append(pc)
        cm = pd.DataFrame(confusion(yte, pred, len(classes)), index=[f"true_{c}" for c in classes],
                          columns=[f"pred_{c}" for c in classes])
        cm.to_csv(res / f"confusion_{name}_test_counts.csv")

        p = flags[["path", "label", "sibling", "sibling_name", "sibling_content"]].copy()
        p.insert(2, "y_true", yte)
        p.insert(3, "model", name)
        p.insert(4, "y_pred", pred)
        for i, c in enumerate(classes):
            p[f"prob_{c}"] = proba[:, i]
        preds.append(p)

        # validation predictions of the tuned model fitted on train only (the fit that model selection scored)
        vm = make_model(name, best[name], cfg, proba=True).fit(X["train"], ytr)
        vpred, vproba = vm.predict(X["val"]), vm.predict_proba(X["val"])
        tune_tab = pd.read_csv(res / f"baseline_tuning_{name}.csv")
        check(f"{name}: val macro-F1 of the train-only refit equals the best tuning score",
              abs(f1_score(yva, vpred, average="macro") - tune_tab.val_macro_f1.max()) < 1e-12)
        v = pd.DataFrame({"path": df["path"].values[idx["val"]], "label": df["label"].values[idx["val"]],
                          "y_true": yva, "model": name, "y_pred": vpred})
        for i, c in enumerate(classes):
            v[f"prob_{c}"] = vproba[:, i]
        val_preds.append(v)

        for grp, mask in (("detected sibling", sib), ("no detected sibling", ~sib)):
            gm = full_metrics(yte[mask], pred[mask], proba[mask], classes)
            glo, ghi = bootstrap_ci(yte[mask], pred[mask])
            breakdown.append({"model": name, "group": grp, "n": int(mask.sum()), "accuracy": gm["accuracy"],
                              "macro_f1": gm["macro_f1"], "macro_f1_ci_lo": glo, "macro_f1_ci_hi": ghi,
                              "balanced_accuracy": gm["balanced_accuracy"], "ece": gm["ece"]})
            for i, c in enumerate(classes):
                cmask = mask & (yte == i)
                pc_sib.append({"model": name, "group": grp, "class": c, "n": int(cmask.sum()),
                               "recall": float((pred[cmask] == i).mean()) if cmask.any() else np.nan})
        dlo, dhi = _diff_ci(yte, pred, sib)
        breakdown.append({"model": name, "group": "difference (sibling - no sibling)", "n": len(yte),
                          "accuracy": (pred[sib] == yte[sib]).mean() - (pred[~sib] == yte[~sib]).mean(),
                          "accuracy_diff_ci_lo": dlo, "accuracy_diff_ci_hi": dhi})

    pd.concat(preds).to_csv(res / "predictions_test.csv.gz", index=False)
    pd.concat(val_preds).to_csv(res / "predictions_val.csv.gz", index=False)
    pd.DataFrame(metrics).to_csv(res / "export_metrics_test.csv", index=False)
    pd.concat(per_class).to_csv(res / "export_per_class_test.csv", index=False)
    pd.DataFrame(breakdown).to_csv(res / "sibling_breakdown_test.csv", index=False)
    pd.DataFrame(pc_sib).to_csv(res / "sibling_per_class_test.csv", index=False)
    back = pd.read_csv(res / "predictions_test.csv.gz")
    check("exported predictions reload with the expected rows", len(back) == len(MODEL_NAMES) * len(yte))
    pd.DataFrame(checks).to_csv(res / "export_checks.csv", index=False)
    print(f"export done in {time.time() - t0:.1f}s; {sum(c['ok'] for c in checks)}/{len(checks)} checks pass")
    if not all(c["ok"] for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
