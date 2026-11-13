"""Stage 15 (Member A): label and data quality (section 15, stage 3).

    python scripts/15_label_audit.py [--top 50] [--repeats 3]

Out-of-fold class probabilities from repeated grouped 5-fold CV (random forest, all features, stage-4 parameters if
present), averaged per image. Two views:
  - the most confidently wrong images (to inspect by eye: results/label_audit_top.png)
  - confident learning (Northcutt et al.): per-class threshold t_j = mean predicted p_j over images labelled j; an image
    is flagged when some other class j clears its threshold and beats the given label.
Nothing is removed automatically. After looking at the images, list confirmed label errors in
data/label_exclusions.csv (path,reason) and re-run stage 01: the whole source group of each listed photo is dropped.
Outputs (results/): label_audit.csv, label_audit_top.png, label_audit_summary.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_cache, load_manifest  # noqa: E402
from src.evaluate import pretty  # noqa: E402
from src.features import load_all_features  # noqa: E402
from src.models import make_model  # noqa: E402
from src.utils import Timer, load_json, save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    k = len(classes)
    df = load_manifest(cfg)
    y, groups = df["class_id"].values, df["group"].values
    X, _ = load_all_features(cfg, df)
    bp = res / "best_params.json"
    params = load_json(bp)["rf"] if bp.exists() else {}

    P, n = np.zeros((len(df), k)), np.zeros(len(df))
    for r in range(args.repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=cfg["seed"] + r)
        with Timer(f"out-of-fold probabilities, repeat {r + 1}/{args.repeats}"):
            for tr, te in cv.split(X, y, groups):
                m = make_model("rf", params, cfg).fit(X[tr], y[tr])
                P[np.ix_(te, m.classes_)] += m.predict_proba(X[te])
                n[te] += 1
    P /= n[:, None]

    p_given = P[np.arange(len(y)), y]
    thresh = np.array([P[y == j, j].mean() if (y == j).any() else 1.0 for j in range(k)])
    above = np.where(P >= thresh, P, -1.0)
    above[np.arange(len(y)), y] = -1.0          # only other classes can be the suspected true label
    cand = above.argmax(1)
    cl_flag = (above.max(1) > 0) & (P[np.arange(len(y)), cand] > p_given)
    pred = P.argmax(1)
    out = pd.DataFrame({"path": df["path"], "group": groups, "label": df["label"], "predicted": np.array(classes)[pred],
                        "p_given": p_given, "p_predicted": P.max(1),
                        "margin": P.max(1) - p_given, "confident_learning_flag": cl_flag,
                        "suspected_label": np.where(cl_flag, np.array(classes)[cand], "")})
    out = out.sort_values("margin", ascending=False)
    out.to_csv(res / "label_audit.csv", index=False)

    top = out[out.predicted != out.label].head(args.top)
    imgs = load_cache(cfg, df)
    pos = {p: i for i, p in enumerate(df["path"])}
    ncol = 10
    nrow = max(1, int(np.ceil(len(top) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.2 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, r) in zip(np.ravel(axes), top.iterrows()):
        ax.imshow(imgs[pos[r.path]])
        ax.set_title(f"L: {pretty(r.label)[:14]}\nP: {pretty(r.predicted)[:14]} {r.p_predicted:.2f}", fontsize=6)
    fig.tight_layout()
    fig.savefig(res / "label_audit_top.png", dpi=110)
    plt.close(fig)

    s = {"n_images": len(df), "n_wrong_out_of_fold": int((pred != y).sum()),
         "n_confident_learning_flags": int(cl_flag.sum()),
         "flags_by_label": out[out.confident_learning_flag].label.value_counts().to_dict(),
         "class_thresholds": dict(zip(classes, thresh.round(3)))}
    save_json(s, res / "label_audit_summary.json")
    print(f"{s['n_confident_learning_flags']} images flagged by confident learning; "
          f"{s['n_wrong_out_of_fold']} wrong out of fold. Inspect results/label_audit_top.png, then list confirmed "
          f"label errors in data/label_exclusions.csv (path,reason) and re-run stage 01.")


if __name__ == "__main__":
    main()
