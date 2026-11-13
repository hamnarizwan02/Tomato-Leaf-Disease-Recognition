"""Stage 19: source grouping of the supplied Augmented Dataset for a grouped evaluation (Task 2).

Run with configs/supplied/grouped.yaml, TOMATO_DATA_ROOT pointing at the Augmented Dataset and TOMATO_MANIFEST at
<run>/work/manifest_pipeline.csv. Nothing is trained here; the rules below were fixed before any model was run.

    python scripts/19_group_supplied.py

Grouping rules (an image's group is the union of every edge below; same class only):
    pipeline : stage-01 grouping, i.e. filename regex (dataset.group_regex) + flip-aware dHash candidates confirmed by
               32x32 colour-thumbnail correlation >= duplicates.min_correlation
    name     : same class and same source name (scripts/18 source_name: lower-cased, both augmentation naming schemes
               and the Roboflow export id stripped)
    content  : 16x16 grey thumbnails correlate >= 0.95 under one of the 8 rotations/flips (scripts/18 rule, frozen
               2026-10-27, read from the original files)
Cross-class handling (fixed in advance, same policy as the raw set): every image whose group contains an image in a
    cross-class *content* or *pipeline* pair is dropped (the photo is filed under two classes, its label is unknowable).
    Same source NAME under two classes is only reported, not dropped: short names can collide by chance.
Images whose source cannot be traced stay as singletons unless a rule links them. Copies the rules cannot detect
    (crops, strong recolouring, renamed files) may still cross splits: the grouped protocol is stricter, not leak-free.

Outputs (results_dir/supplied_grouping/):
    manifest_grouped.csv   rows sorted by class then path; group = merged group; split = grouped seed-42 split
    manifest_random.csv    identical rows and order; group = path (ungrouped control: shares the feature store)
    grouping_summary.json, group_sizes.csv, dropped_cross_class.csv, name_collisions_cross_class.csv,
    split_family_check.csv (per protocol and split: test images whose detected family has a member in train/val)
"""
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import build_manifest, group_split  # noqa: E402
from src.protocol import make_splits  # noqa: E402
from src.utils import save_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("stage18", Path(__file__).with_name("18_export_predictions.py"))
s18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s18)                     # the frozen sibling rules: source_name, _dihedral, CONTENT_THRESHOLD


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def content_pairs(D: np.ndarray, thr: float, chunk: int = 256):
    """All pairs (i < j) whose best dihedral thumbnail correlation is >= thr. D: (n, 8, d), D[:, 0] = original."""
    base = D[:, 0, :]
    out_i, out_j, out_c = [], [], []
    for a in range(0, len(D), chunk):
        sim = np.einsum("nkd,md->nkm", D[a:a + chunk], base).max(1)        # (chunk, n)
        ii, jj = np.nonzero(sim >= thr)
        keep = jj > ii + a
        out_i.append(ii[keep] + a), out_j.append(jj[keep]), out_c.append(sim[ii[keep], jj[keep]])
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_c)


def family_check(df, fam, splits, protocol):
    rows = []
    for s in splits:
        ref = set(fam[np.concatenate([s["train"], s["val"]])])
        te = s["test"]
        hit = np.array([f in ref for f in fam[te]])
        rows.append({"protocol": protocol, "split": s["id"], "seed": s["seed"], "n_test": len(te),
                     "n_test_with_family_in_trainval": int(hit.sum()), "frac": float(hit.mean())})
    return rows


def main():
    cfg = load_config()
    if cfg["dataset"]["split_strategy"] != "group":
        raise SystemExit("run with TOMATO_CONFIG=configs/supplied/grouped.yaml")
    out = cfg["paths"]["results_dir"] / "supplied_grouping"
    out.mkdir(parents=True, exist_ok=True)
    root = cfg["paths"]["data_root"]

    df = build_manifest(cfg)                       # pipeline grouping; drops pipeline-detected cross-class copies
    n0 = len(df)
    labels = df["label"].values
    uf = UF(n0)
    first = {}
    for i, g in enumerate(df["group"].values):     # pipeline edges
        if g in first:
            uf.union(i, first[g])
        else:
            first[g] = i
    n_pipeline_groups = len(first)

    names = df["path"].map(lambda p: s18.source_name(Path(p).name)).values
    by_name = pd.DataFrame({"i": np.arange(n0), "label": labels, "name": names})
    for _, idx in by_name.groupby(["label", "name"])["i"]:   # name edges, same class
        idx = idx.values
        for j in idx[1:]:
            uf.union(int(idx[0]), int(j))
    multi = by_name.groupby("name")["label"].nunique()
    collide = by_name[by_name["name"].isin(multi[multi > 1].index)].assign(path=df["path"].values[
        by_name["name"].isin(multi[multi > 1].index).values])
    collide.sort_values(["name", "label"]).to_csv(out / "name_collisions_cross_class.csv", index=False)

    print(f"reading {n0} thumbnails (8 rotations/flips each) ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        D = np.stack(list(ex.map(lambda p: s18._dihedral(root / p), df["path"].values))).astype(np.float32)
    ci, cj, cc = content_pairs(D, s18.CONTENT_THRESHOLD)
    same = labels[ci] == labels[cj]
    for i, j in zip(ci[same], cj[same]):           # content edges, same class
        uf.union(int(i), int(j))
    fam = np.array([uf.find(i) for i in range(n0)])

    cross_imgs = set(ci[~same]) | set(cj[~same])
    bad = {fam[i] for i in cross_imgs}
    drop = np.isin(fam, list(bad))
    df.assign(family=fam)[drop][["path", "label", "family"]].sort_values("family").to_csv(
        out / "dropped_cross_class.csv", index=False)
    keep = ~drop
    g = df[keep].copy()
    g["group"] = [f"{lab}/fam{f:05d}" for lab, f in zip(g["label"], fam[keep])]
    g = g.sort_values(["class_id", "path"]).reset_index(drop=True)
    g["split"] = group_split(g, cfg["dataset"]["group_split"], cfg["seed"]).values
    sizes = g["group"].value_counts()
    sizes.rename_axis("group").reset_index(name="n_images").to_csv(out / "group_sizes.csv", index=False)
    g.to_csv(out / "manifest_grouped.csv", index=False)

    r = g.copy()
    r["group"] = r["label"] + "/" + r["path"]
    r["split"] = group_split(r, cfg["dataset"]["group_split"], cfg["seed"]).values
    r.to_csv(out / "manifest_random.csv", index=False)

    famg = g["group"].values                       # detected family of every kept row (same rows in both manifests)
    rows = family_check(g, famg, make_splits(g, cfg), "grouped") + family_check(r, famg, make_splits(r, cfg), "random")
    chk = pd.DataFrame(rows)
    chk.to_csv(out / "split_family_check.csv", index=False)

    summary = {
        "n_images_scanned": int(n0), "n_pipeline_groups": int(n_pipeline_groups),
        "content_threshold": s18.CONTENT_THRESHOLD, "n_content_pairs": int(len(ci)),
        "n_content_pairs_cross_class": int((~same).sum()),
        "n_names_under_two_classes": int((multi > 1).sum()),
        "n_images_dropped_cross_class": int(drop.sum()), "n_images_kept": int(keep.sum()),
        "n_groups": int(sizes.size), "n_singletons": int((sizes == 1).sum()), "largest_group": int(sizes.max()),
        "largest_group_frac": float(sizes.max() / keep.sum()),
        "per_class_kept": g["label"].value_counts().to_dict(),
        "per_class_groups": g.groupby("label")["group"].nunique().to_dict(),
        "grouped_test_frac_family_in_trainval_max": float(chk[chk.protocol == "grouped"]["frac"].max()),
        "random_test_frac_family_in_trainval_mean": float(chk[chk.protocol == "random"]["frac"].mean()),
    }
    save_json(summary, out / "grouping_summary.json")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["largest_group_frac"] > 0.02:
        print("WARNING: the largest group holds more than 2% of the images: inspect group_sizes.csv for chaining "
              "before running the protocol. The rules are not re-tuned; report it.")
    print(pd.crosstab(g["label"], g["split"]))


if __name__ == "__main__":
    main()
