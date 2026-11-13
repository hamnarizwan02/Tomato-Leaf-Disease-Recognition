"""Stage 24: multi-source manifest for Phase 3 (docs/phase3_design_2026-11-12.md).

    python scripts/24_build_sources.py --out F:/NUST/ML/runs/p4_multisource

Reads configs/sources.yaml and the committed raw manifest. Rules (fixed in the design):
  - our 813 raw images are split into pd_raw (identical to some PlantDoc image) and pk_raw (the rest); labels and
    groups from the committed manifest are kept;
  - duplicates: identical bytes, or 16x16 grey thumbnails with r >= 0.95 under one of the 8 rotations/flips;
  - an image is dropped if it duplicates an image of a higher-priority source (raw, including the 17 raw images the
    main manifest dropped, > plantdoc > plantwild > tomato_village > nazirpur > brahmanbaria > plantvillage);
  - groups = connected duplicates inside one source (raw keeps its committed groups, merged with these).
Outputs (<out>/sources/): manifest_sources.csv, dropped_cross_source.csv, label_conflicts.csv, counts.csv,
summary.json
"""
import argparse
import hashlib
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.config import load_config  # noqa: E402
from src.utils import save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s20", Path(__file__).with_name("20_cross_dataset.py"))
s20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s20)                    # dihedral(), THR (frozen content rule)
EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sources.yaml"))
    args = ap.parse_args()
    out = Path(args.out) / "sources"
    out.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(open(args.config))
    data_root = Path(spec["data_root"])
    classes = load_config()["dataset"]["classes"]

    rows = []
    raw_root = spec["raw"]["root"]
    man = pd.read_csv(ROOT / spec["raw"]["manifest"])
    for r in man.itertuples():
        rows.append({"path": f"{raw_root}/{r.path}", "source": "raw", "folder": r.path.split("/")[0],
                     "label6": r.label, "binary": "healthy" if r.label == "healthy" else "diseased",
                     "priority": 0, "group0": f"raw/{r.group}", "keep_candidate": True})
    in_manifest = set(man["path"])
    for p in sorted((data_root / raw_root).rglob("*")):        # raw images dropped by the main manifest: match only
        rel = p.relative_to(data_root / raw_root).as_posix()
        if p.is_file() and p.suffix.lower() in EXTS and rel not in in_manifest:
            rows.append({"path": f"{raw_root}/{rel}", "source": "raw_dropped", "folder": p.parent.name,
                         "label6": None, "binary": None, "priority": 0, "group0": None, "keep_candidate": False})
    for name, s in spec["sources"].items():
        n0 = len(rows)
        for p in sorted((data_root / s["root"]).rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS and p.parent.name in s["folders"]:
                l6, b = s["folders"][p.parent.name]
                rows.append({"path": p.relative_to(data_root).as_posix(), "source": name, "folder": p.parent.name,
                             "label6": l6, "binary": b, "priority": s["priority"], "group0": None,
                             "keep_candidate": True})
        print(f"{name}: {len(rows) - n0} images")
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"total {n} images; reading thumbnails and hashes ...")

    paths = [data_root / p for p in df["path"]]
    with ThreadPoolExecutor(8) as ex:
        D = np.stack(list(ex.map(s20.dihedral, paths))).astype(np.float32)
        shas = list(ex.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), paths))
    df["sha256"] = shas
    base = D[:, 0, :]
    I, J, C = [], [], []
    for a in range(0, n, 256):
        sim = np.einsum("nkd,md->nkm", D[a:a + 256], base).max(1)
        ii, jj = np.nonzero(sim >= s20.THR)
        k = jj > ii + a
        I.append(ii[k] + a), J.append(jj[k]), C.append(sim[ii[k], jj[k]])
    I, J, C = np.concatenate(I), np.concatenate(J), np.concatenate(C)
    by_sha = df.groupby("sha256").indices
    extra = [(min(v), x) for v in by_sha.values() if len(v) > 1 for x in sorted(v)[1:]]
    if extra:
        ei, ej = np.array([e[0] for e in extra]), np.array([e[1] for e in extra])
        I, J, C = np.r_[I, ei], np.r_[J, ej], np.r_[C, np.ones(len(ei))]
    pairs = pd.DataFrame({"i": I, "j": J, "corr": C}).drop_duplicates(["i", "j"])
    print(f"{len(pairs)} duplicate pairs")

    src = df["source"].values
    dedup_src = np.where(np.isin(src, ["raw", "raw_dropped"]), "raw", src)
    prio = df["priority"].values
    # pd_raw: raw images with a PlantDoc duplicate
    pd_hit = set()
    for i, j in zip(pairs.i, pairs.j):
        for a, b in ((i, j), (j, i)):
            if src[a] == "raw" and src[b] == "plantdoc":
                pd_hit.add(a)
    df.loc[list(pd_hit), "source"] = "pd_raw"
    df.loc[(df["source"] == "raw"), "source"] = "pk_raw"

    drop, partner = np.zeros(n, bool), {}
    for i, j in zip(pairs.i, pairs.j):
        if dedup_src[i] == dedup_src[j]:
            continue
        lo, hi = (i, j) if prio[i] > prio[j] else (j, i)      # lo = lower priority (larger number) -> dropped
        if prio[lo] > prio[hi]:
            drop[lo] = True
            partner.setdefault(lo, hi)
    dr = df.loc[drop, ["path", "source", "label6", "binary"]].copy()
    dr["kept_copy"] = [df.at[partner[i], "path"] for i in dr.index]
    dr["kept_source"] = [df.at[partner[i], "source"] for i in dr.index]
    dr["kept_label6"] = [df.at[partner[i], "label6"] for i in dr.index]
    dr["kept_binary"] = [df.at[partner[i], "binary"] for i in dr.index]
    dr.to_csv(out / "dropped_cross_source.csv", index=False)
    conf = dr[(dr["kept_source"] != "raw_dropped") & ((dr["label6"].fillna("-") != dr["kept_label6"].fillna("-"))
                                                      | (dr["binary"] != dr["kept_binary"]))]
    conf.to_csv(out / "label_conflicts.csv", index=False)

    keep = (~drop) & df["keep_candidate"].values
    # within-source groups
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs.i, pairs.j):
        if keep[i] and keep[j] and dedup_src[i] == dedup_src[j]:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    first = {}
    for i in np.where(keep)[0]:
        g = df.at[i, "group0"]
        if isinstance(g, str):
            if g in first:
                ri, rj = find(i), find(first[g])
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            else:
                first[g] = i
    roots = np.array([find(i) for i in range(n)])
    m = df[keep].copy()
    m["group"] = [f"{dedup_src[i]}/g{roots[i]:05d}" for i in m.index]
    m["class_id6"] = m["label6"].map({c: k for k, c in enumerate(classes)}).fillna(-1).astype(int)
    m["y_bin"] = (m["binary"] == "diseased").astype(int)
    m = m[["path", "source", "folder", "label6", "class_id6", "binary", "y_bin", "group", "sha256"]]
    m = m.sort_values(["source", "path"]).reset_index(drop=True)
    m.to_csv(out / "manifest_sources.csv", index=False)

    counts = pd.crosstab(m["source"], m["label6"].fillna("(binary only)"), margins=True)
    counts["healthy_bin"] = m.groupby("source")["binary"].apply(lambda s: (s == "healthy").sum())
    counts.to_csv(out / "counts.csv")
    summary = {"n_scanned": int(n), "n_pairs": int(len(pairs)), "n_dropped_cross_source": int(drop.sum()),
               "dropped_by_source": dr["source"].value_counts().to_dict(),
               "n_label_conflicts": int(len(conf)), "n_pd_raw": int((m.source == "pd_raw").sum()),
               "n_pk_raw": int((m.source == "pk_raw").sum()), "n_final": int(len(m)),
               "per_source": m["source"].value_counts().to_dict(),
               "groups_per_source": m.groupby("source")["group"].nunique().to_dict(),
               "largest_group": int(m["group"].value_counts().max())}
    save_json(summary, out / "summary.json")
    print(counts.to_string())
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
