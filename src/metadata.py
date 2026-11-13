"""Image metadata (size, aspect, compression, file-name pattern) and source clusters.  (Owner: Member A)

Used for the source-shortcut work (section 15, stage 2): the metadata-only reference row in every protocol table, the
source clusters for leave-one-cluster-out evaluation, and pruning features that track image size or compression.
Reads image headers only (no decoding) and caches work/metadata.csv in manifest order.
"""
from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

META_FEATURES = ["width", "height", "aspect", "log_pixels", "file_bytes", "bytes_per_pixel"]

# first match wins; the raw dataset mostly has random names, so "other" is expected to dominate
NAME_PATTERNS = [
    ("whatsapp", r"whatsapp"),
    ("screenshot", r"screen ?shot"),
    ("camera", r"^(img|dsc|dscn|dcim|pxl|photo|image)[_\- ]?\d"),
    ("timestamp", r"(19|20)\d{2}[01]\d[0-3]\d"),
    ("digits", r"^[\d_\- ()]+$"),
    ("hex_or_uuid", r"^[0-9a-f\-]{16,}$"),
]


def name_pattern(path: str) -> str:
    stem = PurePosixPath(path).stem.lower()
    return next((name for name, rx in NAME_PATTERNS if re.search(rx, stem)), "other")


def image_metadata(cfg, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """One row per manifest row (same order): path + META_FEATURES + megapixels + name_pattern."""
    f = cfg["paths"]["work_dir"] / "metadata.csv"
    if f.exists() and not force:
        md = pd.read_csv(f)
        if md["path"].tolist() == df["path"].tolist():
            return md
    root = cfg["paths"]["data_root"]

    def meta(p):
        with Image.open(root / p) as im:
            w, h = im.size
        return w, h, (root / p).stat().st_size

    with ThreadPoolExecutor(max_workers=8) as ex:
        m = list(ex.map(meta, df["path"]))
    md = pd.DataFrame(m, columns=["width", "height", "file_bytes"])
    md.insert(0, "path", df["path"].values)
    md["megapixels"] = md.width * md.height / 1e6
    md["aspect"] = md.width / md.height
    md["log_pixels"] = np.log10(md.width * md.height)
    md["bytes_per_pixel"] = md.file_bytes / (md.width * md.height)
    md["name_pattern"] = [name_pattern(p) for p in df["path"]]
    md.to_csv(f, index=False)
    return md


def shortcut_correlations(X: np.ndarray, T: np.ndarray, rows=None) -> np.ndarray:
    """(n_features, n_targets) Spearman correlation of every feature with every metadata target on `rows`."""
    rows = np.arange(len(X)) if rows is None else np.asarray(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(np.column_stack([X[rows], T[rows]]))[0]
    rho = np.atleast_2d(rho)[: X.shape[1], X.shape[1]:]
    return np.nan_to_num(rho, nan=0.0)


def pruned_columns(X: np.ndarray, T: np.ndarray, rows, threshold: float) -> np.ndarray:
    """Columns whose |Spearman| with every target is at most `threshold`, measured on the training rows only."""
    return np.where(np.abs(shortcut_correlations(X, T, rows)).max(1) <= threshold)[0]


def source_clusters(md: pd.DataFrame, groups, ks, seed: int):
    """KMeans on standardised log size, log aspect, log bytes per pixel and the file-name pattern; k by silhouette.

    Every source group gets ONE cluster (the majority of its images): resized copies of a photo differ in size, and
    leaving them in different clusters would leak copies across leave-one-cluster-out folds.
    Returns (cluster per row, table of k vs silhouette)."""
    Z = np.column_stack([md["log_pixels"], np.log(md["aspect"]), np.log10(md["bytes_per_pixel"])])
    Z = StandardScaler().fit_transform(Z)
    pats = pd.get_dummies(md["name_pattern"]).values.astype(float)
    if pats.shape[1] > 1:
        Z = np.column_stack([Z, pats])
    sel, best = [], None
    for k in ks:
        if k >= len(Z):
            continue
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        s = float(silhouette_score(Z, lab, sample_size=min(len(Z), 3000), random_state=seed))
        sel.append({"k": k, "silhouette": s})
        if best is None or s > best[0]:
            best = (s, lab)
    lab = pd.Series(best[1])
    g = pd.Series(np.asarray(groups))
    majority = lab.groupby(g).agg(lambda s: s.value_counts().index[0])
    return g.map(majority).values, pd.DataFrame(sel)
