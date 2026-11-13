"""Leaf-relative lesion features (Phase 3, docs/phase3_design_2026-11-12.md). Definitions fixed before any Phase 3 run.

Every feature is measured inside the leaf mask of src/segment.py and, where possible, RELATIVE to the same leaf's own
green tissue, so a global brightness, contrast or white-balance change largely cancels out:
  - colour classes of leaf pixels (OpenCV HSV, H 0-179): green reference H 35-85 & S >= 40; yellow H 15-35 & S >= 60 &
    V >= 60; brown (H < 15 or H >= 165) & S >= 40 & V >= 40; dark V < 0.6 x median V of the green reference;
    pale S < 35 & V >= 0.8 x median V of the green reference; non-green = leaf minus green reference.
  - LAB differences (non-green mean minus green-reference mean), hue shift, chromaticity r = R/(R+G+B) etc.
  - spots: connected components of yellow | brown | dark | pale pixels (3x3 opening, >= 10 px): density, size,
    largest, circularity; share of non-green pixels within 8 px of the leaf margin.
"""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

EPS = 1e-6


def lesion_features(img: np.ndarray, mask: np.ndarray) -> OrderedDict:
    out: OrderedDict = OrderedDict()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = mask.astype(bool)
    n_leaf = float(leaf.sum()) + EPS

    green = leaf & (H >= 35) & (H <= 85) & (S >= 40)
    if green.sum() < 50:                       # almost no green tissue: use the greener half of the leaf
        f = img.astype(np.float32)
        s = f.sum(2) + 1.0
        exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / s
        thr = np.median(exg[leaf]) if leaf.any() else 0.0
        green = leaf & (exg >= thr)
    vref = float(np.median(V[green])) if green.any() else float(np.median(V[leaf]) if leaf.any() else 128.0)

    yellow = leaf & (H >= 15) & (H < 35) & (S >= 60) & (V >= 60)
    brown = leaf & ((H < 15) | (H >= 165)) & (S >= 40) & (V >= 40)
    dark = leaf & (V < 0.6 * vref)
    pale = leaf & (S < 35) & (V >= 0.8 * vref)
    nongreen = leaf & ~green
    for name, m in (("yellow", yellow), ("brown", brown), ("dark", dark), ("pale", pale), ("nongreen", nongreen)):
        out[f"lesion_{name}_frac"] = float(m.sum()) / n_leaf

    if nongreen.sum() >= 20 and green.sum() >= 20:
        d = lab[nongreen].mean(0) - lab[green].mean(0)
        hue_shift = float(np.median(H[nongreen]) - np.median(H[green]))
    else:
        d, hue_shift = np.zeros(3, np.float32), 0.0
    out["lesion_dL"], out["lesion_da"], out["lesion_db"] = float(d[0]), float(d[1]), float(d[2])
    out["lesion_hue_shift"] = hue_shift

    a, b = lab[..., 1][leaf] - 128.0, lab[..., 2][leaf] - 128.0
    g_chroma = float(np.hypot(lab[..., 1][green] - 128.0, lab[..., 2][green] - 128.0).mean()) if green.any() else 1.0
    out["lesion_ab_spread_rel"] = float(np.sqrt(a.var() + b.var()) / (g_chroma + 1.0)) if leaf.any() else 0.0

    f = img.astype(np.float32)[leaf]
    s = f.sum(1, keepdims=True) + 1.0
    ch = f / s
    for k, c in enumerate("rgb"):
        out[f"lesion_chroma_{c}_mean"] = float(ch[:, k].mean()) if len(ch) else 0.0
    out["lesion_chroma_r_std"] = float(ch[:, 0].std()) if len(ch) else 0.0
    out["lesion_chroma_g_std"] = float(ch[:, 1].std()) if len(ch) else 0.0

    spots = ((yellow | brown | dark | pale) & leaf).astype(np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 10]
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], float)
    out["lesion_spot_density"] = 1000.0 * len(keep) / n_leaf
    out["lesion_spot_mean_frac"] = float(areas.mean() / n_leaf) if len(areas) else 0.0
    out["lesion_spot_largest_frac"] = float(areas.max() / n_leaf) if len(areas) else 0.0
    circ = []
    for i in keep[:200]:
        m = (lab_img == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            per = cv2.arcLength(cnts[0], True)
            circ.append(4 * np.pi * stats[i, cv2.CC_STAT_AREA] / (per * per + EPS))
    out["lesion_spot_circularity"] = float(np.mean(circ)) if circ else 0.0

    edge = leaf & ~cv2.erode(leaf.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    out["lesion_margin_share"] = float((nongreen & edge).sum()) / (float(nongreen.sum()) + EPS)
    return out


def lesion_schema() -> list[str]:
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    return list(lesion_features(img, np.ones((256, 256), bool)).keys())
