"""Illumination normalisation (Phase 2, experiment B).

normalize(img, ill) -> uint8 RGB image. `ill` is config["preprocess"]["illumination"] (set in configs/illum_*.yaml,
absent from config.yaml = no normalisation):
    {"method": "grayworld"}                                  scale R, G, B so their means equal the grey mean
    {"method": "clahe", "clip_limit": 2.0, "tile_grid": 8}   CLAHE on the L channel of LAB (a, b untouched)
Each image is normalised on its own: no parameter is estimated from other images, labels or results.
Applied after any test-time perturbation and before segmentation and features (src/features.py), so it acts like a
preprocessing step of a deployed pipeline. Gray World assumes the scene averages to grey; leaf close-ups are mostly
green, so it shifts their colours, which is part of what experiment B measures.
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize(img: np.ndarray, ill: dict | None) -> np.ndarray:
    if not ill:
        return img
    method = ill["method"]
    if method == "grayworld":
        x = img.astype(np.float32)
        means = x.reshape(-1, 3).mean(0)
        x *= means.mean() / np.maximum(means, 1e-6)
        return np.clip(x, 0, 255).astype(np.uint8)
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        t = int(ill.get("tile_grid", 8))
        lab[..., 0] = cv2.createCLAHE(clipLimit=float(ill.get("clip_limit", 2.0)), tileGridSize=(t, t)).apply(
            np.ascontiguousarray(lab[..., 0]))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    raise ValueError(f"unknown illumination method: {method}")
