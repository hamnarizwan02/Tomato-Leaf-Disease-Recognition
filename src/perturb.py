"""Controlled test-time perturbations.  (Owner: Member A)

perturb(img, kind, level, idx, sub) -> uint8 RGB image, deterministic given (seed, image idx, kind, level).
core kinds     : brightness | contrast | shadow | background
extended kinds : gamma | color_cast | blur | motion_blur | jpeg | noise | downup | occlusion | rotate | scale
combo          : level "kind@level+kind@level" applies the parts left to right
Background replacement uses the cached reference mask (sub["ref_mask"]) when there is one, so it does not depend on
the mask the features use (problem 23).
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from .segment import segment_leaf

# value of each numeric kind that leaves the image unchanged; severity = |level - identity|
IDENTITY = {"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "color_cast": 0.0, "shadow": 0.0, "blur": 0.0,
            "motion_blur": 0.0, "jpeg": 100.0, "noise": 0.0, "downup": 1.0, "occlusion": 0.0, "rotate": 0.0,
            "scale": 1.0}


def condition_name(kind: str, level) -> str:
    return f"{kind}_{level}"


def _rng(seed: int, idx: int, kind: str, level) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(idx), zlib.crc32(condition_name(kind, level).encode())])


def _shadow_map(shape, strength: float, rng) -> np.ndarray:
    h, w = shape
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = ((xx - w / 2) * np.cos(theta) + (yy - h / 2) * np.sin(theta)) / (max(h, w) / 2)
    t = 1.0 / (1.0 + np.exp(-(proj - rng.uniform(-0.3, 0.3)) / 0.15))
    return 1.0 - strength * t


def _lowfreq(h, w, scale, rng) -> np.ndarray:
    small = rng.random((scale, scale, 3), dtype=np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _synthetic_background(kind: str, shape, rng) -> np.ndarray:
    h, w = shape
    if kind == "gray":
        return np.full((h, w, 3), 128, np.float32)
    if kind == "noise":
        return _lowfreq(h, w, 6, rng) * 255.0
    base = {"soil": (110, 80, 55), "grass": (60, 115, 40)}[kind]
    bg = np.array(base, np.float32) + (_lowfreq(h, w, 10, rng) - 0.5) * 60
    bg += (rng.random((h, w, 1), dtype=np.float32) - 0.5) * 30
    return bg


_REF_MASKS: dict = {}


def _ref_mask(path: str, idx: int, shape) -> np.ndarray:
    if path not in _REF_MASKS:
        _REF_MASKS[path] = np.load(path, mmap_mode="r")
    h, w = shape
    return np.unpackbits(_REF_MASKS[path][idx])[: h * w].reshape(h, w).astype(bool)


def _replace_background(img: np.ndarray, kind: str, idx: int, rng, sub: dict) -> np.ndarray:
    if sub.get("ref_mask"):
        mask = _ref_mask(sub["ref_mask"], idx, img.shape[:2])
    else:
        mask = segment_leaf(img, sub["segmentation"])
    if mask.all():                      # segmentation failed -> nothing to replace
        return img.astype(np.float32)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    alpha = cv2.GaussianBlur(mask, (0, 0), 2)[..., None]
    bg = _synthetic_background(kind, img.shape[:2], rng)
    return alpha * img.astype(np.float32) + (1 - alpha) * bg


def _motion_kernel(length: int, rng) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    return k / max(k.sum(), 1e-8)


def _affine(img: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _occlude(x: np.ndarray, frac: float, rng) -> np.ndarray:
    h, w = x.shape[:2]
    aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    rh = int(min(h, round(np.sqrt(frac * h * w / aspect))))
    rw = int(min(w, round(frac * h * w / max(rh, 1))))
    y0, x0 = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
    x = x.copy()
    x[y0:y0 + rh, x0:x0 + rw] = 128.0
    return x


def _parse_level(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def parse_combo(level: str) -> list[tuple[str, object]]:
    return [(k, _parse_level(v)) for k, v in (part.split("@") for part in str(level).split("+"))]


def _apply(img: np.ndarray, kind: str, level, idx: int, sub: dict) -> np.ndarray:
    rng = _rng(sub["seed"], idx, kind, level)
    x = img.astype(np.float32)
    if kind == "brightness":
        x = x * float(level)
    elif kind == "contrast":
        m = x.mean()
        x = (x - m) * float(level) + m
    elif kind == "shadow":
        x = x * _shadow_map(x.shape[:2], float(level), rng)[..., None]
    elif kind == "background":
        x = _replace_background(img, str(level), idx, rng, sub)
    elif kind == "gamma":
        x = 255.0 * (x / 255.0) ** float(level)
    elif kind == "color_cast":
        s = float(level)
        x = x * np.array([1 + s, 1.0, 1 - s], np.float32)
    elif kind == "blur":
        x = cv2.GaussianBlur(x, (0, 0), float(level))
    elif kind == "motion_blur":
        x = cv2.filter2D(x, -1, _motion_kernel(int(level), rng), borderType=cv2.BORDER_REFLECT_101)
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        x = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    elif kind == "noise":
        x = x + rng.normal(0.0, float(level), x.shape).astype(np.float32)
    elif kind == "downup":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, round(w * float(level))), max(1, round(h * float(level)))),
                           interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    elif kind == "occlusion":
        x = _occlude(x, float(level), rng)
    elif kind == "rotate":
        x = _affine(x, float(level) * rng.choice([-1, 1]), 1.0)
    elif kind == "scale":
        x = _affine(x, 0.0, float(level))
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return np.clip(x, 0, 255).astype(np.uint8)


def perturb(img: np.ndarray, kind: str, level, idx: int, sub: dict) -> np.ndarray:
    if kind == "combo":
        for k, lv in parse_combo(level):
            img = _apply(img, k, lv, idx, sub)
        return img
    return _apply(img, kind, level, idx, sub)


def all_conditions(cfg, extended: bool = False) -> list[tuple[str, object]]:
    """Core conditions; with extended=True also perturbations_extended and the combos."""
    conds = [(kind, level) for kind, levels in cfg["perturbations"].items() for level in levels]
    if extended:
        conds += [(kind, level) for kind, levels in (cfg.get("perturbations_extended") or {}).items()
                  for level in levels]
        conds += [("combo", c) for c in cfg.get("perturbation_combos") or []]
    return conds
