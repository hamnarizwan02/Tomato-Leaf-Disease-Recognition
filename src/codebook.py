"""Learned features with classical unsupervised ML (k-means codebooks / bag of visual words). No deep learning.

Per image (256 px cache, HSV leaf mask), a fixed seeded sample of leaf pixels is stored:
  - colour descriptors: CIELAB (L, a, b) of N_PIX leaf pixels;
  - texture descriptors (8-d): on the L channel divided by the leaf's own L standard deviation (contrast
    normalisation), |Sobel x| and |Sobel y| after Gaussian blur sigma 1 and 3, Laplacian of Gaussian sigma 1 and 3,
    and local standard deviation in 5x5 and 9x9 windows.
A codebook is learned with MiniBatchKMeans on the descriptors of a set of FIT images (in LOSO: the training sources
only), and every image becomes a histogram: the share of its sampled leaf pixels nearest to each centroid
("colour words", "textons"). Each colour word is a colour that can be shown as a swatch.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

N_PIX = 1500
K_COLOUR, K_TEXTON = 24, 32


def descriptors(img: np.ndarray, mask: np.ndarray, seed: int):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    leaf = mask.astype(bool)
    if leaf.sum() < 50:
        leaf = np.ones_like(leaf)
    Ln = L / (L[leaf].std() + 1.0)
    feats = []
    for s in (1.0, 3.0):
        g = cv2.GaussianBlur(Ln, (0, 0), s)
        feats += [np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)),
                  cv2.Laplacian(g, cv2.CV_32F, ksize=3)]
    for k in (5, 9):
        m = cv2.blur(Ln, (k, k))
        feats.append(np.sqrt(np.maximum(cv2.blur(Ln * Ln, (k, k)) - m * m, 0)))
    T = np.stack(feats, -1)
    ys, xs = np.nonzero(leaf)
    idx = np.random.default_rng(seed).choice(len(ys), N_PIX, replace=len(ys) < N_PIX)
    return lab[ys[idx], xs[idx]].astype(np.float16), T[ys[idx], xs[idx]].astype(np.float16)


def fit_codebook(samples: np.ndarray, fit_rows, k: int, seed: int = 42, per_image: int = 200) -> MiniBatchKMeans:
    """samples: (n_images, N_PIX, d). Fit on per_image random pixels of each fit image."""
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, samples.shape[1], (len(fit_rows), per_image))
    D = samples[np.asarray(fit_rows)[:, None], pick].reshape(-1, samples.shape[2]).astype(np.float32)
    mu, sd = D.mean(0), D.std(0) + 1e-6
    km = MiniBatchKMeans(k, random_state=seed, batch_size=4096, n_init=3, max_iter=200).fit((D - mu) / sd)
    km.mu_, km.sd_ = mu, sd
    return km


def histograms(km: MiniBatchKMeans, samples: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.zeros((len(samples), km.n_clusters), np.float32)
    for a in range(0, len(samples), chunk):
        S = samples[a:a + chunk].astype(np.float32)
        lab = km.predict(((S.reshape(-1, S.shape[2]) - km.mu_) / km.sd_)).reshape(len(S), -1)
        for i, row in enumerate(lab):
            out[a + i] = np.bincount(row, minlength=km.n_clusters) / len(row)
    return out


def centroids_raw(km: MiniBatchKMeans) -> np.ndarray:
    return km.cluster_centers_ * km.sd_ + km.mu_
