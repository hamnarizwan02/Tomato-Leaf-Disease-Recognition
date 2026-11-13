"""Stage 16 (Phase 2, experiment A): features of train-only augmented copies.

    TOMATO_CONFIG=configs/aug_geo.yaml python scripts/16_extract_augmented.py            # build (skips if up to date)
    TOMATO_CONFIG=configs/aug_geo.yaml python scripts/16_extract_augmented.py --force

`copies` augmented versions of EVERY manifest image are made once (src/augment.py) and stored with their source row;
scripts/13_protocol.py --augment then adds, per split, only the copies of that split's training images.
Output: work/features/augmented/<name>.npy + <name>_index.csv + <name>.meta.json; results/augmentation_<name>_gallery.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.augment import augment, aug_files, aug_key, extract_augmented, save_augmented  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import cache_file, load_cache, load_manifest  # noqa: E402
from src.utils import load_json, set_seed  # noqa: E402
from src.features import worker_cfg  # noqa: E402


def gallery(cfg, df, imgs, path, n_img=4):
    aug = cfg["augmentation"]
    k = max(int(aug["copies"]), 1)
    rows = np.random.default_rng(cfg["seed"]).choice(len(df), n_img, replace=False)
    fig, axes = plt.subplots(n_img, k + 1, figsize=(2.1 * (k + 1), 2.1 * n_img))
    for i, r in enumerate(rows):
        axes[i, 0].imshow(imgs[r])
        axes[i, 0].set_title(f"original ({df['label'][r][:12]})", fontsize=7)
        for c in range(k):
            axes[i, c + 1].imshow(augment(np.ascontiguousarray(imgs[r]), r, c, aug, cfg["seed"]) if aug["copies"] else imgs[r])
            axes[i, c + 1].set_title(f"copy {c}", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"augmentation '{aug['name']}': {aug}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.get("augmentation"):
        raise SystemExit("no augmentation block in the config: set TOMATO_CONFIG=configs/aug_<name>.yaml")
    set_seed(cfg["seed"])
    df = load_manifest(cfg)
    imgs = load_cache(cfg, df)
    aug = cfg["augmentation"]
    gallery(cfg, df, imgs, cfg["paths"]["results_dir"] / f"augmentation_{aug['name']}_gallery.png")

    fx, fi, fm = aug_files(cfg)
    if fm.exists() and load_json(fm).get("key") == aug_key(cfg, df) and fx.exists() and fi.exists() and not args.force:
        print(f"{fx.name}: up to date")
        return
    t0 = time.time()
    X, index = extract_augmented(cache_file(cfg), np.arange(len(df)), aug, worker_cfg(cfg), cfg["n_jobs"])
    save_augmented(cfg, df, X, index)
    print(f"augmentation '{aug['name']}': {X.shape} ({aug['copies']} copies x {len(df)} images) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
