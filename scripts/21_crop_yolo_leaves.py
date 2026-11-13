"""Stage 21: turn a YOLO leaf-box dataset into one image per leaf, for use as an external test set (scripts/20).

    python scripts/21_crop_yolo_leaves.py --images <dir> --labels <dir> --out <dir> --names 0=healthy 1=diseased

Every box becomes <out>/<class name>/<image stem>_b<k>.png (lossless, so no extra JPEG compression), enlarged by
--margin of its size on each side and clipped to the image. Images without a label file are skipped and counted.
Class names come from --names, which must be checked against the images by eye (YOLO files carry only numbers).
Writes <out>/crops.csv (crop, source image, class id, box).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--labels", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", required=True, nargs="+", help="id=name pairs")
    ap.add_argument("--margin", type=float, default=0.1)
    args = ap.parse_args()
    names = dict(n.split("=", 1) for n in args.names)
    out = Path(args.out)
    rows, skipped, n_img = [], 0, 0
    for img_dir, lab_dir in zip(args.images, args.labels):
        for p in sorted(Path(img_dir).iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            lf = Path(lab_dir) / (p.stem + ".txt")
            if not lf.exists():
                skipped += 1
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            n_img += 1
            for k, line in enumerate(l for l in lf.read_text().splitlines() if l.strip()):
                cid, cx, cy, w, h = line.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                w2, h2 = w * (0.5 + args.margin), h * (0.5 + args.margin)
                x0, x1 = int(max(0, cx - w2)), int(min(W, cx + w2))
                y0, y1 = int(max(0, cy - h2)), int(min(H, cy + h2))
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                d = out / names[cid]
                d.mkdir(parents=True, exist_ok=True)
                dst = d / f"{p.stem}_b{k}.png"
                cv2.imencode(".png", img[y0:y1, x0:x1])[1].tofile(str(dst))
                rows.append({"crop": dst.relative_to(out).as_posix(), "source": str(p), "class_id": cid,
                             "box_px": f"{x0},{y0},{x1},{y1}"})
    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["crop", "source", "class_id", "box_px"])
        w.writeheader()
        w.writerows(rows)
    print(f"{n_img} images, {len(rows)} crops, {skipped} images without labels skipped -> {out}")


if __name__ == "__main__":
    main()
