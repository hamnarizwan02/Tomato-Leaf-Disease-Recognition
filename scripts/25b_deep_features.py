"""Stage 25b: deep-extractor arm (frozen pretrained networks as FEATURE EXTRACTORS ONLY; no training, no
fine-tuning). All classifiers trained on these features are classical ML (scripts/26, 27). Run in .venv-dl:

    TORCH_HOME=F:/NUST/ML/model_cache_dl/torch HF_HOME=F:/NUST/ML/model_cache_dl/hf \
    .venv-dl/Scripts/python scripts/25b_deep_features.py --run F:/NUST/ML/runs/p4_multisource

  - resnet_embed: torchvision ResNet-50 (IMAGENET1K_V2 weights), global-average-pooled 2,048-d, standard transform;
  - clip_embed: OpenCLIP ViT-B-32 ("openai" weights) image embedding, L2-normalised, 512-d;
  - clip_concepts: cosine similarity of clip_embed with the text embedding of each concept in configs/concepts.yaml
    (template "a photo of {}"), symptom concepts first, then context concepts.
Whole images (no leaf mask). Outputs (<run>/work/): features_resnet.npy, features_clip.npy, features_concepts.npy,
features_concepts.names.json, deep_meta.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, tf_a, tf_b):
        self.paths, self.tf_a, self.tf_b = paths, tf_a, tf_b

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img.thumbnail((1024, 1024))            # large field photos: cap the decode size before the transforms
        return self.tf_a(img), self.tf_b(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    run = Path(args.run)
    work = run / "work"
    work.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(open(ROOT / "configs" / "sources.yaml"))
    con = yaml.safe_load(open(ROOT / "configs" / "concepts.yaml"))
    df = pd.read_csv(run / "sources" / "manifest_sources.csv")
    paths = [str(Path(spec["data_root"]) / p) for p in df["path"]]

    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    resnet = torchvision.models.resnet50(weights=w).eval()
    resnet.fc = torch.nn.Identity()
    clip, _, clip_tf = open_clip.create_model_and_transforms(con["model"], pretrained=con["pretrained"])
    clip.eval()
    tok = open_clip.get_tokenizer(con["model"])
    names = [f"symptom_{k}" for k in con["symptom"]] + [f"context_{k}" for k in con["context"]]
    texts = [con["template"].format(v) for v in list(con["symptom"].values()) + list(con["context"].values())]
    with torch.no_grad():
        T = clip.encode_text(tok(texts)).float()
        T = T / T.norm(dim=-1, keepdim=True)

    dl = torch.utils.data.DataLoader(DS(paths, w.transforms(), clip_tf), batch_size=args.batch, num_workers=2)
    R, E = [], []
    t0 = time.time()
    with torch.no_grad():
        for k, (a, b) in enumerate(dl):
            R.append(resnet(a).numpy())
            e = clip.encode_image(b).float()
            E.append((e / e.norm(dim=-1, keepdim=True)).numpy())
            if k % 20 == 0:
                print(f"{(k + 1) * args.batch}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    R, E = np.concatenate(R).astype(np.float32), np.concatenate(E).astype(np.float32)
    Cs = E @ T.numpy().T
    np.save(work / "features_resnet.npy", R)
    np.save(work / "features_clip.npy", E)
    np.save(work / "features_concepts.npy", Cs.astype(np.float32))
    json.dump(names, open(work / "features_concepts.names.json", "w"), indent=1)
    json.dump({"texts": dict(zip(names, texts)), "clip": [con["model"], con["pretrained"]],
               "resnet": "torchvision resnet50 IMAGENET1K_V2", "torch": torch.__version__,
               "open_clip": open_clip.__version__, "n": len(paths),
               "manifest_paths_md5": hashlib.md5("|".join(df["path"]).encode()).hexdigest(),
               "seconds": round(time.time() - t0)}, open(work / "deep_meta.json", "w"), indent=1)
    print("done", R.shape, E.shape, Cs.shape)


if __name__ == "__main__":
    main()
