"""Figures for the project status report. Reads only saved run outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("F:/NUST/ML/runs")
OUT = Path("F:/NUST/ML/report/fig")
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF})


def bars(ax, groups, series, colors, values, errs, labels):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, c) in enumerate(zip(series, colors)):
        v, e = np.array(values[i], float), errs[i]
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=c, label=s, edgecolor=SURF, linewidth=1)
        if e is not None:
            ax.errorbar(xs, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for xx, vv in zip(xs, v):
            ax.text(xx, vv + 0.02 + (0.04 if e is not None else 0), f"{vv:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)


# ---- Figure 1: supplied set, split rule
g = pd.read_csv(RUNS / "p2_supplied_grouped/results/protocol_grouped/summary_clean.csv")
r = pd.read_csv(RUNS / "p2_supplied_grouped/results/protocol_random/summary_clean.csv")
s = pd.read_csv(RUNS / "p2_supplied_repro/results/baseline_test_metrics.csv").set_index("model")
models = ["rf", "svm", "lr"]
get = lambda d, m, c: float(d[(d.feature_set == "all") & (d.model == m)][c].iloc[0])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars(ax, ["Random forest", "SVM", "Logistic regression"],
     ["Image-level split", "Supplied split", "Grouped split (Task 2)"], [C1, C2, C3],
     [[get(r, m, "macro_f1_mean") for m in models], [s.loc[m, "test_macro_f1"] for m in models],
      [get(g, m, "macro_f1_mean") for m in models]],
     [[get(r, m, "macro_f1_std") for m in models], None, [get(g, m, "macro_f1_std") for m in models]], None)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Test macro-F1")
fig.tight_layout()
fig.savefig(OUT / "fig1_supplied_split_rule.png", dpi=220)

# ---- Figure 2: cross-dataset
cd = RUNS / "p3_cross_dataset/results/cross_dataset"
p = pd.read_csv(cd / "plantdoc/summary.csv")
nz = pd.read_csv(cd / "nazirpur/summary.csv")
pick = lambda d, sub, m, c: float(d[(d.subset == sub) & (d.model == m)][c].iloc[0])
vals, errs = [], []
for m in models:
    vals.append([pick(p, "own_test", m, "macro_f1_present_mean"), pick(p, "external_independent", m, "macro_f1_present_mean"),
                 pick(nz, "external_independent", m, "macro_f1_present_mean")])
    errs.append([pick(p, "own_test", m, "macro_f1_present_std"), pick(p, "external_independent", m, "macro_f1_present_std"),
                 pick(nz, "external_independent", m, "macro_f1_present_std")])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars(ax, ["Our test split\n(Pakistan raw set)", "PlantDoc\n(310 new images)", "Nazirpur, Bangladesh\n(septoria + healthy)"],
     ["Random forest", "SVM", "Logistic regression"], [C1, C2, C3], vals, errs, None)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Macro-F1 (classes present)")
fig.tight_layout()
fig.savefig(OUT / "fig2_cross_dataset.png", dpi=220)

# ---- Figure 3: feature-level effects across datasets (model-free)
E = pd.read_csv(RUNS / "p3_cross_dataset/results/feature_explain/healthy_vs_diseased.csv", index_col=0)
dsets = ["raw", "plantdoc", "nazirpur", "brahmanbaria"]
cons = E[E.consistent].index.tolist()
top_raw = E.reindex(E["raw"].abs().sort_values(ascending=False).index).index[:8].tolist()
rows = cons + [f for f in top_raw if f not in cons]
M = E.loc[rows, dsets].values
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
im = ax.imshow(M, cmap=cmap, vmin=-0.35, vmax=0.35, aspect="auto")
ax.set_xticks(range(4), ["Our raw set", "PlantDoc", "Nazirpur", "Brahmanbaria"])
pretty = lambda f: f.replace("color_", "colour: ").replace("texture_", "texture: ").replace("shape_", "shape: ")
ax.set_yticks(range(len(rows)), [pretty(f) for f in rows], fontsize=8)
ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(M[i, j]) > 0.22 else INK)
ax.axhline(len(cons) - 0.5, color=INK, lw=1.2)
ax.text(3.62, 0.5 * (len(cons) - 1), "consistent\nin all four", va="center", fontsize=7.5, color=INK2)
ax.text(3.62, len(cons) + 0.5 * (len(rows) - len(cons) - 1), "strongest in\nour raw set", va="center", fontsize=7.5,
        color=INK2)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.2)
cb.set_label("Effect (AUC − 0.5): + higher in diseased", fontsize=7.5, color=INK2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig3_feature_effects.png", dpi=220)
print("ok", rows)
