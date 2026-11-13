"""Stage 29: Phase 3 figures (multi-source LOSO and feature-level explanation).

    python scripts/29_figures_phase3.py --run F:/NUST/ML/runs/p4_multisource

Reads results/phase3_summary/*.csv (stage 28) and results/explain/*.csv (stage 27); writes PNG (200 dpi) to
results/figures/. Every figure is skipped quietly if its inputs do not exist yet.
Colours: arms use the validated categorical order blue/orange/aqua/yellow (light surface); bars carry direct value
labels because aqua and yellow are below 3:1 contrast. Effects use a diverging blue<->red map with a grey midpoint.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import lab2rgb

ARM_ORDER = ["formula", "kmeans", "formula+kmeans", "deep"]
ARM_COL = {"formula": "#2a78d6", "kmeans": "#eb6834", "formula+kmeans": "#1baf7a", "deep": "#eda100"}
ARM_LABEL = {"formula": "Formula", "kmeans": "K-means words", "formula+kmeans": "Formula + k-means",
             "deep": "Frozen deep"}
TASK_LABEL = {"bin": "Healthy vs diseased (balanced accuracy)", "six": "Six classes (macro-F1, classes present)"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#d9d8d4", "#f0a090", "#c0392b"])
SRC_LABEL = {"pk_raw": "Raw (PK only)", "pd_raw": "Raw (PlantDoc-dup.)", "plantdoc": "PlantDoc",
             "plantwild": "PlantWild", "tomato_village": "Tomato-Village", "nazirpur": "Nazirpur",
             "brahmanbaria": "Brahmanbaria", "plantvillage": "PlantVillage"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def arms_bar(ov, out):
    """Per task: for each feature set, the best classifier's unseen-source mean (bar) and seen-source mean (tick)."""
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 8.4), sharex=False)    # column-width layout
    for ax, task in zip(axes, ["bin", "six"]):
        t = ov[ov.task == task]
        if t.empty:
            ax.set_visible(False)
            continue
        b = t.loc[t.groupby("set")["loso_mean"].idxmax()].copy()
        b["arm_i"] = b.arm.map(ARM_ORDER.index)
        b = b.sort_values(["arm_i", "loso_mean"], ascending=[False, True]).reset_index(drop=True)
        y = np.arange(len(b))
        ax.barh(y, b.loso_mean, height=0.62, color=b.arm.map(ARM_COL), edgecolor="white", linewidth=2)
        ax.scatter(b.seen_mean, y, marker="|", s=160, color=INK, linewidths=2, zorder=3, label="Seen source")
        for i, r in b.iterrows():
            ax.text(r.loso_mean - 0.012, i, f"{r.loso_mean:.2f}", va="center", ha="right", fontsize=7.5,
                    color=INK)                       # inside the bar end: never confused with the seen tick
        ax.set_yticks(y, [f"{s}  ({m.upper()})" for s, m in zip(b.set, b.model)])
        chance = 0.5 if task == "bin" else None
        if chance:
            ax.axvline(chance, color=MUTED, lw=1, ls="--")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(TASK_LABEL[task], fontsize=9, loc="left")
        ax.set_xlabel("Mean over held-out sources")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COL[a]) for a in ARM_ORDER if a in set(ov.arm)]
    labels = [ARM_LABEL[a] for a in ARM_ORDER if a in set(ov.arm)]
    handles.append(plt.Line2D([], [], marker="|", ls="", color=INK, markersize=10, markeredgewidth=2))
    labels.append("Seen-source score (same model)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "p3_arms_loso_vs_seen.png", bbox_inches="tight")
    plt.close(fig)


def source_heatmap(per, ov, out):
    """Best set of each arm (by LOSO mean) x held-out source."""
    for task in ["bin", "six"]:
        o = ov[ov.task == task]
        if o.empty:
            continue
        best = o.loc[o.groupby("arm")["loso_mean"].idxmax()]
        rows, labels = [], []
        for a in ARM_ORDER:
            r = best[best.arm == a]
            if r.empty:
                continue
            r = r.iloc[0]
            p = per[(per.task == task) & (per.set == r.set) & (per.model == r.model)].set_index("source")["primary"]
            rows.append(p)
            labels.append(f"{ARM_LABEL[a]}\n{r.set}\n({r.model.upper()})")
        M = pd.concat(rows, axis=1)                                   # rows = sources, columns = arms
        M = M.loc[[c for c in SRC_LABEL if c in M.index]]
        fig, ax = plt.subplots(figsize=(1.15 * M.shape[1] + 2.2, 0.42 * M.shape[0] + 1.5))
        lo = 0.4 if task == "bin" else 0.0
        im = ax.imshow(M.values, cmap=SEQ, vmin=lo, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v > lo + 0.45 * (1 - lo) else INK)
        ax.set_yticks(range(M.shape[0]), [SRC_LABEL[c] for c in M.index], fontsize=7.5)
        ax.set_xticks(range(M.shape[1]), labels, fontsize=6.5)
        ax.xaxis.tick_top()
        ax.spines[:].set_visible(False)
        ax.set_xlabel(f"Held-out source: {TASK_LABEL[task]}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(out / f"p3_heatmap_{task}.png", bbox_inches="tight")
        plt.close(fig)


def stable_strip(E, st, out, top=16):
    """Per-source effects (AUC - 0.5) for the most source-stable features of each pool."""
    st = st.sort_values(["frac_sig_majority", "median_abs_eff"], ascending=False)
    feats = st.groupby("pool").head(top // max(1, st.pool.nunique()) + 2).head(top)
    feats = feats.sort_values("frac_sig_majority").feature.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(feats) + 1.2))
    for i, f in enumerate(feats):
        g = E[E.feature == f]
        ax.hlines(i, g.eff.min(), g.eff.max(), color=GRID, lw=3, zorder=1)
        sig = (g.lo > 0) | (g.hi < 0)
        ax.scatter(g.eff[sig], [i] * sig.sum(), s=22, c=g.eff[sig], cmap=DIV, vmin=-0.35, vmax=0.35,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(g.eff[~sig], [i] * (~sig).sum(), s=22, facecolors="white", edgecolors=MUTED, linewidths=0.8,
                   zorder=2)
        r = st[st.feature == f].iloc[0]
        ax.text(0.47, i, f"{r.n_sig_majority}/{r.n_sources}", va="center", fontsize=7, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(feats)), [f"{f}  [{st[st.feature == f].pool.iloc[0]}]" for f in feats], fontsize=7)
    ax.set_xlim(-0.5, 0.55)
    ax.set_xlabel("Effect = AUC − 0.5 (diseased vs healthy), one dot per source; hollow = CI includes 0")
    ax.set_title("Most source-stable features (right: sources significant with the majority sign)", fontsize=9,
                 loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "p3_stable_features.png", bbox_inches="tight")
    plt.close(fig)


def colour_words(cw, st, out):
    """The 24 global colour words as swatches, annotated with their pooled effect direction."""
    lab = cw[["L", "a", "b"]].values.astype(float)        # OpenCV 8-bit CIELAB: L*255/100, a+128, b+128
    lab = np.c_[lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128]
    rgb = np.clip(lab2rgb(lab[None])[0], 0, 1)
    eff = st.set_index("feature")["median_abs_eff"] * st.set_index("feature")["majority_sign"]
    frac = st.set_index("feature")["frac_sig_majority"]
    order = np.argsort([eff.get(f, 0) for f in cw.feature])
    per_row = 12
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for k, i in enumerate(order):
        f = cw.feature.iloc[i]
        x, y0 = k % per_row, -(k // per_row) * 2.0
        ax.add_patch(plt.Rectangle((x, y0), 0.9, 1, facecolor=rgb[i], edgecolor=GRID, lw=0.6))
        e = eff.get(f, np.nan)
        ax.text(x + 0.45, y0 - 0.12, f"{e:+.2f}" if np.isfinite(e) else "", ha="center", va="top", fontsize=6.5)
        ax.text(x + 0.45, y0 - 0.5, f"{frac.get(f, np.nan):.0%}" if f in frac else "", ha="center", va="top",
                fontsize=6, color=MUTED)
    ax.set_xlim(-0.1, per_row)
    ax.set_ylim(-2.95, 1.1)
    ax.axis("off")
    ax.set_title("K-means colour words, ordered by signed median effect\n(below: effect; share of sources "
                 "significant)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out / "p3_colour_words.png", bbox_inches="tight")
    plt.close(fig)


def selection_bar(S, out):
    """Explanation-guided selection: all features vs source-stable subset, LOSO balanced accuracy."""
    g = S.groupby(["pool", "features", "model"])["balanced_accuracy"].mean().reset_index()
    g = g[g.model == ("lr" if "lr" in set(g.model) else g.model.iloc[0])]
    pools = list(dict.fromkeys(g.pool))
    feats = list(dict.fromkeys(g.features))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.8 / len(feats)
    for k, f in enumerate(feats):
        v = [g[(g.pool == p) & (g.features == f)].balanced_accuracy.mean() for p in pools]
        x = np.arange(len(pools)) + k * w
        ax.bar(x, v, width=w, color=cols[k % 4], edgecolor="white", linewidth=2, label=f)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(len(pools)) + w * (len(feats) - 1) / 2, pools)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy, held-out source (mean)")
    ax.set_title(f"Explanation-guided feature selection ({g.model.iloc[0].upper()})", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=len(feats), loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "p3_selection.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    res = Path(a.run) / "results"
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    s3 = res / "phase3_summary"
    if (s3 / "overview.csv").exists():
        ov, per = pd.read_csv(s3 / "overview.csv"), pd.read_csv(s3 / "per_source.csv")
        arms_bar(ov, out)
        source_heatmap(per, ov, out)
    ex = res / "explain"
    if (ex / "stability.csv").exists():
        E, st = pd.read_csv(ex / "effects.csv"), pd.read_csv(ex / "stability.csv")
        stable_strip(E, st, out)
        if (ex / "colour_words_global.csv").exists():
            colour_words(pd.read_csv(ex / "colour_words_global.csv"), st, out)
    if (ex / "selection_metrics.csv").exists():
        selection_bar(pd.read_csv(ex / "selection_metrics.csv"), out)
    print("figures:", sorted(p.name for p in out.glob("p3_*.png")))


if __name__ == "__main__":
    main()
