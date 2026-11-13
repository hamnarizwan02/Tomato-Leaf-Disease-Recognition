"""Stage 27: feature-level explanation across sources (docs/phase3_design_2026-11-12.md, "Explanation").

    python scripts/27_explain_multisource.py --run F:/NUST/ML/runs/p4_multisource

Feature pools (classifiers are always classical ML):
  formula  = hand (105) + lesion (20);
  kmeans   = 24 colour words + 32 textons (classical unsupervised ML, src/codebook.py). For the descriptive effect
             analysis one codebook is learned on ALL images without labels (so a feature means the same in every
             source); in the LOSO parts it is re-learned in each fold from the training sources only;
  concepts = 24 CLIP concepts (deep-extractor arm, frozen; only if scripts/25b has been run).
 1. effects   : per source, healthy-vs-diseased effect = ROC AUC(diseased) - 0.5 with a stratified 1,000-resample
                bootstrap 95% CI. Source-stable: CI excludes 0 with the majority sign in >= 75% of the sources having
                >= 30 images per class. Also each disease vs healthy (sources with >= 20 of both): "signatures".
 2. glass-box : L1 logistic regression (C = 0.1, standardised) per LOSO fold (binary task) on concepts and on formula
                features; coefficients saved; sign stability across folds.
 3. selection : per LOSO fold, features chosen on the TRAINING sources only (|effect| >= 0.05 with one sign in
                >= 75% of eligible training sources); RF and L1-LR on the selection vs on all features of the pool;
                scored on the held-out source.
 4. RF permutation importance on the held-out source (formula pool), per fold, vs the feature's stability score.
Outputs: <run>/results/explain/ effects.csv, stability.csv, signatures.csv, glassbox_coef.csv, glassbox_metrics.csv,
selection_metrics.csv, selected_features.csv, rf_heldout_importance.csv, summary.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.codebook import K_COLOUR, K_TEXTON, centroids_raw, fit_codebook, histograms  # noqa: E402

SEED, B, MIN_N, STABLE_FRAC, SEL_EFFECT = 42, 1000, 30, 0.75, 0.05
CLASSES = ["early_blight", "late_blight", "septoria_leaf_spot", "leaf_mold", "yellow_leaf_curl_virus", "healthy"]


def auc_matrix(X, y):
    """AUC of every column for y == 1 (Mann-Whitney via ranks; ties averaged)."""
    r = rankdata(X, axis=0)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_effects(X, y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    pt = auc_matrix(X, y) - 0.5
    bs = np.empty((B, X.shape[1]), np.float32)
    yy = np.r_[np.ones(len(i1), int), np.zeros(len(i0), int)]
    for b in range(B):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        bs[b] = auc_matrix(X[idx], yy) - 0.5
    return pt, np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def stability(E: pd.DataFrame, sources):
    """E: rows = (source, feature) with eff, lo, hi. Returns per-feature stability table."""
    rows = []
    for f, g in E.groupby("feature"):
        g = g[g.source.isin(sources)]
        sign = np.sign(np.sign(g.eff).sum()) or 1.0
        sig = ((g.lo > 0) & (sign > 0)) | ((g.hi < 0) & (sign < 0))
        rows.append({"feature": f, "pool": g.pool.iloc[0], "majority_sign": int(sign), "n_sources": len(g),
                     "n_sig_majority": int(sig.sum()), "frac_sig_majority": float(sig.mean()),
                     "n_sign_agree": int((np.sign(g.eff) == sign).sum()),
                     "median_abs_eff": float(g.eff.abs().median()), "min_abs_eff": float(g.eff.abs().min()),
                     "stable": bool(sig.mean() >= STABLE_FRAC)})
    return pd.DataFrame(rows).sort_values(["frac_sig_majority", "median_abs_eff"], ascending=False)


def select_on_training(Xp, y, src, train_sources, min_n=MIN_N):
    """Feature indices with |effect| >= SEL_EFFECT and the same sign in >= 75% of eligible training sources."""
    effs = []
    for s in train_sources:
        m = src == s
        if (y[m] == 1).sum() >= min_n and (y[m] == 0).sum() >= min_n:
            effs.append(auc_matrix(Xp[m], y[m]) - 0.5)
    E = np.array(effs)
    if len(E) == 0:
        return np.arange(Xp.shape[1])
    sign = np.sign(np.sign(E).sum(0))
    ok = ((np.sign(E) == sign) & (np.abs(E) >= SEL_EFFECT)).mean(0) >= STABLE_FRAC
    return np.where(ok & (sign != 0))[0]


def l1lr():
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                                              class_weight="balanced", random_state=SEED, max_iter=5000))


def rf():
    return RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def evaluate(model, X, y):
    pred = model.predict(X)
    P = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return {"balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "auc": roc_auc_score(y, P) if len(set(y)) == 2 else np.nan,
            "healthy_recall": float((pred[y == 0] == 0).mean()), "disease_recall": float((pred[y == 1] == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run, work = Path(args.run), Path(args.run) / "work"
    out = run / "results" / "explain"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(run / "sources" / "manifest_sources.csv")
    names = {"formula": json.load(open(work / "features_hand.names.json")) +
             json.load(open(work / "features_lesion.names.json"))}
    X = {"formula": np.hstack([np.load(work / "features_hand.npy"), np.load(work / "features_lesion.npy")])}
    desc = {"colour": np.load(work / "desc_colour.npy", mmap_mode="r"),
            "texture": np.load(work / "desc_texture.npy", mmap_mode="r")}

    def kmeans_feats(fit_rows):
        kc, kt = fit_codebook(desc["colour"], fit_rows, K_COLOUR), fit_codebook(desc["texture"], fit_rows, K_TEXTON)
        return np.hstack([histograms(kc, desc["colour"]), histograms(kt, desc["texture"])]), kc, kt
    X["kmeans"], kc_all, kt_all = kmeans_feats(np.arange(len(df)))
    names["kmeans"] = [f"colour_word_{i:02d}" for i in range(K_COLOUR)] + [f"texton_{i:02d}" for i in range(K_TEXTON)]
    cl = centroids_raw(kc_all)                          # CIELAB (OpenCV scale) of each colour word, for swatches
    pd.DataFrame({"feature": names["kmeans"][:K_COLOUR], "L": cl[:, 0], "a": cl[:, 1], "b": cl[:, 2]}).to_csv(
        out / "colour_words_global.csv", index=False)
    pd.DataFrame(centroids_raw(kt_all), columns=["sobel_x_s1", "sobel_y_s1", "log_s1", "sobel_x_s3", "sobel_y_s3",
                                                 "log_s3", "lstd_5", "lstd_9"]).assign(
        feature=names["kmeans"][K_COLOUR:]).to_csv(out / "textons_global.csv", index=False)
    if (work / "features_concepts.npy").exists():
        X["concepts"] = np.load(work / "features_concepts.npy")
        names["concepts"] = json.load(open(work / "features_concepts.names.json"))
    pools = list(X)
    y, src = df["y_bin"].values, df["source"].values
    sources = sorted(df.source.unique())
    rng = np.random.default_rng(SEED)

    # 1. effects with bootstrap CIs
    rows = []
    for pool in pools:
        for s in sources:
            m = src == s
            if (y[m] == 1).sum() < 5 or (y[m] == 0).sum() < 5:
                continue
            pt, lo, hi = boot_effects(X[pool][m], y[m], rng)
            for j, f in enumerate(names[pool]):
                rows.append({"pool": pool, "source": s, "feature": f, "n_healthy": int((y[m] == 0).sum()),
                             "n_diseased": int((y[m] == 1).sum()), "eff": pt[j], "lo": lo[j], "hi": hi[j]})
    E = pd.DataFrame(rows)
    E.to_csv(out / "effects.csv", index=False)
    eligible = [s for s in sources if (y[src == s] == 1).sum() >= MIN_N and (y[src == s] == 0).sum() >= MIN_N]
    st = stability(E, eligible)
    st.to_csv(out / "stability.csv", index=False)
    print("eligible sources:", eligible)
    print(st.groupby("pool")["stable"].agg(["sum", "size"]))

    # disease signatures: each disease vs healthy, formula + concepts, point effects per source
    sig = []
    lab6 = df["label6"].values
    for pool in pools:
        for d in CLASSES[:-1]:
            for s in sources:
                m = (src == s) & np.isin(lab6, [d, "healthy"])
                yy = (lab6[m] == d).astype(int)
                if yy.sum() >= 20 and (1 - yy).sum() >= 20:
                    e = auc_matrix(X[pool][m], yy) - 0.5
                    for j, f in enumerate(names[pool]):
                        sig.append({"pool": pool, "disease": d, "source": s, "feature": f, "eff": e[j],
                                    "n_disease": int(yy.sum()), "n_healthy": int((1 - yy).sum())})
    pd.DataFrame(sig).to_csv(out / "signatures.csv", index=False)

    # 2-4. LOSO folds (binary task)
    coef, gm, sm, sel_rows, imp_rows = [], [], [], [], []
    stab_score = dict(zip(st.feature, st.frac_sig_majority))
    for hold in sources:
        te, tr = src == hold, src != hold
        train_sources = [s for s in sources if s != hold]
        Xf = dict(X)
        Xf["kmeans"] = kmeans_feats(np.where(tr)[0])[0]            # codebook from the training sources only
        fold_pools = pools + ["formula+kmeans"] + (["all"] if "concepts" in X else [])
        for pool in fold_pools:
            parts = {"formula+kmeans": ["formula", "kmeans"], "all": ["formula", "kmeans", "concepts"]}.get(pool,
                                                                                                            [pool])
            Xp = np.hstack([Xf[q] for q in parts])
            nm = sum((names[q] for q in parts), [])
            sel = select_on_training(Xp[tr], y[tr], src[tr], train_sources)
            for f in sel:
                sel_rows.append({"heldout": hold, "pool": pool, "feature": nm[f]})
            for label, cols in (("all", np.arange(Xp.shape[1])), ("selected", sel)):
                if len(cols) == 0:
                    continue
                for mname, mk in (("l1lr", l1lr), ("rf", rf)):
                    mdl = mk().fit(Xp[tr][:, cols], y[tr])
                    sm.append({"heldout": hold, "pool": pool, "features": label, "n_features": len(cols),
                               "model": mname, **evaluate(mdl, Xp[te][:, cols], y[te])})
                    if mname == "l1lr" and label == "all" and pool in pools:
                        w = mdl[-1].coef_.ravel()
                        for j, f in enumerate(nm):
                            coef.append({"heldout": hold, "pool": pool, "feature": f, "coef": w[j]})
                        gm.append({"heldout": hold, "pool": pool, "n_nonzero": int((w != 0).sum()),
                                   **evaluate(mdl, Xp[te][:, cols], y[te])})
                    if mname == "rf" and label == "all" and pool == "formula":
                        pi = permutation_importance(mdl, Xp[te], y[te], scoring="balanced_accuracy", n_repeats=3,
                                                    random_state=SEED, n_jobs=1)
                        for j, f in enumerate(nm):
                            imp_rows.append({"heldout": hold, "feature": f, "importance": pi.importances_mean[j],
                                             "stability": stab_score.get(f, np.nan)})
        print(f"fold {hold} done", flush=True)
    C = pd.DataFrame(coef)
    C.to_csv(out / "glassbox_coef.csv", index=False)
    pd.DataFrame(gm).to_csv(out / "glassbox_metrics.csv", index=False)
    S = pd.DataFrame(sm)
    S.to_csv(out / "selection_metrics.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out / "selected_features.csv", index=False)
    I = pd.DataFrame(imp_rows)
    I.to_csv(out / "rf_heldout_importance.csv", index=False)

    cs = C.assign(sign=np.sign(C.coef)).groupby(["pool", "feature"]).agg(
        mean_coef=("coef", "mean"), n_pos=("sign", lambda s: int((s > 0).sum())),
        n_neg=("sign", lambda s: int((s < 0).sum())), n_zero=("sign", lambda s: int((s == 0).sum()))).reset_index()
    cs["sign_stable"] = (cs[["n_pos", "n_neg"]].max(1) == len(sources))
    cs.to_csv(out / "glassbox_coef_stability.csv", index=False)
    per_fold = I.groupby("heldout").apply(lambda g: spearmanr(g.importance, g.stability)[0])
    summary = {"eligible_sources": eligible,
               "n_stable": st.groupby("pool")["stable"].sum().to_dict(),
               "stable_features": st[st.stable].feature.tolist(),
               "selection_mean": S.groupby(["pool", "features", "model"])[["balanced_accuracy", "auc",
                                                                             "healthy_recall"]].mean().round(4)
               .reset_index().to_dict("records"),
               "glassbox_sign_stable": cs[cs.sign_stable].feature.tolist(),
               "rf_importance_vs_stability_spearman_per_fold": per_fold.round(3).to_dict()}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
