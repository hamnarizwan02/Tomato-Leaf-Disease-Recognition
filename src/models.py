"""Classical models + validation-based tuning.  (Owner: Member B)

Protocol: hyper-parameters are chosen on the VALIDATION split with the model fitted on TRAIN only.
Final models are refitted on TRAIN+VAL with the chosen parameters and touch the TEST split only once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("rf", "svm", "lr")


class PlattSVC(ClassifierMixin, BaseEstimator):
    """RBF SVC whose predictions are the plain SVC's; predict_proba from Platt (sigmoid) calibration fitted on
    internal cross-validated decision values (CalibratedClassifierCV, ensemble=False). Replaces SVC(probability=True),
    which scikit-learn 1.9 deprecates."""

    def __init__(self, C=1.0, gamma="scale", class_weight=None, random_state=None):
        self.C, self.gamma, self.class_weight, self.random_state = C, gamma, class_weight, random_state

    def _svc(self):
        return SVC(kernel="rbf", C=self.C, gamma=self.gamma, class_weight=self.class_weight,
                   random_state=self.random_state)

    def fit(self, X, y):
        self.svc_ = self._svc().fit(X, y)
        cv = int(min(5, np.bincount(np.unique(y, return_inverse=True)[1]).min()))
        self.cal_ = CalibratedClassifierCV(self._svc(), method="sigmoid", cv=max(cv, 2), ensemble=False).fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        return self.svc_.predict(X)

    def predict_proba(self, X):
        return self.cal_.predict_proba(X)


def make_model(name: str, params: dict, cfg: dict, proba: bool = False):
    """proba=True makes the SVM fit Platt-scaled probabilities too (slower; predictions are unchanged)."""
    seed = cfg["seed"]
    cw = cfg["models"].get("class_weight")
    if name == "rf":
        p = {"n_estimators": cfg["models"]["rf"]["n_estimators"], **params}
        return RandomForestClassifier(random_state=seed, n_jobs=cfg["n_jobs"], class_weight=cw, **p)
    if name == "svm":      # RBF-kernel SVM; scaling is essential for margin-based methods
        if proba:
            return make_pipeline(StandardScaler(), PlattSVC(random_state=seed, class_weight=cw, **params))
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed, class_weight=cw, **params))
    if name == "lr":       # L2-regularised multinomial logistic regression
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, random_state=seed, class_weight=cw, **params))
    raise ValueError(name)


def tune(name: str, Xtr, ytr, Xva, yva, cfg: dict):
    """Grid search on the validation split. Returns (best_params, table of all configurations)."""
    grid = list(ParameterGrid(cfg["models"][name]["grid"]))
    rows = []
    for p in grid:
        pred = make_model(name, p, cfg).fit(Xtr, ytr).predict(Xva)
        rows.append({**{k: str(v) for k, v in p.items()},
                     "val_accuracy": accuracy_score(yva, pred),
                     "val_macro_f1": f1_score(yva, pred, average="macro")})
    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_i = int(np.argmax([r["val_macro_f1"] for r in rows]))
    return grid[best_i], table
