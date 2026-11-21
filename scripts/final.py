"""FINAL SUBMISSION: process A = baseline PCA, process B = improved logistic regression.

Self-contained: needs only numpy, pandas, scikit-learn (all in environment.yml).

PROCESS A - BASELINE (PCA monitoring, learns from good batches only)
  1. Features: mean and standard deviation of every sensor, per batch.
  2. Scale with the good batches; PCA keeps 90% of their variation.
  3. Two statistics per batch: T2 (extreme along normal patterns), Q (can't be rebuilt).
  4. Limits: the value 95% of UNSEEN good batches stay under (5-fold within good batches).
  5. Rejected if T2 or Q is above its limit. Rejected batches are never used.

PROCESS B - IMPROVED (logistic regression, learns from good AND rejected batches)
  1. Features per sensor: mean, std, median, 5th/95th percentile, robust noise
     (typical row-to-row jitter), relative step (largest jump within the batch divided
     by the batch's own level), plus batch length.
  2. Scale with median / MAD of the good batches.
  3. Logistic regression with balanced classes -> probability of "rejected".
  4. Cut-off: chosen on out-of-sample probabilities (5-fold CV) to maximise F2, which
     counts catching rejects twice as much as avoiding false alarms. On ties the lowest
     cut-off is taken. Policy: "accept only when confident; in doubt, reject".

VALIDATION
  Repeated stratified 5-fold CV on the labelled batches (A: 5 repeats, B: 10 repeats).
  Every batch is judged by models that never saw it. For B the cut-off is chosen inside
  each training part only (nested). Reported: the four leaderboard metrics (accuracy,
  precision, recall, F1; "rejected" = positive class).

HOW TO RUN
  As a script from the repo root:   python scripts/final.py
  In a notebook: paste everything except the last two lines, set DATA and OUT, then
      submission = main()
      metrics = submit(submission, user="Team XX")
"""
import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"  # also silences parallel worker processes
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold

# PATHS - adjust to your environment.
# DATA = folder containing BRPMM_template.csv and the "process A" / "process B" folders.
#   From this repo's root:                             Path("input-data/Project 2 - BRPMM")
#   Notebook inside that folder (leaderboard notebook): Path(".")
DATA = Path("input-data/Project 2 - BRPMM")
OUT = Path("scripts/results")  # in a notebook e.g. Path("results")

SEED = 0
CV_REPEATS = {"A": 5, "B": 10}
BETA = 2  # process B cut-off: F2 = recall counts twice as much as precision


# ============================================================================ features
def features_basic(df):
    """Process A: mean and standard deviation of every sensor, per batch."""
    sensors = [c for c in df.columns if c != "batch_id"]
    feats = df.groupby("batch_id", sort=False)[sensors].agg(["mean", "std"])
    feats.columns = [f"{s}_{stat}" for s, stat in feats.columns]
    return feats


def features_all(df):
    """Process B: basic + robust statistics of every sensor, per batch."""
    sensors = [c for c in df.columns if c != "batch_id"]
    rows = {}
    for batch, g in df.groupby("batch_id", sort=False):
        r = {"length": len(g)}
        for s in sensors:
            x = g[s].to_numpy(float)
            med, p05, p95 = np.median(x), np.percentile(x, 5), np.percentile(x, 95)
            d = np.diff(x)
            w = max(1, len(x) // 100)  # step window = 1% of the batch
            r[f"{s}_mean"] = x.mean()
            r[f"{s}_std"] = x.std(ddof=1)
            r[f"{s}_median"], r[f"{s}_p05"], r[f"{s}_p95"] = med, p05, p95
            r[f"{s}_noise"] = 1.4826 * np.median(np.abs(d - np.median(d)))
            r[f"{s}_relstep"] = np.abs(x[w:] - x[:-w]).max() / max(abs(med), p95 - p05, 1e-9)
        rows[batch] = r
    return pd.DataFrame(rows).T


# ============================================================================ process A model
class PCAMonitor:
    """Baseline: learns normal behaviour from good batches; flags batches beyond the T2 or Q limit."""

    def _learn(self, X_good):
        self.mean = X_good.mean(0)
        std = X_good.std(0)
        self.std = np.where(std > 0, std, 1.0)
        self.pca = PCA(n_components=0.90, svd_solver="full").fit((X_good - self.mean) / self.std)
        return self

    def _t2_q(self, X):
        Z = (X - self.mean) / self.std
        coords = self.pca.transform(Z)
        t2 = (coords ** 2 / self.pca.explained_variance_).sum(1)
        q = ((Z - self.pca.inverse_transform(coords)) ** 2).sum(1)
        return t2, q

    def fit(self, X, y):
        X_good = X[y == 0]  # only good batches are used
        self._learn(X_good)
        t2, q = np.zeros(len(X_good)), np.zeros(len(X_good))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(X_good):
            t2[b], q[b] = PCAMonitor()._learn(X_good[a])._t2_q(X_good[b])
        self.t2_limit, self.q_limit = np.percentile(t2, 95), np.percentile(q, 95)
        return self

    def predict(self, X):
        t2, q = self._t2_q(X)
        return ((t2 > self.t2_limit) | (q > self.q_limit)).astype(int)  # 1 = rejected


# ============================================================================ process B model
class LogRegMonitor:
    """Improved: logistic regression + cut-off chosen by F2 on out-of-sample probabilities."""

    @staticmethod
    def _fit_core(X, y):
        good = X[y == 0]
        center = np.median(good, 0)
        spread = 1.4826 * np.median(np.abs(good - center), 0)
        fallback = good.std(0)
        spread = np.where(spread > 1e-12, spread, np.where(fallback > 1e-12, fallback, 1.0))
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
        clf.fit(np.clip((X - center) / spread, -50, 50), y)
        return center, spread, clf

    @staticmethod
    def _proba(core, X):
        center, spread, clf = core
        return clf.predict_proba(np.clip((X - center) / spread, -50, 50))[:, 1]

    def fit(self, X, y):
        # out-of-sample probabilities for the training batches -> choose the cut-off
        oof = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            oof[b] = self._proba(self._fit_core(X[a], y[a]), X[b])
        best = -1.0
        for t in np.unique(oof):  # ascending; only strict improvements -> lowest cut-off on ties
            f = fbeta_score(y, (oof >= t).astype(int), beta=BETA, zero_division=0)
            if f > best + 1e-12:
                best, self.cutoff = f, t
        self.core = self._fit_core(X, y)
        return self

    def predict(self, X):
        return (self._proba(self.core, X) >= self.cutoff).astype(int)  # 1 = rejected


PROCESSES = {
    # process: (feature function, model factory, description)
    "A": (features_basic, PCAMonitor, "baseline PCA (T2/Q, 95% limits) on mean/std"),
    "B": (features_all, LogRegMonitor, "logistic regression on all features, F2 cut-off"),
}


# ============================================================================ validation
def validate(make_model, X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    preds = np.zeros((repeats, len(y)), dtype=int)
    for k, (train, held_out) in enumerate(cv.split(X, y)):
        preds[k // 5, held_out] = make_model().fit(X[train], y[train]).predict(X[held_out])
    metrics = pd.DataFrame([dict(accuracy=accuracy_score(y, p),
                                 precision=precision_score(y, p, zero_division=0),
                                 recall=recall_score(y, p),
                                 f1=f1_score(y, p)) for p in preds]).mean()
    return metrics, preds.mean(0)


# ============================================================================ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, summary = [], {}

    for p, (make_features, make_model, description) in PROCESSES.items():
        folder = DATA / f"process {p}"
        cal = pd.read_csv(folder / "calibration data" / "calibration_data.csv")
        test = pd.read_csv(folder / "test data" / "test_data.csv")
        labels = pd.read_csv(folder / "calibration data" / "calibration_labels.csv").set_index("batch_id")["label"]

        F_cal = make_features(cal).loc[labels.index]
        F_test = make_features(test)[F_cal.columns]
        X, X_test = F_cal.to_numpy(float), F_test.to_numpy(float)
        y = (labels == "rejected").to_numpy().astype(int)

        metrics, flag_rate = validate(make_model, X, y, CV_REPEATS[p])
        summary[p] = metrics
        missed = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 1 and f < 0.5]
        alarms = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 0 and f >= 0.5]

        pred = make_model().fit(X, y).predict(X_test)  # final model on all calibration data
        predictions.append(pd.DataFrame(dict(batch_id=F_test.index, project=p,
                                             label=np.where(pred == 1, "rejected", "accepted"))))

        print(f"\nPROCESS {p}: {description}")
        print(f"  {len(y)} labelled batches ({y.sum()} rejected), {X.shape[1]} features, "
              f"{CV_REPEATS[p]} x 5-fold CV")
        for name, value in metrics.items():
            print(f"  {name:10s} {value:.3f}")
        print(f"  missed rejects ({len(missed)}): {', '.join(missed) or '-'}")
        print(f"  false alarms   ({len(alarms)}): {', '.join(alarms) or '-'}")
        print(f"  test set: {pred.sum()} of {len(pred)} predicted rejected")

    # leaderboard file: same batch_ids, order and columns as the template
    template = pd.read_csv(DATA / "BRPMM_template.csv", dtype={"batch_id": str})
    submission = template[["batch_id", "project"]].merge(pd.concat(predictions), on=["batch_id", "project"], how="left")
    assert list(submission.batch_id) == list(template.batch_id), "rows differ from template"
    assert submission.label.notna().all(), "missing predictions"
    assert list(submission.columns) == list(template.columns), "columns differ from template"

    submission.to_csv(OUT / "submission_final.csv", index=False)
    pd.DataFrame(summary).T.round(4).to_csv(OUT / "final_validation.csv")
    print(f"\nwritten to {OUT.resolve()}: submission_final.csv, final_validation.csv")
    print(submission.groupby(["project", "label"]).size().unstack(fill_value=0).to_string())
    return submission


if __name__ == "__main__":
    main()
