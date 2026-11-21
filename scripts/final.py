"""FINAL SUBMISSION for process A and process B.

Self-contained: needs only numpy, pandas, scikit-learn (all in environment.yml).

LEADERBOARD METRIC (confirmed from our own submission)
  Per process: accuracy, precision, recall, F1 with "accepted" as the POSITIVE class.
  Score = average of F1(A) and F1(B). Missed rejects and false alarms both cost F1,
  so the goal is the fewest total errors, not catching every reject.

PROCESS A - selected by `--compare` (see A_MODEL below)
  pca_limit       baseline PCA on mean/std per sensor, T2/Q above the 95% limit of unseen
                  good batches -> rejected. Learns from good batches only.
  pca_tuned       same PCA score, cut-off tuned to maximise the leaderboard F1.
  ensemble_tuned  average of two opinions, each turned into a percentile:
                    PCA score (how unlike the good batches)
                    extra-trees probability (how much like the known rejects)
                  cut-off tuned to maximise the leaderboard F1.
  pca_or_ensemble rejected if pca_limit OR ensemble_tuned rejects ("in doubt, reject").
                  SELECTED: best validated F1 (0.908 vs 0.891 for pca_limit).

PROCESS B - logistic regression (unchanged; scored 1.0 on the leaderboard)
  Features per sensor: mean, std, median, 5th/95th percentile, robust noise, relative
  step (largest jump within the batch / the batch's own level), plus batch length.
  Cut-off: maximise F2 (recall of rejects counts double) on out-of-sample probabilities.

TUNED CUT-OFFS are always chosen on out-of-sample scores of the training batches
(inner 5-fold CV), never on the batches being judged.

VALIDATION
  Repeated stratified 5-fold CV on the labelled batches (A: 5 repeats, B: 10 repeats).
  Every batch is judged by models that never saw it; cut-offs are tuned inside each
  training part only (nested).

HOW TO RUN (the data folder is found automatically)
  python scripts/final.py             validate the chosen models + write the submission
  python scripts/final.py --a-model pca_limit
                                      same, with another A model (pca_limit = first submission)
  python scripts/final.py --compare   compare the process A candidates (takes ~15 min)

HISTORY: every run also keeps a timestamped copy in results/history/ and adds a row to
results/submission_history.csv, so earlier submissions are never lost.
  In a notebook: paste everything except the last lines (from `if __name__`), then
      submission = main()
      metrics = submit(submission, user="Team XX")
  Output: scripts/results/submission_final.csv
"""
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"  # also silences parallel worker processes
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold

SEED = 0
CV_REPEATS = {"A": 5, "B": 10}
# chosen with --compare (validated leaderboard F1 for A): pca_limit 0.891, pca_tuned 0.886,
# ensemble_tuned 0.906, pca_or_ensemble 0.908 -> pca_or_ensemble
# (it also keeps every rejection of pca_limit, whose rejections the leaderboard confirmed as ~91% correct)
A_MODEL = "pca_or_ensemble"
if "--a-model" in sys.argv:  # override from the command line, e.g. --a-model pca_limit
    A_MODEL = sys.argv[sys.argv.index("--a-model") + 1]


# ============================================================================ paths
def _find_data():
    """Folder with BRPMM_template.csv, searched upwards from the current and the script folder."""
    here = [Path.cwd()]
    try:
        here.append(Path(__file__).resolve().parent)
    except NameError:  # notebook
        pass
    for start in here:
        for base in [start, *start.parents]:
            for c in [base, base / "input-data" / "Project 2 - BRPMM"]:
                if (c / "BRPMM_template.csv").exists():
                    return c
    raise FileNotFoundError("BRPMM_template.csv not found - set DATA by hand")


def _find_out():
    """A "results" folder next to this script (in a notebook: next to the nearest scripts/final.py)."""
    try:
        return Path(__file__).resolve().parent / "results"
    except NameError:
        for base in DATA.parents:
            hits = sorted(base.glob("scripts/final.py")) + sorted(base.glob("*/scripts/final.py"))
            if hits:
                return hits[0].parent / "results"
        return Path.cwd() / "results"


DATA = _find_data()
OUT = _find_out()


# ============================================================================ features
def features_basic(df):
    """Mean and standard deviation of every sensor, per batch."""
    sensors = [c for c in df.columns if c != "batch_id"]
    feats = df.groupby("batch_id", sort=False)[sensors].agg(["mean", "std"])
    feats.columns = [f"{s}_{stat}" for s, stat in feats.columns]
    return feats


def features_all(df):
    """Basic + robust statistics of every sensor, per batch."""
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


def robust_scale_params(X_good):
    center = np.median(X_good, 0)
    spread = 1.4826 * np.median(np.abs(X_good - center), 0)
    fallback = X_good.std(0)
    return center, np.where(spread > 1e-12, spread, np.where(fallback > 1e-12, fallback, 1.0))


# ============================================================================ scorers
# A scorer: fit(X, y) -> self ; score(X) -> array, higher = more likely rejected.
class PCAScorer:
    """Baseline PCA on good batches. Score = max(T2/limit, Q/limit); above 1 = beyond a 95% limit."""

    def __init__(self, cols=None):
        self.cols = cols  # column indices to use (None = all)

    def _learn(self, X_good):
        self.mean = X_good.mean(0)
        std = X_good.std(0)
        self.std = np.where(std > 0, std, 1.0)
        self.pca = PCA(n_components=0.90, svd_solver="full").fit((X_good - self.mean) / self.std)
        return self

    def _t2_q(self, X):
        Z = (X - self.mean) / self.std
        coords = self.pca.transform(Z)
        return (coords ** 2 / self.pca.explained_variance_).sum(1), ((Z - self.pca.inverse_transform(coords)) ** 2).sum(1)

    def fit(self, X, y):
        X = X if self.cols is None else X[:, self.cols]
        X_good = X[y == 0]  # only good batches
        self._learn(X_good)
        t2, q = np.zeros(len(X_good)), np.zeros(len(X_good))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(X_good):
            t2[b], q[b] = PCAScorer()._learn(X_good[a])._t2_q(X_good[b])
        self.t2_limit, self.q_limit = np.percentile(t2, 95), np.percentile(q, 95)
        return self

    def score(self, X):
        t2, q = self._t2_q(X if self.cols is None else X[:, self.cols])
        return np.maximum(t2 / self.t2_limit, q / self.q_limit)


class ClassifierScorer:
    """Supervised classifier on robust-scaled features; score = probability of 'rejected'."""

    def __init__(self, make_clf, cols=None):
        self.make_clf, self.cols = make_clf, cols

    def fit(self, X, y):
        X = X if self.cols is None else X[:, self.cols]
        self.center, self.spread = robust_scale_params(X[y == 0])
        self.clf = self.make_clf().fit(np.clip((X - self.center) / self.spread, -50, 50), y)
        return self

    def score(self, X):
        X = X if self.cols is None else X[:, self.cols]
        return self.clf.predict_proba(np.clip((X - self.center) / self.spread, -50, 50))[:, 1]


class EnsembleScorer:
    """Average percentile of several scorers. Percentiles are relative to out-of-sample scores
    of the training batches, so the scorers are on the same scale."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, X, y):
        self.refs, self.models = [], []
        for make in self.makers:
            oof = np.zeros(len(y))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
                oof[b] = make().fit(X[a], y[a]).score(X[b])
            self.refs.append(np.sort(oof))
            self.models.append(make().fit(X, y))
        return self

    def score(self, X):
        return np.mean([np.searchsorted(r, m.score(X), side="right") / len(r)
                        for r, m in zip(self.refs, self.models)], axis=0)


# ============================================================================ decision rules
def leaderboard_f1(y, pred):
    """F1 with 'accepted' as the positive class (y, pred: 1 = rejected)."""
    return f1_score(1 - y, 1 - pred, zero_division=0)


def f2_rejected(y, pred):
    return fbeta_score(y, pred, beta=2, zero_division=0)


class FixedLimit:
    """Rejected if the score is above 1 (i.e. beyond the scorer's own 95% limit)."""

    def __init__(self, make_scorer):
        self.make_scorer = make_scorer

    def fit(self, X, y):
        self.scorer = self.make_scorer().fit(X, y)
        return self

    def predict(self, X):
        return (self.scorer.score(X) > 1).astype(int)


class TunedCutoff:
    """Rejected if the score is at/above a cut-off chosen to maximise `objective`
    on out-of-sample scores of the training batches (inner 5-fold CV)."""

    def __init__(self, make_scorer, objective):
        self.make_scorer, self.objective = make_scorer, objective

    def fit(self, X, y):
        oof = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            oof[b] = self.make_scorer().fit(X[a], y[a]).score(X[b])
        best = -1.0
        for t in np.unique(oof):  # ascending; only strict improvements -> lowest cut-off on ties
            v = self.objective(y, (oof >= t).astype(int))
            if v > best + 1e-12:
                best, self.cutoff = v, t
        self.scorer = self.make_scorer().fit(X, y)
        return self

    def predict(self, X):
        return (self.scorer.score(X) >= self.cutoff).astype(int)


class EitherRejects:
    """Two-opinion rule: rejected if ANY of the models rejects ("in doubt, reject")."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, X, y):
        self.models = [make().fit(X, y) for make in self.makers]
        return self

    def predict(self, X):
        return np.max([m.predict(X) for m in self.models], axis=0)


# ============================================================================ models per process
def basic_cols(columns):
    return [i for i, c in enumerate(columns) if c.endswith(("_mean", "_std"))]


def make_extratrees():
    return ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2,
                                random_state=SEED, n_jobs=-1)


def make_logreg():
    return LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)


def make_ensemble_tuned(cols):
    return TunedCutoff(lambda: EnsembleScorer([lambda: PCAScorer(basic_cols(cols)),
                                               lambda: ClassifierScorer(make_extratrees)]), leaderboard_f1)


A_CANDIDATES = {
    # name: (feature function, model factory taking the feature column names)
    "pca_limit": (features_basic, lambda cols: FixedLimit(lambda: PCAScorer())),
    "pca_tuned": (features_basic, lambda cols: TunedCutoff(lambda: PCAScorer(), leaderboard_f1)),
    "ensemble_tuned": (features_all, lambda cols: make_ensemble_tuned(cols)),
    "pca_or_ensemble": (features_all, lambda cols: EitherRejects([
        lambda: FixedLimit(lambda: PCAScorer(basic_cols(cols))),
        lambda: make_ensemble_tuned(cols)])),
}
B_MODEL = (features_all, lambda cols: TunedCutoff(lambda: ClassifierScorer(make_logreg), f2_rejected))


# ============================================================================ validation
def load(p):
    folder = DATA / f"process {p}"
    cal = pd.read_csv(folder / "calibration data" / "calibration_data.csv")
    test = pd.read_csv(folder / "test data" / "test_data.csv")
    labels = pd.read_csv(folder / "calibration data" / "calibration_labels.csv").set_index("batch_id")["label"]
    return cal, test, labels


def leaderboard_metrics(y, pred):
    acc_true, acc_pred = 1 - y, 1 - pred  # 'accepted' = positive
    return dict(accuracy=accuracy_score(acc_true, acc_pred),
                precision=precision_score(acc_true, acc_pred, zero_division=0),
                recall=recall_score(acc_true, acc_pred, zero_division=0),
                f1=f1_score(acc_true, acc_pred, zero_division=0),
                rejects_caught=pred[y == 1].mean(),
                false_alarms=pred[y == 0].mean())


def validate(make_model, X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    preds = np.zeros((repeats, len(y)), dtype=int)
    for k, (train, held_out) in enumerate(cv.split(X, y)):
        preds[k // 5, held_out] = make_model().fit(X[train], y[train]).predict(X[held_out])
    per_repeat = pd.DataFrame([leaderboard_metrics(y, p) for p in preds])
    return per_repeat.mean(), per_repeat.std(), preds.mean(0)


def prepare(p, make_features):
    cal, test, labels = load(p)
    F_cal = make_features(cal).loc[labels.index]
    F_test = make_features(test)[F_cal.columns]
    y = (labels == "rejected").to_numpy().astype(int)
    return F_cal, F_test, y, labels


# ============================================================================ main
def compare_a():
    """Compare the process A candidates under the leaderboard metric."""
    rows = {}
    cache = {}
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(A_CANDIDATES)
    for name, (make_features, make_model) in A_CANDIDATES.items():
        if name not in only:
            continue
        if make_features not in cache:
            cache[make_features] = prepare("A", make_features)
        F_cal, _, y, _ = cache[make_features]
        cols = list(F_cal.columns)
        mean, sd, _ = validate(lambda: make_model(cols), F_cal.to_numpy(float), y, CV_REPEATS["A"])
        rows[name] = {**mean.round(3), "f1_sd": round(sd.f1, 3)}
        print(f"  {name:15s} F1 {mean.f1:.3f} (sd {sd.f1:.3f})  acc {mean.accuracy:.3f}  "
              f"prec {mean.precision:.3f}  rec {mean.recall:.3f}  | rejects caught {mean.rejects_caught:.2f}  "
              f"false alarms {mean.false_alarms:.3f}", flush=True)
    res = pd.DataFrame(rows).T
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "compare_A.csv", index_label="model")
    print(f"\nbest F1: {res.f1.astype(float).idxmax()}   (written to {OUT / 'compare_A.csv'})")


def save_history(submission, summary, score):
    """Keep a timestamped copy of every submission file and log it in submission_history.csv."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hist_dir = OUT / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    name = f"submission_{stamp}_A-{A_MODEL}.csv"
    submission.to_csv(hist_dir / name, index=False)
    rejected = submission[submission.label == "rejected"].groupby("project").size()
    row = pd.DataFrame([dict(
        timestamp=stamp, a_model=A_MODEL, b_model="logreg_f2",
        validated_f1_a=round(summary["A"].f1, 4), validated_f1_b=round(summary["B"].f1, 4),
        validated_score=round(score, 4),
        rejected_a=int(rejected.get("A", 0)), rejected_b=int(rejected.get("B", 0)),
        file=f"history/{name}", leaderboard_score="")])  # fill in by hand after submitting
    log = OUT / "submission_history.csv"
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"history: {hist_dir / name} (logged in {log.name})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, summary = [], {}
    models = {"A": A_CANDIDATES[A_MODEL], "B": B_MODEL}
    names = {"A": A_MODEL, "B": "logistic regression, F2 cut-off"}

    for p, (make_features, make_model) in models.items():
        F_cal, F_test, y, labels = prepare(p, make_features)
        cols = list(F_cal.columns)
        X, X_test = F_cal.to_numpy(float), F_test.to_numpy(float)

        mean, sd, flag_rate = validate(lambda: make_model(cols), X, y, CV_REPEATS[p])
        summary[p] = mean
        pred = make_model(cols).fit(X, y).predict(X_test)  # final model on all calibration data
        predictions.append(pd.DataFrame(dict(batch_id=F_test.index, project=p,
                                             label=np.where(pred == 1, "rejected", "accepted"))))

        missed = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 1 and f < 0.5]
        alarms = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 0 and f >= 0.5]
        print(f"\nPROCESS {p}: {names[p]}  ({len(y)} labelled batches, {y.sum()} rejected, "
              f"{CV_REPEATS[p]} x 5-fold CV)")
        print("  leaderboard metrics ('accepted' = positive):")
        for m in ["accuracy", "precision", "recall", "f1"]:
            print(f"    {m:10s} {mean[m]:.3f}")
        print(f"  rejects caught {mean.rejects_caught:.2f}, false alarms {mean.false_alarms:.3f}")
        print(f"  missed rejects ({len(missed)}): {', '.join(missed) or '-'}")
        print(f"  false alarms   ({len(alarms)}): {', '.join(alarms) or '-'}")
        print(f"  test set: {pred.sum()} of {len(pred)} predicted rejected")

    score = (summary["A"].f1 + summary["B"].f1) / 2
    print(f"\nVALIDATED SCORE (mean F1 of A and B): {score:.3f}")

    template = pd.read_csv(DATA / "BRPMM_template.csv", dtype={"batch_id": str})
    submission = template[["batch_id", "project"]].merge(pd.concat(predictions), on=["batch_id", "project"], how="left")
    assert list(submission.batch_id) == list(template.batch_id), "rows differ from template"
    assert submission.label.notna().all(), "missing predictions"
    assert list(submission.columns) == list(template.columns), "columns differ from template"

    submission.to_csv(OUT / "submission_final.csv", index=False)
    pd.DataFrame(summary).T.round(4).assign(score=round(score, 4)).to_csv(OUT / "final_validation.csv")
    print(f"written to {OUT}: submission_final.csv, final_validation.csv")
    save_history(submission, summary, score)
    print(submission.groupby(["project", "label"]).size().unstack(fill_value=0).to_string())
    return submission


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_a()
    else:
        main()
