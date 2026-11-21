"""BASELINE: classic PCA batch monitoring, identical for process A and process B.

THE METHOD
  1. FEATURES  Turn every batch into one row of numbers: for every sensor, its
               MEAN (typical level) and STANDARD DEVIATION (how much it moves).
  2. SCALE     Put all features on the same scale, using the GOOD batches only.
  3. PCA       Learn the main patterns of variation of the GOOD batches
               (keep enough components to explain 90% of their variation).
  4. SCORE     Two classic statistics per batch:
                 T2 = how far the batch lies ALONG the normal patterns
                      ("an extreme version of normal behaviour")
                 Q  = how badly PCA can REBUILD the batch
                      ("a pattern that never occurs in good batches")
  5. LIMITS    For each statistic: the value that 95% of good batches stay under.
               Computed on good batches the PCA has NOT seen (see `fit`), because
               batches the PCA learned from look extra normal -> limits too tight.
  6. DECIDE    Rejected if T2 or Q is above its limit.

Rejected batches are never used to build the model or set the limits.
They are only used to MEASURE how well it works (validation).

VALIDATION
  Repeated stratified 5-fold cross-validation: the labelled batches are split into
  5 parts (each with the same share of rejects). Each part is held out once while the
  model is built from the GOOD batches of the other 4 parts, then the held-out part
  is classified. Repeated with different random splits (A: 5x, B: 10x, since B has
  only 5 rejects). Every batch is thus judged several times by models that never saw it.

METRICS
  The leaderboard reports accuracy, precision, recall and F1 per process, with
  "rejected" as the positive class. The validation reports the same four, plus:
  balanced accuracy, AUC (ranking quality) and the false-alarm rate.

OUTPUT (in the folder set by OUT below)
  baseline_validation.csv            metrics per process
  baseline_validation_per_batch.csv  how often each batch was flagged, and its label
  submission_baseline.csv            leaderboard file, same rows/order as the template

Requirements: numpy, pandas, scikit-learn (all in environment.yml).

The data folder is found automatically (see _find_data below).
Run as a script from anywhere:
    python scripts/baseline.py
Or paste into a notebook (without the last two lines) and run:
    submission = main()
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold

# PATHS - found automatically (same logic as final.py).
# DATA = the folder containing BRPMM_template.csv and the "process A" / "process B" folders,
# searched upwards from the current folder and from this script's folder.
# OUT = a "results" folder next to this script.
def _find_data():
    here = [Path.cwd()]
    try:
        here.append(Path(__file__).resolve().parent)  # the folder of this script (not in notebooks)
    except NameError:
        pass
    for start in here:
        for base in [start, *start.parents]:
            for c in [base, base / "input-data" / "Project 2 - BRPMM"]:
                if (c / "BRPMM_template.csv").exists():
                    return c
    raise FileNotFoundError("BRPMM_template.csv not found - set DATA by hand")


def _find_out():
    try:
        return Path(__file__).resolve().parent / "results"
    except NameError:  # notebook: look for a scripts/ folder near the data
        for base in DATA.parents:
            hits = sorted(base.glob("scripts/baseline.py")) + sorted(base.glob("*/scripts/baseline.py"))
            if hits:
                return hits[0].parent / "results"
        return Path.cwd() / "results"


DATA = _find_data()
OUT = _find_out()

VARIANCE_KEPT = 0.90  # step 3: share of the good batches' variation PCA must explain
LIMIT_PERCENTILE = 95  # step 5: limit = value that this % of good batches stay under
CV_REPEATS = {"A": 5, "B": 10}


# ----------------------------------------------------------------------------- step 1
def batch_features(df):
    """One row per batch: mean and standard deviation of every sensor."""
    sensors = [c for c in df.columns if c != "batch_id"]
    feats = df.groupby("batch_id", sort=False)[sensors].agg(["mean", "std"])
    feats.columns = [f"{sensor}_{stat}" for sensor, stat in feats.columns]
    return feats


# ----------------------------------------------------------------------------- steps 2-6
class PCAMonitor:
    """Learns normal behaviour from good batches; flags batches that deviate from it."""

    def _learn_normal(self, X_good):
        # step 2: scale with the good batches (features that never vary get scale 1)
        self.mean = X_good.mean(axis=0)
        std = X_good.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        # step 3: PCA on the scaled good batches
        self.pca = PCA(n_components=VARIANCE_KEPT, svd_solver="full")
        self.pca.fit((X_good - self.mean) / self.std)
        return self

    def t2_and_q(self, X):
        # step 4: the two statistics
        Z = (X - self.mean) / self.std
        coords = self.pca.transform(Z)  # position along the normal patterns
        t2 = (coords ** 2 / self.pca.explained_variance_).sum(axis=1)
        rebuilt = self.pca.inverse_transform(coords)  # best rebuild from the normal patterns
        q = ((Z - rebuilt) ** 2).sum(axis=1)
        return t2, q

    def fit(self, X_good):
        self._learn_normal(X_good)
        # step 5: limits from good batches the PCA has not seen.
        # Split the good batches in 5 parts; score each part with a PCA built on the other 4.
        t2 = np.zeros(len(X_good))
        q = np.zeros(len(X_good))
        for part_learn, part_score in KFold(5, shuffle=True, random_state=0).split(X_good):
            helper = PCAMonitor()._learn_normal(X_good[part_learn])
            t2[part_score], q[part_score] = helper.t2_and_q(X_good[part_score])
        self.t2_limit = np.percentile(t2, LIMIT_PERCENTILE)
        self.q_limit = np.percentile(q, LIMIT_PERCENTILE)
        return self

    def score(self, X):
        """One number per batch: the larger of T2/limit and Q/limit. Above 1 = beyond a limit."""
        t2, q = self.t2_and_q(X)
        return np.maximum(t2 / self.t2_limit, q / self.q_limit)

    def predict(self, X):
        # step 6: 1 = rejected, 0 = accepted
        return (self.score(X) > 1).astype(int)


# ----------------------------------------------------------------------------- validation
def validate(X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    scores = np.zeros((repeats, len(y)))
    flags = np.zeros((repeats, len(y)), dtype=int)
    for k, (train, held_out) in enumerate(cv.split(X, y)):
        good_train = train[y[train] == 0]  # the model only ever sees good batches
        monitor = PCAMonitor().fit(X[good_train])
        scores[k // 5, held_out] = monitor.score(X[held_out])
        flags[k // 5, held_out] = monitor.predict(X[held_out])
    metrics = dict(
        # the four leaderboard metrics ("rejected" = positive class)
        accuracy=np.mean([accuracy_score(y, f) for f in flags]),
        precision=np.mean([precision_score(y, f, zero_division=0) for f in flags]),
        recall=np.mean([recall_score(y, f) for f in flags]),
        f1=np.mean([f1_score(y, f) for f in flags]),
        # extra diagnostics
        balanced_accuracy=np.mean([balanced_accuracy_score(y, f) for f in flags]),
        auc=np.mean([roc_auc_score(y, s) for s in scores]),
        false_alarms=np.mean([f[y == 0].mean() for f in flags]),
    )
    return metrics, flags.mean(axis=0)  # flag rate per batch over the repeats


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics, per_batch, predictions = {}, [], []

    for p in "AB":
        folder = DATA / f"process {p}"
        cal = pd.read_csv(folder / "calibration data" / "calibration_data.csv")
        test = pd.read_csv(folder / "test data" / "test_data.csv")
        labels = pd.read_csv(folder / "calibration data" / "calibration_labels.csv").set_index("batch_id")["label"]

        F_cal = batch_features(cal).loc[labels.index]
        F_test = batch_features(test)
        X = F_cal.to_numpy(float)
        y = (labels == "rejected").to_numpy().astype(int)

        # validation on the labelled batches
        metrics, flag_rate = validate(X, y, CV_REPEATS[p])
        all_metrics[p] = metrics
        per_batch.append(pd.DataFrame(dict(project=p, label=labels, flag_rate=flag_rate)))

        # final model: all good calibration batches -> classify the test batches
        monitor = PCAMonitor().fit(X[y == 0])
        pred = monitor.predict(F_test.to_numpy(float))
        predictions.append(pd.DataFrame(dict(batch_id=F_test.index, project=p,
                                             label=np.where(pred == 1, "rejected", "accepted"))))

        missed = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 1 and f < 0.5]
        false_alarms = [b for b, f, t in zip(labels.index, flag_rate, y) if t == 0 and f >= 0.5]
        print(f"\nPROCESS {p}: {len(y)} labelled batches ({y.sum()} rejected), {X.shape[1]} features, "
              f"PCA keeps {monitor.pca.n_components_} components")
        for name, value in metrics.items():
            print(f"  {name:18s} {value:.3f}")
        print(f"  missed rejects ({len(missed)}): {', '.join(missed) or '-'}")
        print(f"  false alarms   ({len(false_alarms)}): {', '.join(false_alarms) or '-'}")
        print(f"  test set: {pred.sum()} of {len(pred)} predicted rejected")

    # leaderboard file: same batch_ids, same order, same columns as the template
    template = pd.read_csv(DATA / "BRPMM_template.csv", dtype={"batch_id": str})
    submission = template[["batch_id", "project"]].merge(pd.concat(predictions), on=["batch_id", "project"], how="left")
    assert list(submission.batch_id) == list(template.batch_id), "rows differ from template"
    assert submission.label.notna().all(), "missing predictions"
    assert list(submission.columns) == list(template.columns), "columns differ from template"

    submission.to_csv(OUT / "submission_baseline.csv", index=False)
    pd.DataFrame(all_metrics).T.round(4).to_csv(OUT / "baseline_validation.csv")
    pd.concat(per_batch).round(3).to_csv(OUT / "baseline_validation_per_batch.csv", index_label="batch_id")
    print(f"\nwritten to {OUT.resolve()}: submission_baseline.csv, baseline_validation.csv, "
          "baseline_validation_per_batch.csv")
    return submission  # in a notebook: submission = main(); submit(submission, user="Team XX")


if __name__ == "__main__":
    main()
