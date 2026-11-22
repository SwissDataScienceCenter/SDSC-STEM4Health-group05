"""GPU experiment for process A: a 1D-CNN on the raw sensor curves as an extra opinion.

final.py is NOT changed; this script imports its building blocks (features, PCA, extra-trees,
tuned cut-offs, nested CV) and adds one new scorer.

WHY
  A's models only see summary features (mean, std, percentiles, steps per sensor). A CNN reads the
  whole curve, so it can pick up shape and timing differences that summaries average away.
  With 161 batches a single net is noisy, so we train a BAG of K nets with different seeds and
  average them. On a GPU the K nets are trained together as one grouped-convolution network,
  so K=32 costs about the same as K=1.

INPUT per batch: every sensor resampled to T points, then z-scored per sensor AND time point
  against the good training batches (deviation from the average good trajectory), plus the
  batch length. Scaling is learned inside each training fold only.

CANDIDATES (all validated with final.validate: repeated stratified 5-fold, cut-offs tuned inside)
  pca_or_ensemble     reference: the submitted model of final.py
  cnn_tuned           CNN bag alone, cut-off tuned for the leaderboard F1
  pca_or_ensemble3    PCA limit OR tuned ensemble of (PCA, extra-trees, CNN)

HOW TO RUN (repo root)
  python scripts/gpu_A.py                      compare (GPU: ~minutes; CPU: very slow)
  python scripts/gpu_A.py --quick              smoke test: 1 repeat, 4 nets, fewer epochs
  python scripts/gpu_A.py --only cnn_tuned,pca_or_ensemble3
  python scripts/gpu_A.py --submit             also write results/submission_gpu.csv with the best
                                               A model (B = final.py's model, unchanged)
Output: scripts/results/compare_gpu_A.csv (+ submission_gpu.csv with --submit)
"""
import os
import sys
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch import nn

import final as FN

FN.PCA_VARIANCE = 0.90  # the setting of the submitted model (0.975 validated worse: F1 0.895 vs 0.908)
QUICK = "--quick" in sys.argv
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
T = 128                                  # points per batch
K = 4 if QUICK else 32                   # nets per bag
EPOCHS = 60 if QUICK else 200
REPEATS = 1 if QUICK else FN.CV_REPEATS["A"]
SEED = FN.SEED


# ============================================================================ data
def resample(df, sensors, batches, n=T):
    """(n_batches, n, n_sensors + 1): curves on n equally spaced points, plus the batch length."""
    groups = dict(tuple(df.groupby("batch_id", sort=False)))
    out = []
    for b in batches:
        x = groups[b][sensors].to_numpy(float)
        src, dst = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
        cols = [np.interp(dst, src, x[:, j]) for j in range(x.shape[1])]
        out.append(np.column_stack(cols + [np.full(n, len(x), float)]))
    return np.stack(out)


def prepare():
    """Features (as in final.py) and curves, packed side by side in one matrix so final.py's
    CV, ensembles and cut-offs can split rows as usual. The CNN unpacks the curve part."""
    cal, test, labels = FN.load("A")
    sensors = [c for c in cal.columns if c != "batch_id" and cal[c].nunique() > 1]
    F_cal = FN.features_all(cal).loc[labels.index]
    F_test = FN.features_all(test)[F_cal.columns]
    S_cal = resample(cal, sensors, labels.index)
    S_test = resample(test, sensors, F_test.index)
    X = np.hstack([F_cal.to_numpy(float), S_cal.reshape(len(S_cal), -1)])
    X_test = np.hstack([F_test.to_numpy(float), S_test.reshape(len(S_test), -1)])
    y = (labels == "rejected").to_numpy().astype(int)
    return X, X_test, y, list(F_cal.columns), S_cal.shape[1:], F_test.index


# ============================================================================ CNN bag
class CNNBag(nn.Module):
    """K independent small 1D-CNNs in one module (grouped convolutions: group k = net k)."""

    def __init__(self, c, k, h=16, dropout=0.3):
        super().__init__()
        self.k = k
        self.body = nn.Sequential(
            nn.Conv1d(k * c, k * h, 7, padding=3, groups=k), nn.ReLU(),
            nn.Conv1d(k * h, k * h, 5, padding=2, groups=k), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(k * h, k, 1, groups=k)

    def forward(self, x):                      # x: (B, c, T) -> logits (B, k)
        z = self.body(x.repeat(1, self.k, 1))  # same input for every net
        z = torch.amax(z, 2, keepdim=True)     # strongest response anywhere in the batch
        return self.head(self.drop(z)).squeeze(2)


class CNNScorer:
    """Supervised; score = mean probability of 'rejected' over the K nets of the bag."""

    def __init__(self, n_feat, shape):
        self.n_feat, self.shape = n_feat, shape

    def _curves(self, X):
        S = X[:, self.n_feat:].reshape(len(X), *self.shape)
        return np.clip((S - self.mean) / self.std, -10, 10).transpose(0, 2, 1)  # (B, c, T)

    def fit(self, X, y):
        S = X[:, self.n_feat:].reshape(len(X), *self.shape)
        good = S[y == 0]
        self.mean = good.mean(0)
        std = good.std(0)
        floor = 0.05 * good.std((0, 1))
        self.std = np.maximum(std, np.where(floor > 1e-12, floor, 1.0))
        Z = torch.tensor(self._curves(X), dtype=torch.float32, device=DEVICE)
        Y = torch.tensor(y, dtype=torch.float32, device=DEVICE).view(-1, 1).expand(-1, K)
        torch.manual_seed(SEED)
        self.net = CNNBag(Z.shape[1], K).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-2)
        pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.net.train()
        for _ in range(EPOCHS):  # full batch: ~130 batches fit in memory easily
            opt.zero_grad()
            noisy = Z + 0.1 * torch.randn_like(Z)  # light augmentation against overfitting
            loss_fn(self.net(noisy), Y).backward()
            opt.step()
        self.net.eval()
        return self

    def score(self, X):
        with torch.no_grad():
            Z = torch.tensor(self._curves(X), dtype=torch.float32, device=DEVICE)
            return torch.sigmoid(self.net(Z)).mean(1).cpu().numpy()


# ============================================================================ candidates
def candidates(cols, n_feat, shape):
    feat = list(range(n_feat))      # feature part only (the curves are not features)
    basic = FN.basic_cols(cols)
    pca = lambda: FN.PCAScorer(basic)
    trees = lambda: FN.ClassifierScorer(FN.make_extratrees, feat)
    cnn = lambda: CNNScorer(n_feat, shape)
    tuned = lambda makers: FN.TunedCutoff(lambda: FN.EnsembleScorer(makers), FN.leaderboard_f1)
    return {
        "pca_or_ensemble": lambda: FN.EitherRejects([lambda: FN.FixedLimit(pca), lambda: tuned([pca, trees])]),
        "cnn_tuned": lambda: FN.TunedCutoff(cnn, FN.leaderboard_f1),
        "pca_or_ensemble3": lambda: FN.EitherRejects([lambda: FN.FixedLimit(pca), lambda: tuned([pca, trees, cnn])]),
    }


def write_submission(name, make_model, X, X_test, y, test_ids):
    pred_a = make_model().fit(X, y).predict(X_test)
    F_cal, F_test, y_b, _ = FN.prepare("B", FN.B_MODEL[0])
    pred_b = FN.B_MODEL[1](list(F_cal.columns)).fit(F_cal.to_numpy(float), y_b).predict(F_test.to_numpy(float))
    preds = pd.concat([
        pd.DataFrame(dict(batch_id=test_ids, project="A", label=np.where(pred_a == 1, "rejected", "accepted"))),
        pd.DataFrame(dict(batch_id=F_test.index, project="B", label=np.where(pred_b == 1, "rejected", "accepted")))])
    template = pd.read_csv(FN.DATA / "BRPMM_template.csv", dtype={"batch_id": str})
    sub = template[["batch_id", "project"]].merge(preds, on=["batch_id", "project"], how="left")
    assert sub.label.notna().all() and list(sub.batch_id) == list(template.batch_id)
    sub.to_csv(FN.OUT / "submission_gpu.csv", index=False)
    print(f"\nwritten {FN.OUT / 'submission_gpu.csv'} (A: {name}, {pred_a.sum()} of {len(pred_a)} rejected; "
          f"B: final.py model, {pred_b.sum()} rejected)")


def main():
    print(f"device {DEVICE} | {K} nets per bag, {EPOCHS} epochs, {REPEATS} x 5-fold CV"
          + ("  [QUICK smoke test - numbers not comparable]" if QUICK else ""), flush=True)
    X, X_test, y, cols, shape, test_ids = prepare()
    cands = candidates(cols, len(cols), shape)
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(cands)
    rows = {}
    for name in only:
        t0 = time.time()
        mean, sd, _ = FN.validate(cands[name], X, y, REPEATS)
        rows[name] = {**mean.round(3), "f1_sd": round(sd.f1, 3), "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  {name:17s} F1 {mean.f1:.3f} (sd {sd.f1:.3f})  acc {mean.accuracy:.3f}  "
              f"rejects caught {mean.rejects_caught:.2f}  false alarms {mean.false_alarms:.3f}  "
              f"[{rows[name]['minutes']} min]", flush=True)
    res = pd.DataFrame(rows).T
    FN.OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(FN.OUT / ("compare_gpu_A_quick.csv" if QUICK else "compare_gpu_A.csv"), index_label="model")
    best = res.f1.astype(float).idxmax()
    print(f"\nbest validated F1: {best}")
    if "--submit" in sys.argv:
        write_submission(best, cands[best], X, X_test, y, test_ids)


if __name__ == "__main__":
    main()
