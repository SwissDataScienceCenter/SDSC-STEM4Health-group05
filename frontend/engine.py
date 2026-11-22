"""Computation layer of the Batch Monitoring Studio. No Streamlit here, so it can be run and tested on its own.

Generic: works for any dataset in the challenge format
  - calibration data: one row per time step, a `batch_id` column + numeric sensor columns
  - calibration labels: `batch_id`, `label` (accepted / rejected)
  - new batches (optional): same columns as the calibration data, no labels

Every step of our strategy is one function:
  quality()     step 0  is the data usable?
  explore()     step 1  what does the data look like? linear or not? correlated or not?
  baseline()    step 2  label-free PCA with 95% limits, judged out-of-fold
  benchmark()   step 3  candidate models under nested cross-validation
  recommend()   step 3  pick a model, with reasons taken from steps 1-3
  monitor()     step 4  score batches: two opinions -> accept / review / reject + "why"

The models and features are the ones we submitted (scripts/final.py).
"""
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final as FN  # noqa: E402

N_POINTS = 200  # every batch is resampled to this many points for plots
ZONES = ["accept", "review", "reject"]


# ============================================================================ data
def load_sample(p):
    """Challenge data for process A or B: calibration data, labels, and the test batches as 'new' batches."""
    cal, test, labels = FN.load(p)
    return dict(cal=cal, labels=labels, new=test)


def sensors_of(df):
    return [c for c in df.columns if c != "batch_id" and pd.api.types.is_numeric_dtype(df[c])]


def active_sensors(df):
    """Sensors that actually vary (constant set-points carry no information)."""
    return [s for s in sensors_of(df) if df[s].std() > 1e-9]


def y_of(labels):
    return (labels == "rejected").to_numpy().astype(int)


# ============================================================================ step 0: quality
def quality(cal, labels, new=None):
    """A checklist: (status, title, detail) with status in ok / warn / fail."""
    checks = []
    add = lambda s, t, d: checks.append(dict(status=s, title=t, detail=d))
    sensors = sensors_of(cal)
    ids = cal.batch_id.unique()
    lab_ids = set(labels.index)

    add("ok" if "batch_id" in cal else "fail", "Batch identifier", "Column `batch_id` present.")
    unknown = set(labels.unique()) - {"accepted", "rejected"}
    add("fail" if unknown else "ok", "Label values",
        f"Unexpected labels: {sorted(unknown)}" if unknown else "Only `accepted` and `rejected`.")
    no_label = [b for b in ids if b not in lab_ids]
    add("warn" if no_label else "ok", "Every batch has a label",
        f"{len(no_label)} batches without a label are ignored." if no_label else f"All {len(ids)} batches labelled.")

    n_bad = int((labels == "rejected").sum())
    share = n_bad / max(len(labels), 1)
    add("warn" if n_bad < 10 else "ok", "Enough rejected examples",
        f"{n_bad} rejected batches ({share:.0%}). "
        + ("Few rejects: supervised models may overfit, so a label-free model (PCA) stays important."
           if n_bad < 10 else "Enough to train a supervised model next to the label-free one."))

    miss = cal[sensors].isna().mean()
    bad = miss[miss > 0]
    add("warn" if len(bad) else "ok", "Missing values",
        ", ".join(f"{s} {v:.1%}" for s, v in bad.items()) if len(bad) else "No missing values.")

    flat = [s for s in sensors if cal[s].std() <= 1e-9]
    add("ok", "Constant sensors",
        f"{len(flat)} constant (set-points, ignored in plots): {', '.join(flat)}" if flat else "None.")

    lengths = cal.groupby("batch_id").size()
    cv = lengths.std() / lengths.mean()
    add("warn" if cv > 0.05 else "ok", "Batch length",
        f"{lengths.min()}–{lengths.max()} time steps (median {int(lengths.median())}). "
        + ("Lengths vary, so batches are compared at the same relative progress (0–100%)." if cv > 0.05
           else "Lengths are almost equal."))

    if new is not None:
        missing_cols = [s for s in sensors if s not in new.columns]
        add("fail" if missing_cols else "ok", "New batches have the same sensors",
            f"Missing: {missing_cols}" if missing_cols else f"{new.batch_id.nunique()} new batches, same {len(sensors)} columns.")
    return checks


# ============================================================================ trajectories
def resample(df, sensors, n=N_POINTS):
    """batch_id -> array (n, n_sensors), time rescaled to 0..100% progress."""
    grid = np.linspace(0, 1, n)
    out = {}
    for b, g in df.groupby("batch_id", sort=False):
        x = g[sensors].to_numpy(float)
        t = np.linspace(0, 1, len(x))
        out[b] = np.column_stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])])
    return out


def relative(traj):
    """Each sensor relative to the batch's own level: (x - median) / max(|median|, p95 - p05)."""
    med = np.median(traj, 0)
    scale = np.maximum(np.abs(med), np.percentile(traj, 95, 0) - np.percentile(traj, 5, 0))
    return (traj - med) / np.where(scale > 1e-9, scale, 1.0)


def bands(trajs, good_ids, rel):
    """5th, 50th, 95th percentile of the good batches at every point: (3, n, n_sensors)."""
    stack = np.stack([relative(trajs[b]) if rel else trajs[b] for b in good_ids])
    return np.percentile(stack, [5, 50, 95], axis=0)


# ============================================================================ step 1: explore
def _batch_stats(df, cols):
    """Per-batch statistics used by the linearity test (as in scripts/linearity_test.py)."""
    rows = {}
    for b, g in df.groupby("batch_id"):
        r = {"length": len(g)}
        for c in cols:
            x = g[c].to_numpy()
            r[c + "_med"], r[c + "_p05"], r[c + "_p95"] = np.median(x), np.percentile(x, 5), np.percentile(x, 95)
            r[c + "_noise"] = np.std(np.diff(x))
        rows[b] = r
    return pd.DataFrame(rows).T


def linearity(cal, labels, sensors, max_targets=100, trees=300):
    """Hide one per-batch feature at a time and predict it from the others, across good batches:
    linear model (PCA + ridge) vs non-linear model (random forest). Cross-validated R² per feature."""
    good = cal[cal.batch_id.isin(labels.index[labels == "accepted"])]
    F = _batch_stats(good, sensors)
    F = F.loc[:, F.std() > 1e-12]
    F = (F - F.median()) / F.std()
    rng = np.random.default_rng(0)
    targets = list(F.columns) if F.shape[1] <= max_targets else list(rng.choice(F.columns, max_targets, replace=False))
    kf, res = KFold(5, shuffle=True, random_state=0), []
    for c in targets:
        X, y = F.drop(columns=c).to_numpy(), F[c].to_numpy()
        lin = make_pipeline(PCA(n_components=min(10, len(y) // 3)), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        rf = RandomForestRegressor(trees, min_samples_leaf=3, random_state=0, n_jobs=-1)
        res.append([c] + [r2_score(y, cross_val_predict(m, X, y, cv=kf)) for m in (lin, rf)])
    R = pd.DataFrame(res, columns=["feature", "linear", "nonlinear"])
    R["sensor"] = R.feature.str.rsplit("_", n=1).str[0].replace({"length": "batch length"})
    lin_m, non_m = R.linear.median(), R.nonlinear.median()
    verdict = ("linear" if lin_m >= non_m - 0.05 else "non-linear")
    return dict(per_feature=R, linear=lin_m, nonlinear=non_m, verdict=verdict)


def separation(F, y):
    """How well each feature alone separates rejected from good batches: AUC folded to 0.5–1."""
    rows = []
    for c in F.columns:
        x = F[c].to_numpy(float)
        if np.nanstd(x) < 1e-12:
            continue
        auc = roc_auc_score(y, x)
        rows.append(dict(feature=c, auc=max(auc, 1 - auc), direction="higher" if auc >= 0.5 else "lower"))
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def explore(cal, labels, new=None, linearity_trees=300):
    sensors = active_sensors(cal)
    y = y_of(labels)
    F = FN.features_all(cal[["batch_id"] + sensors]).loc[labels.index]
    good = F[y == 0]

    means = good[[f"{s}_mean" for s in sensors]]
    means.columns = sensors
    corr = means.corr().fillna(0)  # batch-to-batch correlation of sensor levels, good batches
    off = corr.to_numpy()[~np.eye(len(sensors), dtype=bool)]
    within = (cal[cal.batch_id.isin(good.index)].groupby("batch_id")[sensors].corr()
              .groupby(level=1).mean().reindex(index=sensors, columns=sensors).fillna(0))

    basic = good[[c for c in good.columns if c.endswith(("_mean", "_std"))]]
    Z = (basic - basic.mean()) / basic.std().replace(0, 1)
    ev = PCA().fit(Z).explained_variance_ratio_
    Fb = F[basic.columns]
    coords = PCA(2).fit(Z).transform((Fb - basic.mean()) / basic.std().replace(0, 1))

    trajs = resample(pd.concat([cal, new]) if new is not None else cal, sensors)
    good_ids = list(labels.index[labels == "accepted"])
    return dict(
        sensors=sensors, n_batches=len(labels), n_rejected=int(y.sum()),
        lengths=cal.groupby("batch_id").size(),
        corr=corr, within_corr=within, mean_abs_corr=float(np.abs(off).mean()),
        explained=ev, k80=int((np.cumsum(ev) < 0.8).sum() + 1), n_basic=basic.shape[1],
        pca_xy=pd.DataFrame(coords, index=F.index, columns=["PC1", "PC2"]).assign(label=labels.values),
        separation=separation(F, y),
        linearity=linearity(cal, labels, sensors, trees=linearity_trees),
        trajs=trajs, band_raw=bands(trajs, good_ids, False), band_rel=bands(trajs, good_ids, True),
    )


def implications(ex):
    """Plain-language consequences of the exploration for the model choice."""
    out = []
    L = ex["linearity"]
    if L["verdict"] == "linear":
        out.append(("Batch-to-batch variation is **linear**",
                    f"A linear model predicts the features as well as a random forest (R² {L['linear']:.2f} vs "
                    f"{L['nonlinear']:.2f}). PCA and logistic regression can capture the structure."))
    else:
        out.append(("Batch-to-batch variation is **non-linear**",
                    f"A random forest predicts the features clearly better than a linear model (R² "
                    f"{L['nonlinear']:.2f} vs {L['linear']:.2f}). PCA alone will miss part of the structure: "
                    "add a non-linear model (trees) next to it."))
    r = ex["mean_abs_corr"]
    out.append(("Sensors are " + ("**correlated**" if r > 0.3 else "**mostly independent**"),
                f"Mean |correlation| between sensor levels across good batches is {r:.2f}; "
                f"{ex['k80']} of {ex['n_basic']} PCA components are needed for 80% of the variance. "
                + ("PCA compresses the data well." if ex["k80"] <= ex["n_basic"] / 3
                   else "PCA compresses little, so T² and Q are noisier.")))
    top = ex["separation"].head(1)
    if len(top):
        t = top.iloc[0]
        out.append(("Strongest single signal: " + f"`{t.feature}`",
                    f"Alone it separates rejected from good batches with AUC {t.auc:.2f} "
                    f"(rejected batches have {t.direction} values)."
                    + (" A single feature already almost solves the problem." if t.auc > 0.95 else "")))
    share = ex["n_rejected"] / ex["n_batches"]
    out.append(("Class balance", f"{ex['n_rejected']} of {ex['n_batches']} batches rejected ({share:.0%}). "
                + ("Few examples of failure: keep a label-free opinion for unseen faults."
                   if ex["n_rejected"] < 15 else "Enough rejects for a supervised opinion.")))
    return out


# ============================================================================ step 2: baseline
def baseline(F, y, seed=FN.SEED):
    """Label-free PCA on mean/std features with 95% limits, every batch judged out-of-fold."""
    cols = list(F.columns)
    X = F.to_numpy(float)
    bc = FN.basic_cols(cols)
    t2n, qn = np.zeros(len(y)), np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        s = FN.PCAScorer(bc).fit(X[a], y[a])
        t2, q = s._t2_q(X[b][:, bc])
        t2n[b], qn[b] = t2 / s.t2_limit, q / s.q_limit
    score = np.maximum(t2n, qn)
    pred = (score > 1).astype(int)
    res = pd.DataFrame(dict(T2=t2n, Q=qn, score=score, pred=pred, truth=y), index=F.index)
    return dict(per_batch=res, metrics=FN.leaderboard_metrics(y, pred),
                confusion=confusion_matrix(y, pred, labels=[0, 1]))


# ============================================================================ step 3: benchmark
def _pca(cols):
    return FN.PCAScorer(FN.basic_cols(cols))


CANDIDATES = {
    # name: (uses labels?, factory(cols), short description)
    "PCA baseline": (False, lambda c: FN.FixedLimit(lambda: _pca(c)),
                     "Learns good batches only; rejects beyond the 95% T²/Q limit."),
    "PCA, tuned cut-off": (True, lambda c: FN.TunedCutoff(lambda: _pca(c), FN.leaderboard_f1),
                           "Same PCA score, cut-off tuned with the labels."),
    "Logistic regression": (True, lambda c: FN.TunedCutoff(lambda: FN.ClassifierScorer(FN.make_logreg), FN.leaderboard_f1),
                            "Linear classifier on all features, cut-off tuned for F1."),
    "Extra-trees": (True, lambda c: FN.TunedCutoff(lambda: FN.ClassifierScorer(FN.make_extratrees), FN.leaderboard_f1),
                    "Non-linear tree ensemble on all features."),
    "PCA + extra-trees": (True, lambda c: FN.make_ensemble_tuned(c),
                          "Average of the PCA score and the tree probability (as percentiles)."),
    "PCA OR (PCA + extra-trees)": (True, lambda c: FN.EitherRejects([
        lambda: FN.FixedLimit(lambda: _pca(c)), lambda: FN.make_ensemble_tuned(c)]),
                                   "Two opinions: reject if the PCA limit OR the ensemble rejects."),
}
COMPLEXITY = {n: i for i, n in enumerate(["PCA baseline", "PCA, tuned cut-off", "Logistic regression",
                                          "Extra-trees", "PCA + extra-trees", "PCA OR (PCA + extra-trees)"])}


def benchmark(F, y, repeats=3, only=None, progress=None):
    """Nested repeated stratified 5-fold CV; cut-offs tuned on inner folds only."""
    cols = list(F.columns)
    X = F.to_numpy(float)
    rows = []
    names = [n for n in CANDIDATES if only is None or n in only]
    for i, name in enumerate(names):
        uses_labels, make, desc = CANDIDATES[name]
        mean, sd, _ = FN.validate(lambda: make(cols), X, y, repeats)
        rows.append(dict(model=name, labels=uses_labels, description=desc, f1=mean.f1, f1_sd=float(np.nan_to_num(sd.f1)),
                         recall=mean.recall, precision=mean.precision, accuracy=mean.accuracy,
                         rejects_caught=mean.rejects_caught, false_alarms=mean.false_alarms))
        if progress:
            progress((i + 1) / len(names), name)
    return pd.DataFrame(rows).set_index("model")


def recommend(bench, ex):
    """Quality-first rule ("in doubt, reject"): among the models whose F1 is within one standard deviation
    of the best (statistically tied), pick the one that catches the most rejects; ties -> the simpler model."""
    best = bench.f1.idxmax()
    tol = max(float(np.nan_to_num(bench.loc[best, "f1_sd"])), 1e-9)
    close = bench[bench.f1 >= bench.loc[best, "f1"] - tol]
    pick = min(close.index, key=lambda n: (-round(close.loc[n, "rejects_caught"], 3), COMPLEXITY[n]))
    r = bench.loc[pick]
    tied = [n for n in close.index if n != pick]
    reasons = [f"Validated F1 (accepted = positive) **{r.f1:.3f} ± {r.f1_sd:.3f}**"
               + (" — the best of all candidates." if pick == best else
                  f", statistically tied with the best (`{best}`, {bench.loc[best, 'f1']:.3f}).")]
    if tied:
        reasons.append(f"Among the {len(close)} tied models it catches the most rejects (**{r.rejects_caught:.0%}**): "
                       "a missed bad batch costs more than a false alarm, so in doubt we reject.")
    base = bench.loc["PCA baseline"] if "PCA baseline" in bench.index else None
    if base is not None and pick != "PCA baseline":
        same = round(r.rejects_caught, 2) == round(base.rejects_caught, 2)
        reasons.append(f"Beats the label-free baseline (F1 {base.f1:.3f}): "
                       + (f"same catches ({r.rejects_caught:.0%}) " if same else
                          f"catches {r.rejects_caught:.0%} of rejects vs {base.rejects_caught:.0%}, ")
                       + f"with {r.false_alarms:.0%} false alarms vs {base.false_alarms:.0%}.")
    L = ex["linearity"]
    if pick == "Logistic regression":
        reasons.append(f"Fits the exploration: the data is {L['verdict']}, so a linear decision boundary is enough.")
    elif "extra-trees" in pick or pick == "Extra-trees":
        reasons.append(f"Fits the exploration: the data is {L['verdict']} (random forest R² {L['nonlinear']:.2f} vs "
                       f"linear {L['linear']:.2f}), so a tree model adds what PCA misses.")
    if "OR" in pick:
        reasons.append("Keeps a label-free PCA opinion, which can flag new kinds of faults never seen in training.")
    return dict(model=pick, reasons=reasons)


def supervised_partner(model):
    """The 'like a known reject?' opinion that goes with the chosen model."""
    return "Logistic regression" if model == "Logistic regression" else (
        "Extra-trees" if model == "Extra-trees" else "PCA + extra-trees")


# ============================================================================ step 4: monitor
def _odds_ratio(score, cutoff):
    """For scores between 0 and 1: odds(score) / odds(cut-off); 1 = at the cut-off."""
    odds = lambda s: np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))
    return odds(np.asarray(score, float)) / odds(cutoff)


def decision_risk(m, X):
    """The fitted model's own decision as a risk: >= 1 means it rejects, so a sensitivity
    threshold t can be applied as risk >= t (t = 1 is the model as fitted)."""
    if isinstance(m, FN.EitherRejects):
        return np.max([decision_risk(sub, X) for sub in m.models], axis=0)
    s = m.scorer.score(X)
    if isinstance(m, FN.FixedLimit):
        return s
    if isinstance(m.scorer, FN.PCAScorer):  # tuned PCA cut-off: positive, unbounded score
        return s / m.cutoff
    return _odds_ratio(s, m.cutoff)  # probabilities and percentiles in 0..1


class Monitor:
    """Two opinions per batch, each scaled so that 1.0 = that opinion's own cut-off.
    Decision = the recommended model's own decision."""

    def __init__(self, model):
        self.model = model
        self.partner = supervised_partner(model)

    def fit(self, F, y):
        self.cols = list(F.columns)
        X = F.to_numpy(float)
        self.pca = CANDIDATES["PCA baseline"][1](self.cols).fit(X, y)
        self.sup = CANDIDATES[self.partner][1](self.cols).fit(X, y)
        self.decider = CANDIDATES[self.model][1](self.cols).fit(X, y)
        self.center, self.spread = FN.robust_scale_params(X[y == 0])
        return self

    def score(self, F):
        X = F[self.cols].to_numpy(float)
        pca = self.pca.scorer.score(X)
        sup = _odds_ratio(self.sup.scorer.score(X), self.sup.cutoff)
        n = (pca > 1).astype(int) + (sup >= 1).astype(int)
        return pd.DataFrame(dict(pca_risk=pca, sup_risk=sup, zone=np.array(ZONES)[n],
                                 decision=np.where(self.decider.predict(X) == 1, "rejected", "accepted"),
                                 decision_risk=decision_risk(self.decider, X)),
                            index=F.index)

    def robust_z(self, F):
        return pd.DataFrame(np.clip((F[self.cols].to_numpy(float) - self.center) / self.spread, -50, 50),
                            index=F.index, columns=self.cols)


def monitor(F_cal, y, labels, F_new, model, folds=5):
    """Calibration batches are scored out-of-fold (true label known), new batches by the model on all data."""
    cal = []
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=FN.SEED).split(F_cal, y):
        m = Monitor(model).fit(F_cal.iloc[a], y[a])
        cal.append(m.score(F_cal.iloc[b]).join(m.robust_z(F_cal.iloc[b])))
    cal = pd.concat(cal).loc[F_cal.index]
    cal["truth"] = labels.values
    full = Monitor(model).fit(F_cal, y)
    out = dict(cal=cal, partner=full.partner, model=model,
               z_cols=full.cols, rejected_z=full.robust_z(F_cal[y == 1]))
    if F_new is not None:
        out["new"] = full.score(F_new).join(full.robust_z(F_new))
    return out


# ============================================================================ "why rejected"
FEATURE_TEXT = {
    "mean": ("average level higher than normal", "average level lower than normal"),
    "median": ("typical level higher than normal", "typical level lower than normal"),
    "std": ("swings more than normal", "unusually steady"),
    "p05": ("low values not as low as normal", "drops lower than normal"),
    "p95": ("peaks higher than normal", "never reaches its normal peak"),
    "noise": ("noisier than normal", "smoother than normal"),
    "relstep": ("sudden jump or dip inside the batch", "fewer sudden changes than normal"),
}


def describe_feature(f, z):
    if f == "length":
        return "batch runs longer than normal" if z > 0 else "batch is shorter than normal (stopped early?)"
    stat = f.rsplit("_", 1)[1]
    up, down = FEATURE_TEXT.get(stat, ("higher than normal", "lower than normal"))
    return up if z > 0 else down


def outside_window(traj, band, j):
    """Longest stretch (in % progress) where sensor j is outside the good band, or None."""
    lo, hi = band[0][:, j], band[2][:, j]
    out = (traj[:, j] < lo) | (traj[:, j] > hi)
    best, start, run_start = (0, None), None, None
    for i, o in enumerate(np.append(out, False)):
        if o and run_start is None:
            run_start = i
        elif not o and run_start is not None:
            if i - run_start > best[0]:
                best = (i - run_start, run_start)
            run_start = None
    if best[0] < 4:
        return None
    n = len(out)
    return round(100 * best[1] / n), round(100 * (best[1] + best[0]) / n)


def why(z_row, traj=None, band_raw=None, band_rel=None, sensors=None, top=3):
    """Top sensors with their most unusual feature, in plain words, plus where in the batch it happens."""
    s = z_row.abs()
    by_sensor = {}
    for f, v in s.items():
        sensor = "batch length" if f == "length" else f.rsplit("_", 1)[0]
        if v > by_sensor.get(sensor, (0, None))[0]:
            by_sensor[sensor] = (v, f)
    ranked = sorted(by_sensor.items(), key=lambda kv: -kv[1][0])
    contrib = pd.Series({k: v[0] for k, v in ranked})
    reasons = []
    for sensor, (v, f) in ranked[:top]:
        if v < 2:
            break
        r = dict(sensor=sensor, z=float(v), feature=f, text=describe_feature(f, z_row[f]), window=None)
        if traj is not None and sensors and sensor in sensors:
            j = sensors.index(sensor)
            band = band_rel if f.endswith("relstep") else band_raw
            r["window"] = outside_window(relative(traj) if f.endswith("relstep") else traj, band, j)
        reasons.append(r)
    return contrib, reasons


def nearest_reject(z_row, rejected_z):
    """The most similar known rejected batch (robust-scaled feature distance)."""
    if rejected_z is None or not len(rejected_z):
        return None, None
    d = np.sqrt(((np.clip(rejected_z, -10, 10) - np.clip(z_row[rejected_z.columns], -10, 10)) ** 2).mean(1))
    return d.idxmin(), float(d.min())
