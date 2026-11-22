"""Precompute every step of the strategy for the demo samples A and B, so the app opens instantly.

Run once from the repository root (takes ~10-20 minutes, mostly the nested cross-validation):
    python frontend/precompute.py            # both samples
    python frontend/precompute.py A          # one sample
Writes frontend/demo/<A|B>.pkl
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

OUT = Path(__file__).resolve().parent / "demo"
REPEATS = 3


def run(p):
    t0 = time.time()
    log = lambda msg: print(f"[{p} {time.time() - t0:5.0f}s] {msg}", flush=True)
    d = E.load_sample(p)
    cal, labels, new = d["cal"], d["labels"], d["new"]
    y = E.y_of(labels)
    sensors = E.active_sensors(cal)
    F_cal = E.FN.features_all(cal[["batch_id"] + sensors]).loc[labels.index]
    F_new = E.FN.features_all(new[["batch_id"] + sensors])[F_cal.columns]

    res = dict(name=p, quality=E.quality(cal, labels, new))
    log("explore")
    res["explore"] = E.explore(cal, labels, new)
    log("baseline")
    res["baseline"] = E.baseline(F_cal, y)
    log("benchmark")
    res["benchmark"] = E.benchmark(F_cal, y, REPEATS, progress=lambda f, n: log(f"  {n} done"))
    res["recommendation"] = E.recommend(res["benchmark"], res["explore"])
    log(f"recommended: {res['recommendation']['model']}")
    res["monitor"] = E.monitor(F_cal, y, labels, F_new, res["recommendation"]["model"])
    res["labels"] = labels
    res["repeats"] = REPEATS
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{p}.pkl", "wb") as f:
        pickle.dump(res, f)
    log(f"written {OUT / f'{p}.pkl'}")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["B", "A"]):
        run(p)
