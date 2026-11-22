"""Re-apply the recommendation rule to a precomputed demo and redo step 4 if the model changed (no new benchmark).
    python frontend/refresh_monitor.py A
    python frontend/refresh_monitor.py --force A B   # redo step 4 anyway (e.g. after changing Monitor)
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

force = "--force" in sys.argv
for p in [a for a in sys.argv[1:] if a != "--force"] or ["A", "B"]:
    path = Path(__file__).resolve().parent / "demo" / f"{p}.pkl"
    R = pickle.load(open(path, "rb"))
    R["recommendation"] = E.recommend(R["benchmark"], R["explore"])
    model = R["recommendation"]["model"]
    if force or R["monitor"]["model"] != model:
        d = E.load_sample(p)
        cal, labels, new = d["cal"], d["labels"], d["new"]
        sensors = E.active_sensors(cal)
        F_cal = E.FN.features_all(cal[["batch_id"] + sensors]).loc[labels.index]
        F_new = E.FN.features_all(new[["batch_id"] + sensors])[F_cal.columns]
        R["monitor"] = E.monitor(F_cal, E.y_of(labels), labels, F_new, model)
    pickle.dump(R, open(path, "wb"))
    print(p, "->", model)
