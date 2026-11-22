"""Re-run one benchmark candidate for the precomputed demos (after changing it in engine.py), then redo step 4.
    python frontend/rerun_candidate.py "Logistic regression" A B
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

name, procs = sys.argv[1], sys.argv[2:] or ["A", "B"]
for p in procs:
    path = Path(__file__).resolve().parent / "demo" / f"{p}.pkl"
    R = pickle.load(open(path, "rb"))
    d = E.load_sample(p)
    cal, labels, new = d["cal"], d["labels"], d["new"]
    sensors = E.active_sensors(cal)
    F_cal = E.FN.features_all(cal[["batch_id"] + sensors]).loc[labels.index]
    F_new = E.FN.features_all(new[["batch_id"] + sensors])[F_cal.columns]
    y = E.y_of(labels)
    R["benchmark"].loc[name] = E.benchmark(F_cal, y, R["repeats"], only=[name]).loc[name]
    R["recommendation"] = E.recommend(R["benchmark"], R["explore"])
    R["monitor"] = E.monitor(F_cal, y, labels, F_new, R["recommendation"]["model"])
    pickle.dump(R, open(path, "wb"))
    print(p, R["recommendation"]["model"], R["benchmark"].loc[name, ["f1", "rejects_caught", "false_alarms"]].round(3).to_dict(),
          "test rejected:", int((R["monitor"]["new"].decision == "rejected").sum()))
