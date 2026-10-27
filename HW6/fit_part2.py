"""PART 2 fit: thresholds for the decision rules (no neural network).

Computes the 4 PART 2 features (classifiers.part2_features) for every 2 s
window of every recording, then fits each question's thresholds with the same
meet-in-the-middle rule as PART 1 (calibrate_part1.midpoint_threshold):

    up           mean gravity direction while standing (what "upright" means)
    tilted       tilt_deg:                 upright (standing+walking+running) | biking
    still        acc_std, acc_peak, rot_var:  standing | walking
    walk_or_run  acc_std, acc_peak, rot_var:  walking  | running
    bike_or_run  acc_std, acc_peak, rot_var:  biking   | running

and checks the result (leave-one-person-out with >= 2 people). The board runs
the rules with classifiers.Part2Rules; HOW_IT_DECIDES.md section 2 explains them.

Normally this runs inside the GUI's calibration wizard (Shift+C in gui.py),
which also installs the result, keeps the old one in calibration_history/ and
deploys. Run from the command line it only writes --out:

    .venv/Scripts/python HW6/fit_part2.py --out <file>
    .venv/Scripts/python HW6/fit_part2.py --data HW6/testing/data_synthetic --out <file>
"""

import argparse
import json
import time

import numpy as np

import config
import classifiers as C
import dataset
from calibrate_part1 import midpoint_threshold

# question -> (lower activity or activities, upper activity, features)
QUESTIONS = {
    "tilted": (["standing", "walking", "running"], "biking", ["tilt_deg"]),
    "still": (["standing"], "walking", C.MOVEMENT_FEATURES),
    "walk_or_run": (["walking"], "running", C.MOVEMENT_FEATURES),
    "bike_or_run": (["biking"], "running", C.MOVEMENT_FEATURES),
}


def upright_direction(windows, labels):
    """Unit vector of the average gravity direction in the standing windows."""
    acc = windows[labels == "standing"][:, :, [C.COL["ax"], C.COL["ay"], C.COL["az"]]].mean(axis=1)
    up = (acc / np.linalg.norm(acc, axis=1, keepdims=True)).mean(axis=0)
    return up / np.linalg.norm(up)


def features_table(windows, up):
    """{feature: array over windows}"""
    rows = [C.part2_features(w, up) for w in windows]
    return {f: np.array([r[f] for r in rows]) for f in C.PART2_FEATURES}


def fit_questions(feats, labels):
    """-> questions dict for part2_rules.json, plus each test's margin."""
    questions, margins = {}, {}
    for q, (lower, upper, names) in QUESTIONS.items():
        lo_mask, hi_mask = np.isin(labels, lower), labels == upper
        lo_name = "upright" if len(lower) > 1 else lower[0]
        questions[q] = {}
        for f in names:
            lo, hi = feats[f][lo_mask], feats[f][hi_mask]
            if np.median(lo) <= np.median(hi):
                t, m = midpoint_threshold(lo, hi)
                test = {"threshold": round(t, 4), "above": upper, "below": lo_name}
            else:                                    # this feature runs the other way round
                t, m = midpoint_threshold(hi, lo)
                test = {"threshold": round(t, 4), "above": lo_name, "below": upper}
            questions[q][f] = test
            margins[f"{q}.{f}"] = round(m, 3)
    return questions, margins


def predict(rules, windows):
    clf = C.Part2Rules(rules)
    return [clf.classify(w)[0] for w in windows]


def fit_rules(recs, log=print):
    """Recordings -> the part2_rules.json contents (also used by the GUI)."""
    dataset.summarize(recs, log)
    windows, labels, persons, _ = dataset.stack_windows(recs)
    missing = [s for s in C.STATES if not np.any(labels == s)]
    if missing:
        raise ValueError(f"No recordings for {missing}. Record them first.")

    def fit(mask):
        up = upright_direction(windows[mask], labels[mask])
        questions, margins = fit_questions(features_table(windows[mask], up), labels[mask])
        return {"up": [round(float(v), 5) for v in up], "questions": questions}, margins

    people = sorted(set(persons.tolist()))
    if len(people) >= 2:
        how = "leave-one-person-out"
        true, pred = [], []
        for p in people:
            rules, _ = fit(persons != p)
            true += labels[persons == p].tolist()
            pred += predict(rules, windows[persons == p])
    else:
        how = "all data (only 1 person)"
    rules, margins = fit(np.ones(len(labels), bool))
    if len(people) < 2:
        true, pred = labels.tolist(), predict(rules, windows)

    up = np.array(rules["up"])
    feats = features_table(windows, up)
    log(f"\n'Up' (gravity direction while standing, sensor frame): {np.round(up, 2).tolist()}")
    log("\nFeatures per activity (5th - 95th percentile):")
    log(f"{'':10}" + "".join(f"{f:>18}" for f in C.PART2_FEATURES))
    for s in C.STATES:
        m = labels == s
        log(f"{s:10}" + "".join(f"{np.percentile(feats[f][m], 5):8.2f} -{np.percentile(feats[f][m], 95):8.2f}"
                                for f in C.PART2_FEATURES))
    log("\nQuestions (threshold, and which class each side votes for):")
    for q, tests in rules["questions"].items():
        for f, t in tests.items():
            log(f"  {q:12} {f:9} > {t['threshold']:8.3f} -> {t['above']:9} else {t['below']:9}"
                f"  (margin {margins[f'{q}.{f}']:.2f})")

    log(f"\nWindow-by-window result ({how}, before smoothing; 'unsure' keeps the previous state):")
    accuracy = dataset.confusion(true, pred, log=log)
    return {
        "type": "rules",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "up": rules["up"],
        "questions": rules["questions"],
        "margins": margins,
        "accuracy": round(float(accuracy), 4),
        "scored_by": how,
        "people": people,
        "windows_per_state": {s: int(np.sum(labels == s)) for s in C.STATES},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", required=True, help="where to write the result (install it with the GUI)")
    args = ap.parse_args()
    try:
        out = fit_rules(dataset.load_recordings(args.data))
    except ValueError as exc:
        raise SystemExit(f"{exc} ({args.data})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}. To use it, calibrate through the GUI wizard (Shift+C).")


if __name__ == "__main__":
    main()
