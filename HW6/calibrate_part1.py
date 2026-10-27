"""PART 1 one-time calibration: pick the best direction and its 3 thresholds.

Reads every recording in data/ (all people), and for each of the 6 candidate
directions (+x, -x, +y, -y, +z, -z: the 3 axes, both signs):
  1. computes the Part 1 feature for every 2 s window: the peak acceleration
     in that direction (same code the board runs)
  2. sorts the 4 activities by their median peak
  3. places a threshold between each pair of neighbouring activities
  4. scores it (leave-one-person-out accuracy when there are >= 2 people)
The best direction and its thresholds become part1_calibration.json; the board
then only ever looks at that one direction.

Normally this runs inside the GUI's calibration wizard (Shift+C in gui.py),
which also installs the result, keeps the old one in calibration_history/ and
deploys. Run from the command line it only writes --out, so it can never
replace the calibration in use by accident:

    .venv/Scripts/python HW6/calibrate_part1.py --out <file>
    .venv/Scripts/python HW6/calibrate_part1.py --data HW6/testing/data_synthetic --out <file>
"""

import argparse
import json
import time

import numpy as np

import config
import classifiers as C
import dataset


def class_stats(peaks):
    return {"median": float(np.median(peaks)), "p5": float(np.percentile(peaks, 5)),
            "p95": float(np.percentile(peaks, 95)), "std": float(np.std(peaks)), "n": int(len(peaks))}


def best_split(lo, hi):
    """Threshold between two overlapping classes that minimizes balanced error.

    Among equally good thresholds, the middle one is returned."""
    candidates = np.unique(np.concatenate([lo, hi]))
    mids = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
    errs = np.array([np.mean(lo >= t) + np.mean(hi < t) for t in mids])
    best = np.flatnonzero(errs <= errs.min() + 1e-12)
    return float(mids[best[len(best) // 2]])


def midpoint_threshold(lo, hi):
    """Threshold between a lower group of values and a higher one. -> (threshold, margin).

    Used by PART 1 and by every question in PART 2 (fit_part2.py).
    The edges of the two clusters are the lower group's 99th percentile and the
    upper one's 1st (nearly their extremes, but one stray window can't drag a
    threshold). With a clean gap the threshold meets in the middle of the empty
    space, so both sides get the same buffer. (Splitting by each side's spread
    instead gave a very steady activity, like standing, almost no room.) If the
    groups overlap, it's the value with the fewest balanced mistakes.
    margin = gap / (sum of the two std devs): how clearly the two groups separate.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    lo_edge, hi_edge = np.percentile(lo, 99), np.percentile(hi, 1)
    t = (lo_edge + hi_edge) / 2 if lo_edge < hi_edge else best_split(lo, hi)
    margin = (hi_edge - lo_edge) / (max(np.std(lo), 1e-3) + max(np.std(hi), 1e-3))
    return float(t), float(margin)


# Biking | standing boundary: standing gets the WHOLE gap, so the threshold sits
# right at biking's edge instead of in the middle. Standing has no motion, so
# its Part 1 reading is just the slice of gravity along the chosen direction,
# and that moves with the strap angle (along +x on Hudson's board: ~0.018 g per
# degree, so ~5 degrees of strap difference crossed the midpoint threshold and
# standing read as biking). Biking's edge is set by the pedaling itself, which
# the strap angle barely changes. See HOW_IT_DECIDES.md section 1.4.
STRAP_SENSITIVE_PAIR = {"standing", "biking"}
EDGE_SAFETY_G = 0.005    # stay this far outside biking's edge so its edge windows still read biking


def edge_threshold(lo, hi, room_for):
    """Threshold right at one cluster's edge, giving the whole gap to the other.

    room_for = "lo" puts it just below the upper cluster's edge; "hi" just above
    the lower cluster's edge. Falls back to midpoint_threshold if they overlap."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    lo_edge, hi_edge = np.percentile(lo, 99), np.percentile(hi, 1)
    if lo_edge + EDGE_SAFETY_G >= hi_edge - EDGE_SAFETY_G:
        return midpoint_threshold(lo, hi)
    t = lo_edge + EDGE_SAFETY_G if room_for == "hi" else hi_edge - EDGE_SAFETY_G
    margin = (hi_edge - lo_edge) / (max(np.std(lo), 1e-3) + max(np.std(hi), 1e-3))
    return float(t), float(margin)


def fit_thresholds(peaks, labels):
    """-> (order of states by median peak, 3 ascending thresholds, worst normalized margin)."""
    by_state = {s: peaks[labels == s] for s in C.STATES}
    order = sorted(C.STATES, key=lambda s: np.median(by_state[s]))
    thresholds, margins = [], []
    for lo_state, hi_state in zip(order, order[1:]):
        lo, hi = by_state[lo_state], by_state[hi_state]
        if {lo_state, hi_state} == STRAP_SENSITIVE_PAIR:
            t, margin = edge_threshold(lo, hi, room_for="lo" if lo_state == "standing" else "hi")
        else:
            t, margin = midpoint_threshold(lo, hi)
        thresholds.append(t)
        margins.append(margin)
    for i in range(1, 3):                     # keep them strictly increasing
        thresholds[i] = max(thresholds[i], thresholds[i - 1] + 1e-3)
    return order, thresholds, min(margins)


def predict(peaks, order, thresholds):
    clf = C.Part1PeakClassifier("+x", order, thresholds)   # direction unused by classify_peak
    return np.array([clf.classify_peak(p) for p in peaks])


def balanced_accuracy(labels, pred):
    return float(np.mean([np.mean(pred[labels == s] == s) for s in C.STATES if np.any(labels == s)]))


def score_direction(peaks, labels, persons):
    """Leave-one-person-out balanced accuracy if possible, else accuracy on all data."""
    people = sorted(set(persons))
    if len(people) >= 2:
        accs = []
        for p in people:
            train, test = persons != p, persons == p
            if any(not np.any(labels[train] == s) for s in C.STATES):
                continue
            order, th, _ = fit_thresholds(peaks[train], labels[train])
            accs.append(balanced_accuracy(labels[test], predict(peaks[test], order, th)))
        if accs:
            return float(np.mean(accs)), "leave-one-person-out"
    order, th, _ = fit_thresholds(peaks, labels)
    return balanced_accuracy(labels, predict(peaks, order, th)), "all data (only 1 person)"


def calibrate(recs, log=print):
    """Recordings -> the part1_calibration.json contents (also used by the GUI)."""
    dataset.summarize(recs, log)
    windows, labels, persons, _ = dataset.stack_windows(recs)
    missing = [s for s in C.STATES if not np.any(labels == s)]
    if missing:
        raise ValueError(f"No recordings for {missing}. Record them first.")

    log("\nCandidate directions (peak acceleration in g, median per activity):")
    log(f"{'dir':<5}{'order (low -> high peak)':<42}{'accuracy':>10}{'margin':>8}")
    results = []
    for d in C.DIRECTIONS:
        peaks = np.array([C.part1_peak(C.part1_direction_signal(w, d)) for w in windows])
        acc, how = score_direction(peaks, labels, persons)
        order, th, margin = fit_thresholds(peaks, labels)
        stats = {s: class_stats(peaks[labels == s]) for s in C.STATES}
        results.append({"direction": d, "accuracy": acc, "margin": margin, "order": order,
                        "thresholds": th, "stats": stats})
        desc = " < ".join(f"{s[:4]} {stats[s]['median']:.2f}" for s in order)
        log(f"{d:<5}{desc:<42}{acc:>9.1%}{margin:>8.2f}")

    best = max(results, key=lambda r: (round(r["accuracy"], 3), r["margin"]))
    log(f"\nScored by {how}. Best direction: {best['direction']}")
    log("Thresholds: " + ", ".join(f"{a} | {b} at {t:.2f} g"
                                   for a, b, t in zip(best["order"], best["order"][1:], best["thresholds"])))

    peaks = np.array([C.part1_peak(C.part1_direction_signal(w, best["direction"])) for w in windows])
    log("\nWindow-by-window result with the saved thresholds (all data, before smoothing):")
    dataset.confusion(labels, predict(peaks, best["order"], best["thresholds"]), log=log)

    return {
        "direction": best["direction"],
        "order": best["order"],
        "thresholds": [round(t, 4) for t in best["thresholds"]],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scored_by": how,
        "accuracy": round(best["accuracy"], 4),
        "people": sorted(set(persons.tolist())),
        "windows_per_state": {s: int(np.sum(labels == s)) for s in C.STATES},
        "peak_stats_g": best["stats"],
        "all_directions": [{k: r[k] for k in ("direction", "accuracy", "margin", "order", "thresholds")}
                           for r in results],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", required=True, help="where to write the result (install it with the GUI)")
    args = ap.parse_args()
    try:
        out = calibrate(dataset.load_recordings(args.data))
    except ValueError as exc:
        raise SystemExit(f"{exc} ({args.data})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}. To use it, calibrate through the GUI wizard (Shift+C).")


if __name__ == "__main__":
    main()
