"""Run recordings through both parts exactly like the board does.

Uses the saved part1_calibration.json and part2_model.json (or --part1/--part2
files) and prints, for each part, the window-by-window confusion matrix before
smoothing and the accuracy after the board's 3-in-a-row smoothing.

Note: recordings used for calibration/training score higher than new ones.
Record a fresh session (or another person) to see real-world accuracy.

    .venv/Scripts/python HW6/replay.py
    .venv/Scripts/python HW6/replay.py --data HW6/data/alex
"""

import argparse
from pathlib import Path

import numpy as np

import config
import classifiers as C
import dataset


def run(recs, classify):
    """-> (true, raw predictions, smoothed predictions) over all windows."""
    true, raw, smooth = [], [], []
    for r in recs:
        for seg in r.segments:                 # the board restarts its window after a gap
            smoother = C.Smoother()
            for i in range(0, len(seg) - C.WINDOW + 1, C.HOP):
                state = classify(seg[i:i + C.WINDOW])
                true.append(r.label)
                raw.append(state)
                smooth.append(smoother.update(state))
    return true, raw, smooth


def report(name, recs, classify):
    print(f"\n==== {name} ====")
    true, raw, smooth = run(recs, classify)
    print("Each window on its own:")
    dataset.confusion(true, raw)
    t, s = np.array(true, dtype=object), np.array(smooth, dtype=object)
    print(f"After smoothing (what the matrix shows; the first 2 windows of each piece "
          f"have no state yet and count as wrong): {np.mean(t == s):.1%}")
    for person in sorted({r.person for r in recs}):
        tp, _, sp = run([r for r in recs if r.person == person], classify)
        print(f"  {person:<14}{np.mean(np.array(tp, dtype=object) == np.array(sp, dtype=object)):.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--part1", default=str(C.PART1_CALIBRATION_FILE))
    ap.add_argument("--part2", default=str(C.PART2_MODEL_FILE))
    args = ap.parse_args()

    data = Path(args.data)
    # Accept either data/ (people inside) or a single person's folder.
    recs = dataset.load_recordings(data)
    if not recs:
        recs = [r for r in dataset.load_recordings(data.parent) if r.person == data.name]
    if not recs:
        raise SystemExit(f"No recordings in {data}")
    dataset.summarize(recs)

    if Path(args.part1).exists():
        p1 = C.Part1PeakClassifier.load(args.part1)
        print(f"\nPart 1 direction {p1.direction}, order {p1.order}, thresholds {p1.thresholds}")
        report("PART 1: peak acceleration, one direction", recs, lambda w: p1.classify(w)[0])
    else:
        print(f"\nNo {args.part1}: run calibrate_part1.py first")

    if Path(args.part2).exists():
        p2 = C.Part2NeuralNet.load(args.part2)
        report("PART 2: neural network, all IMU outputs", recs, lambda w: p2.classify(w)[0])
    else:
        print(f"\nNo {args.part2}: run train_part2.py first")


if __name__ == "__main__":
    main()
