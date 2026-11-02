"""Run recordings through the exact board code (stepnet.py) and compare with the truth.

For every recording: the network's final step count ± its 95% range vs. what
Hudson counted, the distance ± vs. the measured distance, and a plain
gyro-peak counter (one fixed threshold, no network) for comparison. Note that
the hallway walks were trained on, so only the long walk(s) are a fair test.

    .venv/Scripts/python HW7/replay.py                       # all recordings, installed model
    .venv/Scripts/python HW7/replay.py --plot                # + a figure per recording in figures/replay/
    .venv/Scripts/python HW7/replay.py --data HW7/testing/data_synthetic --no-hw6 --model m.json --plot
"""

import argparse
from pathlib import Path

import numpy as np

import config
import dataset
import labels as L
import stepnet as S


def run_recording(rec, ens):
    """Feed a recording to the board code, reply by reply.

    -> dict of arrays over time: t, steps, steps_sd, dist, dist_sd, rate, rate_sd, epistemic; and the final summary
    """
    win = S.Windower(config.MAX_GAP_ROWS)
    acc = S.StepAccumulator(ens)
    rows = []
    for seq, chunk in rec.replies():
        for window, new_rows, first in win.add_rows(seq, chunk):
            acc.add(ens.predict(window), new_rows, first)
            s = acc.summary()
            p = acc.last
            rows.append((rec.t[win.rows_total - 1], s["steps"], s["steps_sd"], s["dist_m"], s["dist_sd"],
                         p["rate"], p["rate_sd"], p["rate_epistemic"]))
    a = np.array(rows) if rows else np.zeros((0, 8))
    names = ["t", "steps", "steps_sd", "dist", "dist_sd", "rate", "rate_sd", "epistemic"]
    return {n: a[:, i] for i, n in enumerate(names)}, acc.summary()


def true_running_steps(lab, t):
    """Running step count from the stride labels (2 x phase), for plotting against the network."""
    return 2 * lab.phase(t)


def plot(rec, lab, hist, final, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), dpi=90, sharex=True)
    t = hist["t"]
    for k, a in ((1.0, 0.3), (S.Z95, 0.15)):
        a1.fill_between(t, hist["steps"] - k * hist["steps_sd"], hist["steps"] + k * hist["steps_sd"],
                        color="#1a56c4", alpha=a, lw=0)
    a1.plot(t, hist["steps"], color="#1a56c4", lw=1.5, label="network (68% / 95% ranges)")
    if lab is not None and len(lab.knots_t):
        a1.plot(rec.t, true_running_steps(lab, rec.t), color="black", lw=1, ls="--", label="from the stride labels")
    if rec.steps is not None:
        a1.axhline(rec.steps, color="#d63a2f", lw=0.8, ls=":", label=f"counted: {rec.steps}")
    a1.set_ylabel("steps")
    a1.set_title(f"{rec.name}: {final['steps_shown']} ± {S.Z95 * final['steps_sd']:.1f} steps, "
                 f"{final['dist_m']:.1f} ± {S.Z95 * final['dist_sd']:.1f} m", fontsize=10)
    a1.legend(loc="upper left", fontsize=8)
    a1.grid(alpha=0.3)
    a2.fill_between(t, 120 * (hist["rate"] - hist["rate_sd"]), 120 * (hist["rate"] + hist["rate_sd"]),
                    color="#1a56c4", alpha=0.25, lw=0)
    a2.plot(t, 120 * hist["rate"], color="#1a56c4", lw=1.2, label="cadence ± spread")
    a2.plot(t, 120 * hist["epistemic"], color="#c77800", lw=1, label="networks' disagreement")
    a2.set_ylabel("steps/min")
    a2.set_xlabel("seconds")
    a2.legend(loc="upper left", fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec.path.stem}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--no-hw6", action="store_true")
    ap.add_argument("--model", default=str(S.MODEL_FILE))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plots", default=str(config.FIGURES_DIR / "replay"))
    args = ap.parse_args()
    ens = S.StepNetEnsemble.load(Path(args.model))
    recs = dataset.load_recordings(args.data, include_hw6=not args.no_hw6)
    labs, _ = L.label_all(recs, log=None)
    info = ens.info.get("label_info") or {"axis": "gz", "sign": 1, "threshold": 60.0}
    print(f"Model made {ens.created}; plain peak counter uses {info['axis']} above {info['threshold']} deg/s")
    print(f"{'recording':<34}{'kind':<15}{'counted':>8}{'network ± 95%':>18}{'in?':>5}{'peaks':>7}"
          f"{'true m':>8}{'network m ± 95%':>18}")
    for r in recs:
        hist, s = run_recording(r, ens)
        base = 2 * len(L.peaks(L.swing_signal(r.rows, info["axis"], info["sign"]), r.t, info["threshold"]))
        inside = "" if r.steps is None else ("yes" if s["steps_lo"] <= r.steps <= s["steps_hi"] else "NO")
        print(f"{r.name:<34}{r.kind:<15}{'-' if r.steps is None else r.steps:>8}"
              f"{s['steps']:>10.1f} ± {S.Z95 * s['steps_sd']:<5.1f}{inside:>5}{base:>7}"
              f"{'-' if r.distance_m is None else f'{r.distance_m:.1f}':>8}"
              f"{s['dist_m']:>10.1f} ± {S.Z95 * s['dist_sd']:<5.1f}")
        if args.plot:
            plot(r, labs.get(r.name), hist, s, Path(args.plots))
    if args.plot:
        print(f"Plots in {args.plots}")


if __name__ == "__main__":
    main()
