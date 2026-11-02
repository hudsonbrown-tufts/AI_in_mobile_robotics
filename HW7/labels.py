"""Stride times for every recording: the labels the network learns from.

Hudson types in how many steps a walk took. The network works on 2 s windows,
so that one total is turned into *when* each stride happened:

1. The ankle sensor only feels the right leg. The shin swings forward once per
   right-foot step (one stride = 2 steps), which makes one big, clean peak on
   the gyro axis across the shin. That axis (and its sign) is found
   automatically: the gyro axis that varies the most while walking.
2. Peaks are found with a threshold and at least MIN_STRIDE_S apart. The
   threshold is tuned until the number of peaks equals steps / 2 (+-1 when the
   count is odd: which foot started).
3. A walk where no threshold gets within 1 stride is flagged and not used.
4. A check plot per walk goes to figures/labels/.

The stride "phase" then goes up by 1 from one stride to the next (linear in
between, plus half a stride before the first and after the last), scaled so
the whole walk adds up to exactly the typed steps / 2. The number of strides in
any window = phase(end) - phase(start), a smooth, fractional label.

Walks nobody counted (HW6's walking recording) use the median threshold of the
counted walks, and only the part between their first and last stride is used.

    .venv/Scripts/python HW7/labels.py                 # label everything + plots
    .venv/Scripts/python HW7/labels.py --data HW7/testing/data_synthetic --no-hw6
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import config
import dataset
import stepnet as S

MIN_STRIDE_S = 0.5            # two strides can't be closer than this (walking is ~1.0-1.2 s)
SMOOTH_ROWS = 5               # 0.1 s moving average before peak finding
GYRO_AXES = ["gx", "gy", "gz"]
DEFAULT_PERIOD_S = 1.1        # stride period used when a walk has a single stride


@dataclass
class Labels:
    stride_times: np.ndarray       # seconds (recording time)
    total_strides: float           # what the phase adds up to (typed steps / 2)
    knots_t: np.ndarray            # phase(t) = interp(t, knots_t, knots_p)
    knots_p: np.ndarray
    stride_length_m: float | None  # distance / strides (None: unknown)
    valid: tuple | None            # (t_start, t_end): only windows inside are usable; None = all
    threshold: float | None
    ok: bool
    note: str

    def phase(self, t):
        if len(self.knots_t) == 0:
            return np.zeros_like(np.asarray(t, dtype=float))
        return np.interp(t, self.knots_t, self.knots_p)


def swing_signal(rows, axis, sign):
    """The shin-swing gyro trace (deg/s), smoothed, positive at mid-swing."""
    g = sign * rows[:, S.COL[axis]]
    return np.convolve(g, np.ones(SMOOTH_ROWS) / SMOOTH_ROWS, mode="same")


def pick_axis(rows):
    """(axis, sign) of the gyro channel that moves the most; sign makes the big swing peak positive."""
    stds = [rows[:, S.COL[a]].std() for a in GYRO_AXES]
    axis = GYRO_AXES[int(np.argmax(stds))]
    g = rows[:, S.COL[axis]]
    sign = 1 if np.percentile(g, 99) >= -np.percentile(g, 1) else -1
    return axis, sign


def peaks(sig, t, threshold):
    """Stride times: peaks above threshold, at least MIN_STRIDE_S apart."""
    idx, _ = find_peaks(sig, height=threshold, distance=int(MIN_STRIDE_S * S.SAMPLE_HZ))
    return t[idx]


def tune_threshold(sig, t, target):
    """Threshold whose peak count is closest to target (the middle of the best range)."""
    top = np.percentile(sig, 99.5)
    candidates = np.linspace(15.0, max(16.0, 0.95 * top), 300)
    counts = np.array([len(peaks(sig, t, th)) for th in candidates])
    err = np.abs(counts - target)
    best = candidates[err == err.min()]
    return float(np.median(best)), int(counts[err == err.min()][0])


def make_phase(stride_times, total):
    """Knots of the stride phase: 0.5 at the first stride, +1 per stride, total at the end."""
    k = len(stride_times)
    if k == 0:
        return np.array([]), np.array([])
    period_first = stride_times[1] - stride_times[0] if k > 1 else DEFAULT_PERIOD_S
    period_last = stride_times[-1] - stride_times[-2] if k > 1 else DEFAULT_PERIOD_S
    knots_t = np.concatenate([[stride_times[0] - period_first / 2], stride_times,
                              [stride_times[-1] + period_last / 2]])
    knots_p = np.concatenate([[0.0], np.arange(k) + 0.5, [float(k)]])
    return knots_t, knots_p * (total / k)          # scaled so the walk adds up to `total`


def label_counted(rec, axis, sign):
    """Labels for a walk with a typed step count."""
    t = rec.t
    sig = swing_signal(rec.rows, axis, sign)
    target = rec.steps / 2
    threshold, count = tune_threshold(sig, t, target)
    strides = peaks(sig, t, threshold)
    ok = abs(len(strides) - target) <= 1
    note = (f"typed {rec.steps} steps = {target:g} strides, found {len(strides)} "
            f"(threshold {threshold:.0f} deg/s)")
    if not ok:
        note += ": TOO FAR OFF, not used (recount or redo this walk)"
    knots_t, knots_p = make_phase(strides, target)
    length = rec.distance_m / target if rec.distance_m and target > 0 else None
    return Labels(strides, target, knots_t, knots_p, length, None, threshold, ok, note)


def label_uncounted(rec, axis, sign, threshold):
    t = rec.t
    sig = swing_signal(rec.rows, axis, sign)
    strides = peaks(sig, t, threshold)
    if len(strides) < 3:
        return Labels(strides, 0, np.array([]), np.array([]), None, None, threshold, False,
                      f"only {len(strides)} strides found: not used")
    knots_t, knots_p = make_phase(strides, len(strides))
    return Labels(strides, float(len(strides)), knots_t, knots_p, None, (strides[0], strides[-1]),
                  threshold, True, f"not counted: {len(strides)} strides found with the counted walks' "
                                   f"threshold {threshold:.0f} deg/s; only used between the first and last")


def label_standing():
    return Labels(np.array([]), 0.0, np.array([]), np.array([]), 0.0, None, None, True, "standing: 0 steps")


def label_all(recs, log=print):
    """Labels for every recording -> (dict name -> Labels, info about the axis and threshold)."""
    counted = [r for r in recs if r.kind in ("hall", "long")]
    walking_rows = [r.rows for r in counted] or [r.rows for r in recs if r.kind == "walk_uncounted"]
    if walking_rows:
        axis, sign = pick_axis(np.vstack(walking_rows))
    else:
        axis, sign = "gz", 1
    out = {}
    for r in recs:
        if r.kind == "standing":
            out[r.name] = label_standing()
        elif r.kind in ("hall", "long"):
            out[r.name] = label_counted(r, axis, sign)
    tuned = [lab.threshold for r, lab in ((r, out[r.name]) for r in counted) if lab.ok]
    threshold = float(np.median(tuned)) if tuned else 60.0
    for r in recs:
        if r.kind == "walk_uncounted":
            out[r.name] = label_uncounted(r, axis, sign, threshold)
    info = {"axis": axis, "sign": sign, "threshold": round(threshold, 1)}
    if log:
        log(f"Stride labels: shin-swing axis {'+' if sign > 0 else '-'}{axis}, "
            f"median threshold {threshold:.0f} deg/s")
        for r in recs:
            log(f"  {r.name:<36} {'ok ' if out[r.name].ok else 'BAD'} {out[r.name].note}")
    return out, info


def plot_labels(rec, lab, info, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = swing_signal(rec.rows, info["axis"], info["sign"])
    t = rec.t
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=90)
    ax.plot(t, sig, lw=1, color="#1a56c4")
    if lab.threshold is not None:
        ax.axhline(lab.threshold, color="#999999", ls="--", lw=0.8, label=f"threshold {lab.threshold:.0f} deg/s")
    if len(lab.stride_times):
        ax.plot(lab.stride_times, np.interp(lab.stride_times, t, sig), "v", color="#d63a2f",
                label=f"{len(lab.stride_times)} strides")
    ax.set_title(f"{rec.name}: {lab.note}", fontsize=9, color="black" if lab.ok else "#d63a2f")
    ax.set_xlabel("seconds")
    ax.set_ylabel(f"{'+' if info['sign'] > 0 else '-'}{info['axis']} (deg/s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec.path.stem}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--no-hw6", action="store_true", help="don't add HW6's recordings")
    ap.add_argument("--plots", default=str(config.FIGURES_DIR / "labels"))
    args = ap.parse_args()
    recs = dataset.load_recordings(args.data, include_hw6=not args.no_hw6, log=print)
    if not recs:
        raise SystemExit(f"No recordings in {args.data}")
    labs, info = label_all(recs)
    for r in recs:
        if r.kind != "standing":
            plot_labels(r, labs[r.name], info, Path(args.plots))
    print(f"Check plots in {args.plots}")


if __name__ == "__main__":
    main()
