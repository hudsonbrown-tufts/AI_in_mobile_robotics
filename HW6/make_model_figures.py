"""Draw the installed Part 1 and Part 2 models as JPEG figures.

    .venv/Scripts/python HW6/make_model_figures.py

Writes HW6/figures/part1_model.jpg and HW6/figures/part2_model.jpg from the
calibration the board uses (part1_calibration.json, part2_rules.json) and the
recordings in HW6/data, so rerun it after every recalibration.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import config  # noqa: E402
import classifiers as C  # noqa: E402
import dataset  # noqa: E402

OUT_DIR = config.HW6_DIR / "figures"

# Reference palette (dataviz skill): categorical slots 1-4, light surface, text inks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e6e5e1"
ACTIVITY_COLOR = {"standing": "#2a78d6", "walking": "#eb6834", "running": "#1baf7a", "biking": "#eda100"}
BOX_FILL = "#eef3fb"
BOX_EDGE = "#2a78d6"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": INK_3, "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"wrote {path}")


def flow(ax, steps, y, h=0.62, gap=0.035, fill=BOX_FILL, edge=BOX_EDGE, fontsize=9.5):
    """A row of rounded boxes joined by arrows, in axes coordinates (0..1 wide)."""
    n = len(steps)
    margin = 0.01
    w = (1 - 2 * margin - gap * (n - 1)) / n
    for i, text in enumerate(steps):
        x = margin + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    fc=fill, ec=edge, lw=1.2, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                transform=ax.transAxes, wrap=True)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.003, y + h / 2), (x + w + gap - 0.003, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=INK_2, lw=1.2,
                                         transform=ax.transAxes))


# ----------------------------------------------------------------- PART 1 ---
def part1_figure():
    cal = json.loads(C.PART1_CALIBRATION_FILE.read_text(encoding="utf-8"))
    clf = C.Part1PeakClassifier(cal["direction"], cal["order"], cal["thresholds"])
    recs = dataset.load_recordings(config.DATA_DIR)
    windows, labels, _, _ = dataset.stack_windows(recs)
    peaks = np.array([C.part1_peak(C.part1_direction_signal(w, clf.direction)) for w in windows])

    fig = plt.figure(figsize=(12, 7.2))
    fig.suptitle(f"Part 1: the peak acceleration along {clf.direction} decides the activity",
                 x=0.04, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.905, "One number per 2-second window, compared with 3 thresholds. "
             "No gyro, no other axes.", fontsize=10.5, color=INK_2)

    top = fig.add_axes([0.04, 0.70, 0.92, 0.17])
    top.axis("off")
    flow(top, ["MPU-6050\naccelerometer\n200 Hz",
               f"Keep only {clf.direction}\n(highest of every\n4 readings → 50 Hz)",
               "Last 2 s window\n(100 samples,\nnew one every 0.24 s)",
               "peak = the highest\nvalue in the window",
               "Compare with\n3 thresholds",
               "Activity, after\n3 windows in a row\nagree"], y=0.1, h=0.85)

    ax = fig.add_axes([0.12, 0.13, 0.75, 0.45])
    order = clf.order
    rows = {s: i for i, s in enumerate(order)}
    edges = [-1e9] + clf.thresholds + [1e9]
    lo_x, hi_x = min(peaks.min(), clf.thresholds[0]) - 0.3, max(peaks.max(), clf.thresholds[-1]) * 1.15
    ax.set_xscale("symlog", linthresh=0.5, linscale=1.2)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.7, len(order) - 0.3)

    # Decision regions: the band each activity owns, as a light tint of its color.
    for s, a, b in zip(order, edges, edges[1:]):
        ax.axvspan(max(a, lo_x), min(b, hi_x), color=ACTIVITY_COLOR[s], alpha=0.10, lw=0)
    # Threshold lines, labelled inside the plot; neighbours that are close get
    # their labels pushed to opposite sides of the line so they don't collide.
    for i, t in enumerate(clf.thresholds):
        ax.axvline(t, color=INK_2, ls=(0, (4, 3)), lw=1.2)
        close_prev = i > 0 and abs(t - clf.thresholds[i - 1]) < 0.3
        close_next = i < 2 and abs(clf.thresholds[i + 1] - t) < 0.3
        ha = "left" if close_prev else ("right" if close_next else "center")
        ax.text(t, len(order) - 0.45, f" T{i + 1} = {t:.2f} g ", ha=ha, va="center", fontsize=9, color=INK,
                bbox=dict(fc=SURFACE, ec=INK_3, lw=0.6, pad=2, boxstyle="round,pad=0.25"))

    rng = np.random.default_rng(0)
    for s in order:
        p = peaks[labels == s]
        y = rows[s] + rng.uniform(-0.22, 0.22, len(p))
        ax.scatter(p, y, s=14, color=ACTIVITY_COLOR[s], alpha=0.75, edgecolors=SURFACE, linewidths=0.4)
        st = cal["peak_stats_g"][s]
        ax.text(hi_x, rows[s], f"  median {st['median']:.2f} g", va="center", ha="left", fontsize=9,
                color=INK_2, clip_on=False)

    ticks = [v for v in (-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 10, 20) if lo_x <= v <= hi_x]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{s}\n(region {i + 1})" for i, s in enumerate(order)], fontsize=10)
    ax.set_xlabel(f"Peak acceleration along {clf.direction} in a 2 s window (g)  "
                  "(log scale above 0.5 g, linear below)", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Every recorded window (one dot each) and the regions the thresholds create",
                 loc="left", fontsize=11, color=INK, pad=10)

    people = ", ".join(cal.get("people", []))
    fig.text(0.04, 0.025,
             f"Calibration made {cal.get('created', '?')} from {people}'s recordings "
             f"({sum(cal['windows_per_state'].values())} windows). Accuracy {cal['accuracy']:.0%} "
             f"({cal['scored_by']}). Direction chosen as the best of +x, -x, +y, -y, +z, -z.",
             fontsize=8.5, color=INK_2)
    save(fig, "part1_model.jpg")


# ----------------------------------------------------------------- PART 2 ---
FEATURE_LABELS = {"tilt_deg": "Tilt from upright (deg)", "acc_std": "Acceleration variation (g)",
                  "acc_peak": "Acceleration peak (g)", "rot_var": "Rotation variation (deg/s)"}
QUESTION_NAMES = {"tilted": "Q1 tilted", "still": "Q2/Q1b still", "walk_or_run": "Q3 walk|run",
                  "bike_or_run": "Q1a bike|run"}


def node(ax, x, y, w, h, text, fill=BOX_FILL, edge=BOX_EDGE, bold=False, fontsize=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold" if bold else "normal")


def leaf(ax, x, y, state, w=12, h=5.5):
    if state is None:
        node(ax, x, y, w, h, "unsure:\nkeep the last state", fill="#f3f2ef", edge=INK_3, fontsize=8.5)
    else:
        color = ACTIVITY_COLOR[state]
        node(ax, x, y, w, h, state.upper(), fill=color, edge=color, bold=True, fontsize=10.5)


def arrow(ax, a, b, label=None, label_side="left"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=INK_2, lw=1.1))
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx + (-0.8 if label_side == "left" else 0.8), my, label, fontsize=9, color=INK_2,
                ha="right" if label_side == "left" else "left", va="center", fontweight="bold")


def votes_text(tests, wanted):
    parts = []
    for f, t in tests.items():
        sym = ">" if t["above"] == wanted else "<"
        unit = "°/s" if f == "rot_var" else ("°" if f == "tilt_deg" else " g")
        short = {"acc_std": "accel var", "acc_peak": "accel peak", "rot_var": "rot var", "tilt_deg": "tilt"}[f]
        val = f"{t['threshold']:.0f}" if f in ("rot_var", "tilt_deg") else f"{t['threshold']:.2f}"
        parts.append(f"{short} {sym} {val}{unit}")
    return "\n".join(parts)


def part2_figure():
    rules = json.loads(C.PART2_RULES_FILE.read_text(encoding="utf-8"))
    clf = C.Part2Rules(rules)
    q = rules["questions"]

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle("Part 2: nested questions on all 6 IMU channels (no neural network)",
                 x=0.04, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.935, "Four features per 2-second window; each multi-feature question is a majority vote "
             "(2 of 3), so one odd feature is outvoted. Thresholds sit in the middle of each gap, like Part 1.",
             fontsize=10.5, color=INK_2)

    # ---- the question tree
    ax = fig.add_axes([0.03, 0.47, 0.94, 0.44])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    t_tilt = q["tilted"]["tilt_deg"]["threshold"]
    node(ax, 50, 45, 30, 6, f"Q1  Is the shin tilted like lying down?\ntilt > {t_tilt:.0f}° from upright",
         bold=True, fontsize=9.5)
    # Question boxes are QW wide, centred at LX (upright side) and RX (lying side);
    # "yes" leaves sit at the outer edges with a clear arrow gap.
    QW, QH, LX, RX = 26, 9.5, 29, 71

    def question(x, y, title, tests, wanted):
        node(ax, x, y, QW, QH, f"{title}\n2 of 3:\n" + votes_text(tests, wanted).replace("\n", ",  "),
             fontsize=8.3)

    def side_leaf(qx, y, state, right):
        lw = 9 if state is None else 8.5
        lx = 100 - lw / 2 - 0.5 if right else lw / 2 + 0.5
        leaf(ax, lx, y, state, w=lw)
        x0 = qx + QW / 2 + 0.8 if right else qx - QW / 2 - 0.8
        x1 = lx - lw / 2 - 0.8 if right else lx + lw / 2 + 0.8
        arrow(ax, (x0, y), (x1, y))
        ax.text((x0 + x1) / 2, y + 1.5, "yes", fontsize=9, color=INK_2, ha="center", fontweight="bold")

    # lying-down branch (right)
    question(RX, 32, "Q1a  Big impacts and swing anyway?", q["bike_or_run"], "running")
    question(RX, 18, "Q1b  Is the leg still?", q["still"], "standing")
    arrow(ax, (58, 42), (RX - 2, 37.2), "yes (lying down)", "right")
    side_leaf(RX, 32, "running", True)
    arrow(ax, (RX, 27.0), (RX, 23.0), "no", "right")
    side_leaf(RX, 18, None, True)
    leaf(ax, RX, 5, "biking", w=14)
    arrow(ax, (RX, 13.0), (RX, 8.0), "no", "right")
    # upright branch (left)
    question(LX, 32, "Q2  Is the leg still?", q["still"], "standing")
    question(LX, 18, "Q3  Does it move like running?", q["walk_or_run"], "running")
    arrow(ax, (42, 42), (LX + 2, 37.2), "no (upright)", "left")
    side_leaf(LX, 32, "standing", False)
    arrow(ax, (LX, 27.0), (LX, 23.0), "no", "left")
    leaf(ax, LX - 9, 5, "running", w=12)
    leaf(ax, LX + 9, 5, "walking", w=12)
    arrow(ax, (LX - 5, 13.0), (LX - 8, 8.0), "yes", "left")
    arrow(ax, (LX + 5, 13.0), (LX + 8, 8.0), "no", "right")

    # ---- the features of every recorded window, with the thresholds that use them
    recs = dataset.load_recordings(config.DATA_DIR)
    windows, labels, _, _ = dataset.stack_windows(recs)
    feats = [C.part2_features(w, clf.up) for w in windows]
    rng = np.random.default_rng(0)
    order = ["standing", "walking", "running", "biking"]
    for k, f in enumerate(C.PART2_FEATURES):
        a = fig.add_axes([0.08 + (k % 2) * 0.47, 0.27 - (k // 2) * 0.21, 0.40, 0.14])
        vals = np.array([r[f] for r in feats])
        for i, s in enumerate(order):
            v = vals[labels == s]
            a.scatter(v, i + rng.uniform(-0.25, 0.25, len(v)), s=9, color=ACTIVITY_COLOR[s], alpha=0.7,
                      edgecolors=SURFACE, linewidths=0.3)
        # Threshold lines; labels alternate between two heights so close ones don't collide.
        lines = sorted((tests[f]["threshold"], qname) for qname, tests in q.items() if f in tests)
        for j, (t, qname) in enumerate(lines):
            a.axvline(t, color=INK_2, ls=(0, (4, 3)), lw=1)
            a.text(t, len(order) - 0.5 + 0.45 * (j % 2), f" {QUESTION_NAMES[qname]} = {t:.3g}", fontsize=7.5,
                   color=INK, ha="left", va="bottom", bbox=dict(fc=SURFACE, ec="none", pad=0.4))
        a.set_yticks(range(len(order)))
        a.set_yticklabels(order, fontsize=8.5)
        a.set_ylim(-0.6, len(order) + 0.5)
        a.set_title(FEATURE_LABELS[f], loc="left", fontsize=10)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.text(0.04, 0.012,
             f"Rules fitted {rules.get('created', '?')} from {', '.join(rules.get('people', []))}'s recordings "
             f"({sum(rules['windows_per_state'].values())} windows). Accuracy {rules['accuracy']:.1%} "
             f"({rules['scored_by']}). 'Upright' = the gravity direction while standing "
             f"({', '.join(f'{v:.2f}' for v in rules['up'])} in sensor x, y, z).",
             fontsize=8.5, color=INK_2)
    save(fig, "part2_model.jpg")


if __name__ == "__main__":
    part1_figure()
    part2_figure()
