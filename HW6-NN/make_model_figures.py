"""Draw the installed Part 1 and Part 2 models as JPEG figures.

    .venv/Scripts/python HW6/make_model_figures.py

Writes HW6/figures/part1_model.jpg and HW6/figures/part2_model.jpg from the
calibration the board uses (part1_calibration.json, part2_model.json) and the
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
def part2_figure():
    model = json.loads(C.PART2_MODEL_FILE.read_text(encoding="utf-8"))
    net = C.Part2NeuralNet(model)
    shapes = {k: np.array(v).shape for k, v in model["weights"].items()}
    params = {name: int(np.prod(shapes[f"{name}.weight"]) + np.prod(shapes[f"{name}.bias"]))
              for name in ("conv1", "conv2", "conv3", "fc")}

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle("Part 2: a 1D convolutional neural network on all 6 IMU channels",
                 x=0.04, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.905, f"{model['parameters']:,} weights, trained on the laptop with PyTorch and run on "
             "the board with numpy. Shapes are channels × time samples.", fontsize=10.5, color=INK_2)

    # Architecture: one block per layer; height ~ channels, width ~ time samples.
    arch = fig.add_axes([0.03, 0.45, 0.94, 0.40])
    arch.set_xlim(0, 100)
    arch.set_ylim(0, 44)
    arch.axis("off")
    layers = [
        ("Input", 6, 100, "6 × 100\n2 s window\naccel ÷ 2 g\ngyro ÷ 500 °/s", None),
        ("Conv 1", 16, 94, f"16 filters\nkernel 7 (0.14 s)\n+ ReLU\n{params['conv1']:,} weights", "16 × 94"),
        ("Max pool", 16, 47, "keep the larger\nof each pair", "16 × 47"),
        ("Conv 2", 32, 43, f"32 filters\nkernel 5\n+ ReLU\n{params['conv2']:,} weights", "32 × 43"),
        ("Max pool", 32, 21, "keep the larger\nof each pair", "32 × 21"),
        ("Conv 3", 32, 17, f"32 filters\nkernel 5\n+ ReLU\n{params['conv3']:,} weights", "32 × 17"),
        ("Average\nover time", 32, 1, "32 numbers:\nhow much of each\npattern is present", "32"),
        ("Linear\n+ softmax", 4, 1, f"4 probabilities\n{params['fc']:,} weights", "4"),
    ]
    slot = 98 / len(layers)
    base = 14
    centers = [1.0 + slot * (i + 0.5) for i in range(len(layers))]
    widths = [max(1.2, t / 100 * (slot - 3.5)) for _, _, t, _, _ in layers]
    for i, (name, ch, t, note, shape) in enumerate(layers):
        h = 4 + ch * 0.75
        w, cx = widths[i], centers[i]
        fill = "#fdf0e9" if name.startswith("Linear") else ("#f3f2ef" if "pool" in name or "Average" in name
                                                            else BOX_FILL)
        edge = "#eb6834" if name.startswith("Linear") else (INK_3 if fill == "#f3f2ef" else BOX_EDGE)
        arch.add_patch(FancyBboxPatch((cx - w / 2, base), w, h, boxstyle="round,pad=0.15,rounding_size=0.6",
                                      fc=fill, ec=edge, lw=1.2))
        arch.text(cx, base + h + 1.2, name, ha="center", va="bottom", fontsize=10, fontweight="bold")
        if shape:
            arch.text(cx, base + h / 2, shape, ha="center", va="center", fontsize=8.5, color=INK_2,
                      rotation=90 if w < 3 else 0)
        arch.text(cx, base - 1.5, note, ha="center", va="top", fontsize=8.5, color=INK_2)
        if i < len(layers) - 1:
            arch.add_patch(FancyArrowPatch((cx + w / 2 + 0.6, base + 6),
                                           (centers[i + 1] - widths[i + 1] / 2 - 0.6, base + 6),
                                           arrowstyle="-|>", mutation_scale=11, color=INK_2, lw=1.1))
    arch.text(1, 42, "Layer sizes", fontsize=11, fontweight="bold")

    # One real window of each activity: what goes in, and what comes out.
    recs = dataset.load_recordings(config.DATA_DIR)
    examples = {}
    for s in C.STATES:
        r = next((r for r in recs if r.label == s), None)
        if r is not None:
            w = r.windows()
            examples[s] = w[len(w) // 2]
    t = np.arange(C.WINDOW) / C.SAMPLE_HZ

    ex_act = "walking" if "walking" in examples else next(iter(examples))
    win = examples[ex_act]
    ax_a = fig.add_axes([0.06, 0.15, 0.27, 0.24])
    ax_g = fig.add_axes([0.38, 0.15, 0.27, 0.24])
    axis_colors = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
    for axis in "xyz":
        ax_a.plot(t, win[:, C.COL[f"a{axis}"]], color=axis_colors[axis], lw=1.6, label=f"a{axis}")
        ax_g.plot(t, win[:, C.COL[f"g{axis}"]], color=axis_colors[axis], lw=1.6, label=f"g{axis}")
    for a, title, unit in ((ax_a, f"Input example ({ex_act}): acceleration", "g"),
                           (ax_g, f"Input example ({ex_act}): rotation", "°/s")):
        a.set_title(title, loc="left", fontsize=10.5, pad=16)
        a.set_xlabel("time in the window (s)")
        a.set_ylabel(unit)
        a.grid(color=GRID, lw=0.8)
        a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8.5, frameon=False, ncol=3,
                 borderaxespad=0.2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    # Output: the network's probabilities for one real window of each activity.
    ax_p = fig.add_axes([0.73, 0.15, 0.24, 0.24])
    names = [s for s in C.STATES if s in examples]
    probs = np.array([net.classify(examples[s])[1] for s in names])
    ax_p.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(net.classes)):
            v = probs[i, j]
            ax_p.text(j, i, f"{v:.0%}" if v >= 0.005 else "0", ha="center", va="center", fontsize=9,
                      color="white" if v > 0.6 else INK)
    ax_p.set_xticks(range(len(net.classes)))
    ax_p.set_xticklabels(net.classes, fontsize=8.5, rotation=20)
    ax_p.set_yticks(range(len(names)))
    ax_p.set_yticklabels([f"real {s}" for s in names], fontsize=8.5)
    ax_p.set_xlabel("network's probability for each activity")
    ax_p.set_title("Output for one real window of each", loc="left", fontsize=10.5, pad=16)
    for side in ax_p.spines.values():
        side.set_visible(False)

    fig.text(0.04, 0.015,
             f"Model made {model.get('created', '?')} from {', '.join(model.get('people', []))}'s recordings, "
             f"{model.get('epochs', '?')} epochs. Validation accuracy {model.get('validation_accuracy', 0):.0%} "
             f"({model.get('validation', '?')}). A window counts only if the top probability is ≥ "
             f"{C.PART2_MIN_CONFIDENCE:.0%}.", fontsize=8.5, color=INK_2)
    save(fig, "part2_model.jpg")


if __name__ == "__main__":
    part1_figure()
    part2_figure()
