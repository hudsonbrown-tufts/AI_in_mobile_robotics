"""One picture that explains the 5 networks of the ensemble: figures/five_networks.jpg.

Rows:
  1. The recipe all 5 share (with network 1's real layer outputs on a real window)
  2. What makes each network different: its random start and how much it trusted each recording
  3. Each network's own answer on the same walking window (a bell curve)
  4. Putting them together: agreement on normal walking, disagreement on an unusual
     moment, and the 5 networks' cadence over part of the long walk

Uses the installed model and Hudson's recordings:

    .venv/Scripts/python HW7/make_five_nets_figure.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import config  # noqa: E402
import dataset  # noqa: E402
import labels as L  # noqa: E402
import make_figures as MF  # noqa: E402
import stepnet as S  # noqa: E402
import train_stepnet as T  # noqa: E402

COLORS = MF.MEMBER_COLORS
TRUE_RED = MF.RED
SPM = 120          # strides/s -> steps/min


def trust_weights(recs, labs, n_members):
    """The random weight each network gave each recording (same random numbers as training)."""
    train_recs = [r for r in recs if r.kind != "long"]
    sources = T.sources_for(train_recs, labs)
    names = list(dict.fromkeys(s.name for s in sources))
    out = []
    for m in range(n_members):
        rng = np.random.default_rng(T.SEED + 1000 + m)       # train_member's generator for final network m
        w = {}
        for src in sources:
            w.setdefault(src.name, []).append(float(rng.exponential()))
        out.append([float(np.mean(w[n])) for n in names])
    return names, np.array(out)


def short_name(name, recs):
    """hall_20261102_094304.csv -> 'hall 1', etc."""
    hw7 = [r.name for r in recs if r.kind == "hall"]
    if name in hw7:
        return f"walk {hw7.index(name) + 1}"
    if name.startswith("standing"):
        return "stand (HW6)" if "20261023" in name else "stand"
    if name.startswith("walking"):
        return "walk (HW6)"
    return name.split("_")[0]


def box(ax, x, y, w, h, title, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=color, ec="#333333", lw=1.2, transform=ax.transAxes))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=11, weight="bold",
            transform=ax.transAxes)
    ax.text(x + w / 2, y + 0.03, text, ha="center", va="bottom", fontsize=8.5, transform=ax.transAxes,
            color="#222222")


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, lw=1.5,
                                 color="#333333", transform=ax.transAxes))


def softplus(z):
    return np.log1p(np.exp(z))


def math_row(fig, spec, acts):
    """The equations every network computes, with plots of its two activation functions."""
    sub = spec.subgridspec(1, 3, wspace=0.22, width_ratios=[1, 1, 2.5])

    # ReLU, with network 1's real values (layers 1-3, before ReLU) on it
    a = fig.add_subplot(sub[0, 0])
    z = np.linspace(-3, 3, 200)
    a.plot(z, np.maximum(0, z), color="black", lw=2)
    pre = np.concatenate([np.ravel(acts[k]) for k in ("conv1", "conv2", "conv3")])
    off = float(np.mean(pre <= 0))
    sample = np.clip(np.random.default_rng(0).choice(pre, 400, replace=False), -3, 3)
    a.scatter(sample, np.maximum(0, sample), s=8, color=COLORS[0], alpha=0.5,
              label="network 1's real values\n(this window, layers 1-3)")
    a.set_title("ReLU (after each of the 3 conv layers)\n"
                rf"$a = \mathrm{{max}}(0,\, z)$: {off:.0%} of these values were switched off", fontsize=10)
    a.set_xlabel("z (the filter's sum)")
    a.set_ylabel("a (output)")
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=0.3)
    a.text(0, 1.2, "2   The math inside each network: all 5 compute exactly this; only the learned numbers "
                   "in W and b differ", fontsize=13, weight="bold", transform=a.transAxes)

    # softplus, with network 1's real head outputs on it
    a = fig.add_subplot(sub[0, 1])
    z = np.linspace(-5, 3, 200)
    a.plot(z, softplus(z), color="black", lw=2, label="softplus")
    a.plot(z, np.maximum(0, z), color="#999999", ls="--", lw=1, label="ReLU, for comparison")
    z1, z2 = acts["rate_head"]
    mu, sd = softplus(z1), softplus(z2) + S.SIGMA_FLOOR
    a.plot([z1], [softplus(z1)], "o", color=COLORS[0], ms=8)
    a.annotate(f"$z_1$ = {z1:.2f} → μ = {mu:.3f} strides/s = {SPM * mu:.0f} steps/min",
               (z1, softplus(z1)), (-4.9, 2.6), fontsize=8, arrowprops={"arrowstyle": "->", "color": COLORS[0]})
    a.plot([z2], [softplus(z2)], "s", color=COLORS[1], ms=7)
    a.annotate(f"$z_2$ = {z2:.2f} → σ = {sd:.3f} strides/s = ±{SPM * sd:.1f} steps/min",
               (z2, softplus(z2)), (-4.9, 1.7), fontsize=8, arrowprops={"arrowstyle": "->", "color": COLORS[1]})
    a.set_title("softplus (on both heads: μ and σ)\n"
                r"$\mathrm{softplus}(z) = \ln(1 + e^{z})$: a smooth ReLU, always > 0", fontsize=10)
    a.set_xlabel("z (the head's output)")
    a.set_ylim(-0.2, 3.1)
    a.legend(fontsize=7, loc="lower right")
    a.grid(alpha=0.3)

    # the equations, top to bottom in the order the network uses them
    a = fig.add_subplot(sub[0, 2])
    a.axis("off")
    lines = [
        ("Convolution", r"$z_{c,t} = b_c + \Sigma_i\, \Sigma_k\; W_{c,i,k}\; x_{i,\,t+k}$",
         "filter c, input channel i, kernel position k (K = 7, 5, 5)"),
        ("ReLU", r"$a_{c,t} = \mathrm{max}(0,\; z_{c,t})$", "keep a match, drop a mismatch"),
        ("Max-pool", r"$p_{c,t} = \mathrm{max}(a_{c,2t},\; a_{c,2t+1})$", "half as long"),
        ("Average", r"$f_c = (1/17)\; \Sigma_t\; a^{(3)}_{c,t}$", "32 numbers, wherever the stride was"),
        ("Heads", r"$(z_1, z_2) = W f + b$,   $\mu = \ln(1 + e^{z_1})$,   $\sigma = \ln(1 + e^{z_2}) + 0.01$",
         "for cadence, and again for speed"),
        ("Training loss", r"$\mathcal{L} = \ln \sigma + (y - \mu)^2 / (2 \sigma^2)$",
         "y = true value; cadence + speed"),
        ("Backpropagation", r"$w \leftarrow w - \eta\; \partial \mathcal{L} / \partial w$",
         "chain rule gives every gradient; Adam, η = 0.002"),
        ("Combining the 5", r"$\bar{\mu} = (1/5)\, \Sigma_k\, \mu_k$,   "
         r"$\bar{\sigma}^2 = (1/5)\, \Sigma_k\, \sigma_k^2 + (1/5)\, \Sigma_k\, (\mu_k - \bar{\mu})^2$",
         "own unsureness + disagreement"),
    ]
    y = 1.03
    for name, eq, note in lines:
        a.text(0.0, y, name, fontsize=11, weight="bold", va="top", transform=a.transAxes)
        a.text(0.19, y, eq, fontsize=13, va="top", transform=a.transAxes)
        a.text(0.19, y - 0.08, note, fontsize=9, va="top", color="#666666", style="italic",
               transform=a.transAxes)
        y -= 0.137


def main():
    model_json = json.loads(S.MODEL_FILE.read_text(encoding="utf-8"))
    ens = S.StepNetEnsemble(model_json)
    n = len(ens.members)
    recs = dataset.load_recordings(config.DATA_DIR, include_hw6=True)
    labs, _ = L.label_all(recs, log=None)
    picked = {name: (c, o) for name, c, o in MF.pick_windows(recs, labs, ens)}
    typ_c, typ_o = picked["typical"]
    unf_c, unf_o = picked["unfamiliar"]
    x_typ = typ_c[2].astype(np.float64)
    true_typ = SPM * float(typ_c[3][0])
    true_unf = SPM * float(unf_c[3][0])
    outs_typ = [m.forward(x_typ) for m in ens.members]
    outs_unf = [m.forward(unf_c[2].astype(np.float64)) for m in ens.members]
    _, acts = ens.members[0].forward(x_typ, keep=True)
    names, weights = trust_weights(recs, labs, n)
    labels_short = [short_name(nm, recs) for nm in names]

    fig = plt.figure(figsize=(19, 22), dpi=100)
    gs = fig.add_gridspec(5, n, height_ratios=[1.3, 1.45, 1.0, 1.0, 1.25], hspace=0.8, wspace=0.28,
                          left=0.04, right=0.98, top=0.945, bottom=0.04)
    fig.suptitle("The 5 step-counting networks, simply: same recipe, trained a little differently, "
                 "and their agreement tells the board how sure to be", fontsize=17, weight="bold")

    # ---- Row 1: the shared recipe ------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0, 1.02, "1   The recipe all 5 networks share (8,564 numbers each). "
                     "Pictures: network 1 on a real 2 s window of your walking", fontsize=13, weight="bold",
            transform=ax.transAxes)
    steps = [
        ("2 s of ankle motion", "6 signals: 3 accel + 3 gyro\n100 readings each\n6×100, scaled (no activation)",
         "#e8eef9"),
        ("Layer 1: 16 pattern finders", "conv 6→16, kernel 7 (0.14 s)\nactivation: ReLU, then max-pool 2\n"
         "6×100 → 16×94 → 16×47", "#fde9d9"),
        ("Layer 2: 32 finders", "conv 16→32, kernel 5 (sees 0.32 s)\nactivation: ReLU, then max-pool 2\n"
         "16×47 → 32×43 → 32×21", "#fde9d9"),
        ("Layer 3: 32 stride detectors", "conv 32→32, kernel 5 (sees 0.68 s)\nactivation: ReLU\n32×21 → 32×17",
         "#fde9d9"),
        ("Average over time", "mean of each row (no activation)\n'how much of each stride pattern'\n32×17 → 32",
         "#e6f4ea"),
        ("Two heads", "linear 32→2 (cadence), 32→2 (speed)\nactivation: softplus on μ and σ\n→ μ ± σ for each",
         "#fff4cc"),
    ]
    w, h, gap, y0 = 0.145, 0.88, 0.022, 0.02
    xs = [0.006 + i * (w + gap) for i in range(len(steps))]
    for i, (title, text, color) in enumerate(steps):
        box(ax, xs[i], y0, w, h, title, text, color)
        if i:
            arrow(ax, xs[i] - gap + 0.002, xs[i] - 0.002, y0 + h / 2)
    fig.canvas.draw()

    def inset(i):
        bb = ax.get_position()
        left = bb.x0 + (xs[i] + 0.012) * bb.width
        bottom = bb.y0 + (y0 + 0.40) * bb.height
        return fig.add_axes([left, bottom, (w - 0.024) * bb.width, 0.32 * bb.height])

    a = inset(0)
    t = np.arange(S.WINDOW) / S.SAMPLE_HZ
    raw = x_typ * np.array(ens.scales)[:, None]
    for c in range(6):
        a.plot(t, raw[c] / (np.abs(raw[c]).max() + 1e-9) + 2.2 * (5 - c), lw=0.8)
    a.set_yticks([2.2 * (5 - c) for c in range(6)], ["ax", "ay", "az", "gx", "gy", "gz"], fontsize=7)
    a.set_xticks([])
    for i, key in ((1, "relu1"), (2, "relu2"), (3, "relu3")):
        a = inset(i)
        a.imshow(acts[key], aspect="auto", cmap="magma")
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("time in the window →", fontsize=7)
    a = inset(4)
    a.bar(np.arange(32), acts["features"], color="#2e9d4f")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel("32 detectors", fontsize=7)
    a = inset(5)
    g = np.linspace(40, 160, 300)
    m0 = outs_typ[0]
    a.plot(g, MF.gauss(g, SPM * m0[0], SPM * m0[1]), color=COLORS[0], lw=2)
    a.axvline(true_typ, color=TRUE_RED, ls="--", lw=1)
    a.set_yticks([])
    a.tick_params(labelsize=7)
    a.set_xlabel("steps/min (red = true)", fontsize=7)

    # ---- Row 2: the equations and activation functions ---------------------
    math_row(fig, gs[1, :], acts)

    # ---- Row 3: what makes each different ----------------------------------
    for k in range(n):
        a = fig.add_subplot(gs[2, k])
        a.bar(range(len(names)), weights[k], color=COLORS[k])
        a.axhline(1.0, color="#999999", lw=0.8, ls="--")
        a.set_xticks(range(len(names)), labels_short, rotation=45, fontsize=8, ha="right", rotation_mode="anchor")
        a.set_ylim(0, max(3.0, weights.max() * 1.1))
        a.set_title(f"Network {k + 1}: random start #{k + 1}\nhow much it trusted each recording",
                    fontsize=10, color=COLORS[k], weight="bold")
        if k == 0:
            a.set_ylabel("trust (1 = normal)", fontsize=9)
            a.text(0, 1.42, "3   What makes each network different: a different random start, and random "
                            "'trust' in each recording while learning", fontsize=13, weight="bold",
                   transform=a.transAxes)

    # ---- Row 4: each network's answer --------------------------------------
    g = np.linspace(60, 150, 400)
    for k in range(n):
        a = fig.add_subplot(gs[3, k])
        mu, sd = SPM * outs_typ[k][0], SPM * outs_typ[k][1]
        a.fill_between(g, MF.gauss(g, mu, sd), color=COLORS[k], alpha=0.35)
        a.plot(g, MF.gauss(g, mu, sd), color=COLORS[k], lw=2)
        a.axvline(true_typ, color=TRUE_RED, ls="--", lw=1.2)
        a.set_yticks([])
        a.set_xlabel("cadence (steps/min)", fontsize=9)
        a.set_title(f"Network {k + 1} says {mu:.0f} ± {sd:.0f} steps/min\n"
                    f"speed {outs_typ[k][2]:.2f} ± {outs_typ[k][3]:.2f} m/s", fontsize=10, color=COLORS[k])
        if k == 0:
            a.text(0, 1.30, f"4   Each network's own answer on the same walking window "
                            f"(red line = true cadence, {true_typ:.0f} steps/min). "
                            "Width of the curve = how unsure that network is", fontsize=13, weight="bold",
                   transform=a.transAxes)

    # ---- Row 5: putting them together --------------------------------------
    sub = gs[4, :].subgridspec(1, 3, wspace=0.18, width_ratios=[1, 1, 1.4])
    for col, (outs, o, true, title) in enumerate((
            (outs_typ, typ_o, true_typ, "Normal walking: the 5 agree → narrow answer"),
            (outs_unf, unf_o, true_unf, "Their biggest disagreement (a start/turn) → wider answer"))):
        a = fig.add_subplot(sub[0, col])
        lo = min(SPM * (o["rate"] - 4 * o["rate_sd"]), true - 10)
        hi = max(SPM * (o["rate"] + 4 * o["rate_sd"]), true + 10)
        gg = np.linspace(max(0, lo), hi, 400)
        for k, m in enumerate(outs):
            a.plot(gg, MF.gauss(gg, SPM * m[0], SPM * m[1]), color=COLORS[k], lw=1.3)
        a.plot(gg, MF.gauss(gg, SPM * o["rate"], SPM * o["rate_sd"]), color="black", lw=2.5, label="combined")
        a.axvline(true, color=TRUE_RED, ls="--", lw=1.2, label="true")
        a.set_yticks([])
        a.set_xlabel("cadence (steps/min)", fontsize=9)
        a.legend(fontsize=8, loc="upper right")
        a.set_title(f"{title}\ncombined: {SPM * o['rate']:.0f} ± {SPM * o['rate_sd']:.0f} "
                    f"(own unsureness {SPM * o['rate_aleatoric']:.1f}, disagreement {SPM * o['rate_epistemic']:.1f})",
                    fontsize=10)
        if col == 0:
            a.text(0, 1.24, "5   Putting them together: the board averages the 5 answers; "
                            "the more they disagree, the bigger the ± it shows", fontsize=13, weight="bold",
                   transform=a.transAxes)

    a = fig.add_subplot(sub[0, 2])
    longs = [r for r in recs if r.kind == "long" and labs[r.name].ok]
    if longs:
        r = longs[0]
        x, y, _, _ = T.board_windows(r, labs[r.name])
        tt = (np.arange(len(x)) * S.HOP + S.WINDOW) / S.SAMPLE_HZ
        rates = np.array([[m.forward(xi.astype(np.float64))[0] for m in ens.members] for xi in x])
        keep = tt <= tt[0] + 40
        for k in range(n):
            a.plot(tt[keep], SPM * rates[keep, k], color=COLORS[k], lw=1.2, label=f"network {k + 1}")
        a.plot(tt[keep], SPM * y[keep, 0], color=TRUE_RED, ls="--", lw=1.2, label="true (from labels)")
        a.set_title(f"The 5 networks' cadence through the first 40 s of your long walk ({r.name})\n"
                    "they move together; where the lines spread, the board is less sure", fontsize=10)
        a.set_xlabel("seconds")
        a.set_ylabel("steps/min")
        a.legend(fontsize=7, ncol=3, loc="lower center")
        a.grid(alpha=0.3)

    fig.text(0.5, 0.012, f"Model trained {model_json['created']} on {len(model_json['trained_on'])} recordings, "
                         f"{model_json['epochs']} rounds of backpropagation each.  Combined spread² = average of the "
                         "5 networks' own spread² + spread of their answers².", ha="center", fontsize=10,
             color="#444444")
    out = config.FIGURES_DIR / "five_networks.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
