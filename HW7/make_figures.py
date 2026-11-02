"""Figures for HOW_THE_NET_WORKS.md and the data for the animation.

Uses the installed model, its training report and the recordings, so every
picture shows the real trained network on Hudson's real walks:

    figures/how_it_works.png   one real window going through the network
    figures/ensemble.png       the 5 networks on a normal, an ambiguous and an unfamiliar window
    figures/training.png       loss curves (held-out and training) and the gradient check
    figures/reliability.png    claimed vs. actual coverage of the uncertainty
    figures/long_walk.png      the long test walk: count ± range over time vs. the truth
    figures/animation_data.json  every number the animation storyboard needs (HOW_THE_NET_WORKS.md section 10)

    .venv/Scripts/python HW7/make_figures.py
    .venv/Scripts/python HW7/make_figures.py --data HW7/testing/data_synthetic --no-hw6 --model m.json --report m_report.json --out figs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import dataset  # noqa: E402
import labels as L  # noqa: E402
import replay  # noqa: E402
import stepnet as S  # noqa: E402

BLUE, GREEN, RED, ORANGE = "#1a56c4", "#2e9d4f", "#d63a2f", "#c77800"
MEMBER_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]
CHANNELS = ["ax (g)", "ay (g)", "az (g)", "gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]


def r4(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def pick_windows(recs, labs, ens):
    """A typical walking window, the most ambiguous one and the most unfamiliar one."""
    import train_stepnet as T
    cands = []
    for r in recs:
        lab = labs[r.name]
        if not lab.ok or r.kind == "standing":
            continue
        x, y, _, _ = T.board_windows(r, lab)
        for i in range(len(x)):
            cands.append((r, i, x[i], y[i]))
    outs = [S.combine(np.array([m.forward(c[2].astype(np.float64)) for m in ens.members]), ens.c_rate, ens.c_speed)
            for c in cands]
    walking = [i for i, c in enumerate(cands) if c[0].kind == "hall" and c[3][0] > 0.6]
    typical = walking[len(walking) // 3] if walking else 0
    hall_idx = [i for i, c in enumerate(cands) if c[0].kind == "hall"] or list(range(len(cands)))
    ambiguous = max(hall_idx, key=lambda i: outs[i]["rate_aleatoric"])
    unfamiliar = max(range(len(cands)), key=lambda i: outs[i]["rate_epistemic"])
    return [(name, cands[i], outs[i]) for name, i in (("typical", typical), ("ambiguous", ambiguous),
                                                       ("unfamiliar", unfamiliar))]


def window_record(cand, out, ens, keep_layers=False):
    r, i, x, y = cand
    member = [ens.members[k].forward(x.astype(np.float64), keep=keep_layers) for k in range(len(ens.members))]
    rec = {"recording": r.name, "window_index": int(i), "start_s": round(i * S.HOP / S.SAMPLE_HZ, 2),
           "label_rate_strides_per_s": round(float(y[0]), 4),
           "label_speed_m_per_s": None if np.isnan(y[1]) else round(float(y[1]), 4),
           "input_scaled": r4(x), "input_units": r4(x * np.array(ens.scales)[:, None]),
           "members": [r4(m[0] if keep_layers else m) for m in member],
           "ensemble": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}}
    if keep_layers:
        acts = member[0][1]
        rec["layers_network1"] = {k: r4(v) for k, v in acts.items() if k != "input"}
    return rec


def backprop_snapshot(cand, ens, model_json):
    """One backward pass of network 1 on the typical window: the loss and every gradient."""
    import torch
    import train_stepnet as T
    model = T.to_model({k: np.asarray(v, dtype=np.float32) for k, v in model_json["members"][0].items()})
    r, i, x, y = cand
    xb = torch.from_numpy(x[None].astype(np.float32))
    out = model(xb)
    yr = torch.tensor([float(y[0])])
    known = not np.isnan(y[1])
    ys = torch.tensor([float(y[1]) if known else 0.0])
    loss_rate = T.beta_nll(out[:, 0], out[:, 1], yr, beta=0).mean()
    loss_speed = T.beta_nll(out[:, 2], out[:, 3], ys, beta=0).mean() if known else torch.tensor(0.0)
    loss = loss_rate + loss_speed
    model.zero_grad()
    loss.backward()
    grads = {n: p.grad.numpy() for n, p in model.named_parameters()}
    weights = {n: p.detach().numpy() for n, p in model.named_parameters()}
    o = out.detach().numpy()[0]
    return {
        "explain": "Plain Gaussian NLL (beta = 0) of network 1 on the typical window, then loss.backward(). "
                   "grad = d loss / d weight for every weight; gradient descent moves each weight a small step "
                   "against its gradient (training used Adam, lr 0.002, batches of 64).",
        "outputs": {"rate_mu": float(o[0]), "rate_sd": float(o[1]), "speed_mu": float(o[2]), "speed_sd": float(o[3])},
        "labels": {"rate": float(y[0]), "speed": float(y[1]) if known else None},
        "loss": {"rate": loss_rate.item(), "speed": loss_speed.item(), "total": loss.item()},
        "grad_norm_per_layer": {n: float(np.linalg.norm(g)) for n, g in grads.items()},
        "grads": {n: r4(g) for n, g in grads.items() if not n.startswith("conv2") and not n.startswith("conv3")},
        "weights": {n: r4(w) for n, w in weights.items() if not n.startswith("conv2") and not n.startswith("conv3")},
        "lr_for_illustration": 0.002,
    }


def fig_how_it_works(win, ens, out):
    x = np.array(win["input_units"])
    acts = win["layers_network1"]
    t = np.arange(S.WINDOW) / S.SAMPLE_HZ
    fig = plt.figure(figsize=(13, 8.5), dpi=90)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1])
    a0 = fig.add_subplot(gs[0, :2])
    for c in range(3):
        a0.plot(t, x[c], lw=1, label=CHANNELS[c])
    a0b = a0.twinx()
    for c in range(3, 6):
        a0b.plot(t, x[c], lw=1, ls="--", label=CHANNELS[c])
    a0.set_title(f"1. Input: one real 2 s window ({win['recording']}, starting at {win['start_s']:.1f} s): "
                 "accel (solid, g) and gyro (dashed, deg/s)", fontsize=10)
    a0.legend(loc="upper left", fontsize=7, ncol=3)
    a0b.legend(loc="upper right", fontsize=7, ncol=3)
    a0.set_xlabel("seconds")
    a1 = fig.add_subplot(gs[1, :2])
    a1.imshow(np.array(acts["relu1"]), aspect="auto", cmap="magma", extent=[0, 2, 16, 0])
    a1.set_title("2. Layer 1: 16 filters x 94 time steps after ReLU (bright = that filter's shape is here)",
                 fontsize=10)
    a1.set_ylabel("filter")
    a2 = fig.add_subplot(gs[2, :2])
    a2.imshow(np.array(acts["relu3"]), aspect="auto", cmap="magma", extent=[0, 2, 32, 0])
    a2.set_title("3. Layer 3: 32 stride-pattern detectors x 17 time steps after ReLU", fontsize=10)
    a2.set_ylabel("detector")
    a2.set_xlabel("seconds (approximate position in the window)")
    a3 = fig.add_subplot(gs[1:, 2])
    feats = np.array(acts["features"])
    a3.barh(np.arange(32), feats, color=BLUE)
    a3.invert_yaxis()
    a3.set_title("4. Average over time:\n32 'how much of each pattern' numbers", fontsize=10)
    a4 = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0, 150, 400)
    for k, m in enumerate(win["members"]):
        a4.plot(xs, gauss(xs, 120 * m[0], 120 * m[1]), color=MEMBER_COLORS[k], lw=1, alpha=0.8)
    e = win["ensemble"]
    a4.plot(xs, gauss(xs, 120 * e["rate"], 120 * e["rate_sd"]), color="black", lw=2, label="ensemble")
    a4.axvline(120 * win["label_rate_strides_per_s"], color=RED, ls="--", lw=1, label="true (label)")
    a4.set_title("5. Output: cadence as a bell curve\n(5 networks + combined)", fontsize=10)
    a4.set_xlabel("steps/min")
    a4.set_yticks([])
    a4.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "how_it_works.png")
    plt.close(fig)


def fig_ensemble(wins, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=90)
    xs = np.linspace(0, 160, 500)
    titles = {"typical": "normal walking: they agree", "ambiguous": "most ambiguous window: all wide",
              "unfamiliar": "most unfamiliar window: they disagree"}
    for ax, (name, w) in zip(axes, wins.items()):
        for k, m in enumerate(w["members"]):
            ax.plot(xs, gauss(xs, 120 * m[0], 120 * m[1]), color=MEMBER_COLORS[k], lw=1.2, label=f"network {k + 1}")
        e = w["ensemble"]
        ax.plot(xs, gauss(xs, 120 * e["rate"], 120 * e["rate_sd"]), color="black", lw=2, label="combined")
        ax.axvline(120 * w["label_rate_strides_per_s"], color=RED, ls="--", lw=1, label="true")
        ax.set_title(f"{titles[name]}\n{w['recording']} at {w['start_s']:.1f} s", fontsize=9)
        ax.set_xlabel("cadence (steps/min)")
        ax.set_yticks([])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ensemble.png")
    plt.close(fig)


def fig_training(report, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), dpi=90)
    curve = report["heldout_nll_curve"]
    a1.plot(np.arange(1, len(curve) + 1), curve, color=BLUE)
    a1.axvline(report["epochs"], color=RED, ls="--", lw=1, label=f"chosen: {report['epochs']} epochs")
    a1.set_title("held-out loss (walks the networks never saw), averaged over folds", fontsize=10)
    a1.set_xlabel("epoch")
    a1.set_ylabel("Gaussian NLL")
    a1.legend()
    a1.grid(alpha=0.3)
    for k, c in enumerate(report["train_loss_curves"]):
        a2.plot(np.arange(1, len(c) + 1), c, color=MEMBER_COLORS[k], lw=1, label=f"network {k + 1}")
    a2.set_title("training loss (beta-NLL, augmented windows) of the 5 final networks", fontsize=10)
    a2.set_xlabel("epoch")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training.png")
    plt.close(fig)


def fig_reliability(report, out):
    rel = report["validation"]["reliability"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=90)
    claimed = [0.5, 0.6827, 0.95]
    a1.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="perfectly honest")
    names = {"windows_rate": "windows: stride rate", "windows_speed": "windows: speed",
             "walks_steps": "whole walks: steps", "walks_distance": "whole walks: distance"}
    for key, label in names.items():
        if rel.get(key):
            a1.plot(claimed, list(rel[key].values()), "o-", label=label)
    a1.set_xlabel("claimed coverage")
    a1.set_ylabel("actual share of held-out cases inside")
    a1.set_title("Is the ± honest? (leave-one-walk-out)", fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    z = np.array(report["validation"]["z_rate"]) / report["calibration"]["c_rate"]
    a2.hist(np.clip(z, -5, 5), bins=60, density=True, color=BLUE, alpha=0.6, label="held-out windows")
    xs = np.linspace(-5, 5, 300)
    a2.plot(xs, gauss(xs, 0, 1), color="black", label="ideal: standard bell curve")
    a2.set_title("window errors in units of their own spread", fontsize=10)
    a2.set_xlabel("(predicted - true) / spread")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--no-hw6", action="store_true")
    ap.add_argument("--model", default=str(S.MODEL_FILE))
    ap.add_argument("--report", default=str(config.TRAINING_REPORT))
    ap.add_argument("--out", default=str(config.FIGURES_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_json = json.loads(Path(args.model).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    ens = S.StepNetEnsemble(model_json)
    recs = dataset.load_recordings(args.data, include_hw6=not args.no_hw6)
    labs, _ = L.label_all(recs, log=None)

    picked = pick_windows(recs, labs, ens)
    wins = {name: window_record(c, o, ens, keep_layers=(name == "typical")) for name, c, o in picked}
    fig_how_it_works(wins["typical"], ens, out)
    fig_ensemble(wins, out)
    fig_training(report, out)
    fig_reliability(report, out)

    walk = None
    longs = [r for r in recs if r.kind == "long" and labs[r.name].ok]
    if longs:
        r = longs[0]
        hist, final = replay.run_recording(r, ens)
        replay.plot(r, labs[r.name], hist, final, out)
        (out / f"{r.path.stem}.png").replace(out / "long_walk.png")
        walk = {"recording": r.name, "counted_steps": r.steps, "measured_m": r.distance_m,
                "t": r4(hist["t"]), "steps": r4(hist["steps"]), "steps_sd": r4(hist["steps_sd"]),
                "dist_m": r4(hist["dist"]), "dist_sd": r4(hist["dist_sd"]),
                "cadence_spm": r4(120 * hist["rate"]), "cadence_sd": r4(120 * hist["rate_sd"]),
                "epistemic_spm": r4(120 * hist["epistemic"]),
                "true_steps_from_labels": r4(replay.true_running_steps(labs[r.name], hist["t"])),
                "final": final}

    typical = next(c for name, c, _ in picked if name == "typical")
    anim = {
        "about": "Real data for the HOW_THE_NET_WORKS.md animation storyboard (section 10). Inputs are 2 s "
                 "windows of 100 rows at 50 Hz; input_scaled = what the network sees (accel/2 g, gyro/500 deg/s). "
                 "Rates are strides/s (x120 = steps/min).",
        "model": {"created": model_json["created"], "parameters_per_network": model_json["parameters_per_network"],
                  "epochs": model_json["epochs"], "calibration": model_json["calibration"],
                  "layer_shapes": {"input": [6, 100], "conv1": [16, 94], "pool1": [16, 47], "conv2": [32, 43],
                                   "pool2": [32, 21], "conv3": [32, 17], "features": [32], "heads": [2, 2]},
                  "window_s": S.WINDOW_S, "hop_s": S.HOP / S.SAMPLE_HZ, "channels": S.NET_CHANNELS,
                  "scales": model_json["scales"]},
        "windows": wins,
        "conv1_kernels_network1": r4(model_json["members"][0]["conv1.weight"]),
        "backprop": dict(backprop_snapshot(typical, ens, model_json), gradient_check=report.get("gradient_check")),
        "training": {"heldout_nll_curve": report["heldout_nll_curve"], "train_loss_curves": report["train_loss_curves"],
                     "epochs": report["epochs"]},
        "reliability": report["validation"]["reliability"],
        "heldout_walks": report["validation"]["walks"],
        "long_walk": walk,
    }
    (out / "animation_data.json").write_text(json.dumps(anim), encoding="utf-8")
    print(f"Figures and animation_data.json written to {out}")


if __name__ == "__main__":
    main()
