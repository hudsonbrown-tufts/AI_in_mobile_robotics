"""The HW7 explainer animation (ANIMATION_SCRIPT.md): voice + captions, real model and data.

Steps:
  1. prepare(): every number the scenes need, from the installed model and Hudson's recordings
  2. narration: each sentence is spoken by the Windows voice (System.Speech) into its own WAV, so
     each caption appears exactly while its sentence is spoken
  3. each scene is drawn frame by frame with matplotlib (1920x1080, 30 fps) and encoded with
     ffmpeg, several scenes in parallel
  4. the scenes are joined, the narration is added, and captions are also written as an .srt

    .venv/Scripts/python HW7/animation/make_animation.py            # -> HW7/animation/step_counter_explained.mp4
    .venv/Scripts/python HW7/animation/make_animation.py --scenes 3 7 --fps 10   # quick preview of some scenes

Needs the imageio-ffmpeg package (bundled ffmpeg) and Windows' built-in voices.
"""

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
import textwrap
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW7 = HERE.parent
sys.path.insert(0, str(HW7))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

BUILD = HERE / "build"
OUT = HERE / "step_counter_explained.mp4"
W_PX, H_PX = 1920, 1080
VOICE = "Microsoft Zira Desktop"
VOICE_RATE = 3                     # System.Speech: -10 (slow) .. 10 (fast)
GAP_S = 0.2                        # pause between sentences
TAIL_S = 0.5                        # pause at the end of each scene

DARK = "#1d2433"
BLUE, GREEN, RED, ORANGE, PURPLE = "#1a56c4", "#2e9d4f", "#d63a2f", "#c77800", "#7b4fb0"
NET = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]
CH_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
SPM = 120

# ---------------------------------------------------------------- the script --
# Each scene: title, its sentences as (spoken, caption), and "why" cards as
# (from fraction of the scene, title, text).
SCENES = [
    {"title": "How the ankle step counter works",
     "lines": [("How the step counter works, from the network to back propagation.",
                "How the step counter works, from the network to backpropagation.")],
     "why": []},
    {"title": "1 · The sensor",
     "lines": [("The ankle sensor feels one leg: each swing is a stride, or two steps.",
                "The ankle sensor feels one leg: each swing is a stride, or 2 steps."),
               ("It sends six signals, fifty times a second.", "It sends 6 signals, 50 times a second.")],
     "why": []},
    {"title": "2 · The 2-second window",
     "lines": [("Every quarter second, the network reads the last two seconds, scaled to the same size.",
                "Every 0.24 s, the network reads the last 2 seconds, scaled to the same size.")],
     "why": [(0.0, "Scaling, no activation", "Accel ÷ 2 g and gyro ÷ 500 °/s puts all six signals at about −1…+1, "
                                             "so none drowns out the others. These are just measurements, so "
                                             "there is no activation yet.")]},
    {"title": "3 · Layer 1: convolution → ReLU → max-pool",
     "lines": [("Layer one slides sixteen filters along the signals, since a stride looks the same anywhere.",
                "Layer 1 slides 16 filters along the signals, since a stride looks the same anywhere."),
               ("ReLU keeps positive matches and zeroes the rest. Without it, the layers would collapse into one.",
                "ReLU keeps positive matches and zeroes the rest. Without it, the layers collapse into one."),
               ("Max pooling halves the length.", "Max-pool halves the length.")],
     "why": [(0.0, "Convolution", "One small filter is reused at every position in the window, so it finds a heel "
                                  "spike or a swing anywhere. 688 weights instead of tens of thousands."),
             (0.42, "ReLU: a = max(0, z)", "\"Is this pattern here? How much, or not at all.\" Without a "
                                          "non-linear step, three layers equal one plain filter. Its slope is 1 "
                                          "when on, so gradients pass back without shrinking."),
             (0.72, "Max-pool", "Keep the stronger of each pair of neighbours: half as long, so the next layer "
                                "sees twice as far back, and small timing shifts don't matter.")]},
    {"title": "4 · Layers 2 and 3: the same recipe, seeing farther",
     "lines": [("Layers two and three repeat this, each seeing farther back.",
                "Layers 2 and 3 repeat this, each seeing farther back: about 0.7 s by layer 3.")],
     "why": [(0.0, "Why stack layers", "Each layer combines the last one's patterns into longer ones: spike → "
                                       "'swing then heel strike' → most of a stride. Pooling doubles how far back "
                                       "each layer can see.")]},
    {"title": "5 · Average over time",
     "lines": [("Averaging over time keeps how much stride is present, not where.",
                "Averaging over time keeps HOW MUCH stride is present, not WHERE: exactly what a rate needs.")],
     "why": [(0.0, "Why an average", "The step rate shouldn't depend on where in the window a stride lands. "
                                     "The average throws away 'where' and keeps 'how much'. No activation needed.")]},
    {"title": "6 · Two heads: a guess and an uncertainty",
     "lines": [("Two heads give a guess, and an uncertainty.",
                "Two heads give a guess μ and an uncertainty σ, for cadence and for walking speed."),
               ("Softplus keeps both positive.", "Softplus keeps both positive.")],
     "why": [(0.0, "Softplus: ln(1 + eᶻ)", "A cadence and a spread can't be negative. Softplus is a smooth ReLU "
                                          "that never goes flat, so learning never gets stuck. +0.01 keeps σ "
                                          "from ever being 0.")]},
    {"title": "7 · Training in PyTorch: backpropagation",
     "lines": [("Training runs in PyTorch. At first, the guess is wrong.",
                "Training runs in PyTorch. At first, the guess is wrong."),
               ("The loss punishes confident mistakes, and needless vagueness.",
                "The loss punishes confident mistakes AND needless vagueness."),
               ("Backward runs the chain rule in reverse, giving every weight a gradient.",
                "loss.backward() runs the chain rule in reverse, giving every weight a gradient."),
               ("Adam nudges each weight downhill, sixty-six times through the walks.",
                "Adam nudges each weight downhill, 66 times through all the walks.")],
     "why": [(0.0, "The loss", "L = ln σ + (y − μ)² / (2σ²). A small σ with a big error costs a lot; a big σ "
                               "costs too. So σ learns to match the network's real mistakes."),
             (0.45, "Backpropagation", "PyTorch recorded every calculation in the forward pass. The chain rule, "
                                       "run backwards through it, says how much each of the 8,564 weights "
                                       "caused the error.")]},
    {"title": "8 · Why 5 networks",
     "lines": [("Why five networks? One can't tell when it's out of its depth.",
                "Why 5 networks? One network can't tell when it's out of its depth."),
               ("Five copies start differently, and trust each walk differently.",
                "5 copies start from different random weights and trust each walk differently."),
               ("Where the data is clear, they agree. Where it isn't, they disagree, and the uncertainty grows.",
                "Where the data is clear, they agree. Where it isn't, they disagree, and the ± grows.")],
     "why": [(0.0, "Why 5", "One network's σ only knows 'this window is ambiguous'. On motion it never saw it can "
                            "be confidently wrong. Disagreement between copies catches that. 5 is enough to "
                            "measure a spread and still fast on the board.")]},
    {"title": "9 · From windows to a step count ±",
     "lines": [("The board adds cadence times time, and the uncertainty adds up too.",
                "Every 0.24 s the board adds cadence × time, and the uncertainty adds up too."),
               ("COUNT_SENTENCE", "COUNT_CAPTION")],
     "why": []},
    {"title": "10 · Laptop → board",
     "lines": [("PyTorch trains on the laptop; the board just runs the frozen network.",
                "PyTorch trains on the laptop. The board just runs the frozen network (numpy, no backprop).")],
     "why": []},
]


def number_words(n):
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
           "sixteen seventeen eighteen nineteen".split()
    tens = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else "-" + ones[n % 10])
    rest = n % 100
    return ones[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))


# ---------------------------------------------------------------- data --------
def prepare():
    import torch
    import config
    import dataset
    import labels as L
    import make_figures as MF
    import make_five_nets_figure as FN
    import replay
    import stepnet as S
    import train_stepnet as T

    model_json = json.loads(S.MODEL_FILE.read_text(encoding="utf-8"))
    report = json.loads(config.TRAINING_REPORT.read_text(encoding="utf-8"))
    ens = S.StepNetEnsemble(model_json)
    recs = dataset.load_recordings(config.DATA_DIR, include_hw6=True)
    labs, _ = L.label_all(recs, log=None)
    picked = {name: (c, o) for name, c, o in MF.pick_windows(recs, labs, ens)}
    (rec, idx, x_typ, y_typ), ens_typ = picked["typical"]
    (_, _, x_unf, y_unf), ens_unf = picked["unfamiliar"]
    x_typ = x_typ.astype(np.float64)
    out1, acts = ens.members[0].forward(x_typ, keep=True)
    acts = {k: np.asarray(v) for k, v in acts.items()}
    pre = np.concatenate([np.ravel(acts[k]) for k in ("conv1", "conv2", "conv3")])
    k_best = int(np.argmax(acts["relu1"].max(axis=1)))

    # the stretch of the recording around the typical window (scenes 1 and 2)
    start = idx * S.HOP
    a, b = max(0, start - 3 * S.SAMPLE_HZ), min(len(rec.rows), start + S.WINDOW + 3 * S.SAMPLE_HZ)
    seg = rec.rows[a:b][:, :6]

    stand = next(r for r in recs if r.kind == "standing")
    xs, _, _, _ = T.board_windows(stand, labs[stand.name])
    _, acts_stand = ens.members[0].forward(xs[len(xs) // 2].astype(np.float64), keep=True)

    # network 1 before training: the same random start train_member used
    torch.manual_seed(T.SEED + 1000)
    init = T.StrideNet()
    xb = torch.from_numpy(x_typ[None].astype(np.float32))
    out = init(xb)
    yr, ys = torch.tensor([float(y_typ[0])]), torch.tensor([float(y_typ[1])])
    loss = T.beta_nll(out[:, 0], out[:, 1], yr, beta=0).mean() + T.beta_nll(out[:, 2], out[:, 3], ys, beta=0).mean()
    init.zero_grad()
    loss.backward()
    grads = {n: p.grad.numpy().copy() for n, p in init.named_parameters()}
    groups = {"layer 1": ("conv1",), "layer 2": ("conv2",), "layer 3": ("conv3",), "heads": ("rate_head", "speed_head")}
    grad_norm = {g: float(math.sqrt(sum(np.sum(v ** 2) for n, v in grads.items() if n.split(".")[0] in pre_)))
                 for g, pre_ in groups.items()}
    o = out.detach().numpy()[0]
    feats0 = torch.relu(init.conv3(init.pool(torch.relu(init.conv2(init.pool(torch.relu(init.conv1(xb))))))))
    feats0 = feats0.mean(dim=2).detach().numpy()[0]
    z1 = float(init.rate_head(torch.from_numpy(feats0[None])).detach().numpy()[0, 0])
    gw = grads["rate_head.weight"][0]
    j = int(np.argmax(np.abs(gw)))
    dl_dmu = -(float(y_typ[0]) - o[0]) / o[1] ** 2
    sig = 1 / (1 + math.exp(-z1))
    chain = {"j": j, "dl_dmu": dl_dmu, "sigmoid": sig, "feature": float(feats0[j]), "grad": float(gw[j]),
             "product": dl_dmu * sig * float(feats0[j]),
             "w": init.rate_head.weight[0, j].item(),
             "w_new_sgd": init.rate_head.weight[0, j].item() - 0.002 * float(gw[j])}

    # numpy (board) vs PyTorch on the typical window
    final1 = T.to_model({k: np.asarray(v, dtype=np.float32) for k, v in model_json["members"][0].items()})
    with torch.no_grad():
        ref = final1(xb).numpy()[0]
    numpy_diff = float(np.abs(ref - np.array(out1)).max())

    names, weights = FN.trust_weights(recs, labs, len(ens.members))
    longs = [r for r in recs if r.kind == "long" and labs[r.name].ok]
    walk = None
    if longs:
        lw = longs[0]
        hist, final = replay.run_recording(lw, ens)
        walk = {"t": hist["t"], "steps": hist["steps"], "sd": hist["steps_sd"],
                "true": replay.true_running_steps(labs[lw.name], hist["t"]), "counted": lw.steps, "final": final}

    return {
        "x_typ": x_typ, "raw_typ": x_typ * np.array(ens.scales)[:, None], "true_typ": SPM * float(y_typ[0]),
        "true_speed": float(y_typ[1]), "seg": seg, "seg_win_start": (start - a) / S.SAMPLE_HZ,
        "acts": acts, "k_best": k_best, "kernel": np.asarray(model_json["members"][0]["conv1.weight"])[k_best],
        "relu_off": float(np.mean(pre <= 0)), "feat_stand": np.asarray(acts_stand["features"]),
        "members_typ": np.array([m.forward(x_typ) for m in ens.members]), "ens_typ": ens_typ,
        "members_unf": np.array([m.forward(x_unf.astype(np.float64)) for m in ens.members]), "ens_unf": ens_unf,
        "true_unf": SPM * float(y_unf[0]), "out1": np.array(out1),
        "init_out": o, "init_loss": float(loss), "grad_norm": grad_norm, "chain": chain,
        "loss_curve": np.array(report["train_loss_curves"][0]), "epochs": int(model_json["epochs"]),
        "trust_names": [FN.short_name(n, recs) for n in names], "trust": weights,
        "walk": walk, "numpy_diff": numpy_diff, "created": model_json["created"],
        "n_weights": int(model_json["parameters_per_network"]),
        "font": sketch_font(),
    }


def sketch_font():
    """The sketch's 5x7 digit font (column bytes, bit 0 = top), for the LED matrix in scene 10."""
    text = (HW7 / "step-counter-wearable" / "sketch" / "sketch.ino").read_text(encoding="utf-8")
    block = text[text.index("FONT_DIGITS[10][5]"):text.index("FONT_LETTERS")]
    nums = [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]{2})", block)]
    return [nums[i * 5:(i + 1) * 5] for i in range(10)]


# ---------------------------------------------------------------- narration ---
def synthesize(sentences, folder):
    """Speak each sentence into its own WAV with System.Speech. -> list of durations (s)."""
    folder.mkdir(parents=True, exist_ok=True)
    items = [{"path": str(folder / f"s{i:02d}.wav"), "text": s} for i, s in enumerate(sentences)]
    (folder / "items.json").write_text(json.dumps(items), encoding="utf-8")
    ps = folder / "speak.ps1"
    ps.write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$s.SelectVoice('{VOICE}')\n"
        f"$s.Rate = {VOICE_RATE}\n"
        f"$items = Get-Content -Raw -Encoding UTF8 '{folder / 'items.json'}' | ConvertFrom-Json\n"
        "foreach ($it in $items) { $s.SetOutputToWaveFile($it.path); $s.Speak($it.text) }\n"
        "$s.SetOutputToNull()\n", encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)], check=True)
    durs = []
    for it in items:
        durs.append(trim_silence(it["path"]))
    return [it["path"] for it in items], durs


def trim_silence(path, threshold=300, keep_s=0.04):
    """Cut the silence the voice leaves before and after each sentence. -> new length (s)."""
    with wave.open(path) as w:
        params = w.getparams()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    loud = np.where(np.abs(data) > threshold)[0]
    keep = int(keep_s * params.framerate)
    if len(loud):
        data = data[max(0, loud[0] - keep):loud[-1] + keep]
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(data.tobytes())
    return len(data) / params.framerate


def build_timeline(scenes, durs, min_len):
    """Per scene: (start, length, [(sentence start within scene, caption)])."""
    timeline, k, t0 = [], 0, 0.0
    for i, sc in enumerate(scenes):
        caps, t = [], 0.4
        for _, caption in sc["lines"]:
            caps.append((t, caption))
            t += durs[k] + GAP_S
            k += 1
        length = max(t - GAP_S + TAIL_S, min_len[i])
        timeline.append((t0, length, caps))
        t0 += length
    return timeline


def write_audio(paths, scenes, timeline, out):
    with wave.open(paths[0]) as w:
        params = w.getparams()
    rate, width, ch = params.framerate, params.sampwidth, params.nchannels
    total = timeline[-1][0] + timeline[-1][1]
    buf = bytearray(int(total * rate) * width * ch + 16)
    k = 0
    for (start, _, caps) in timeline:
        for t, _ in caps:
            with wave.open(paths[k]) as w:
                data = w.readframes(w.getnframes())
            pos = int((start + t) * rate) * width * ch
            buf[pos:pos + len(data)] = data
            k += 1
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(bytes(buf[:int(total * rate) * width * ch]))


def write_srt(timeline, path):
    def ts(s):
        h, m = int(s // 3600), int(s % 3600 // 60)
        return f"{h:02d}:{m:02d}:{s % 60:06.3f}".replace(".", ",")
    lines, n = [], 1
    for start, length, caps in timeline:
        for i, (t, cap) in enumerate(caps):
            end = caps[i + 1][0] if i + 1 < len(caps) else length
            lines += [str(n), f"{ts(start + t)} --> {ts(start + end - 0.05)}", cap, ""]
            n += 1
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- drawing -----
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(u, a, b):
    return ease((u - a) / max(1e-9, b - a))


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def base(fig, sc, t, length, caps):
    fig.clf()
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.925, sc["title"], fontsize=34, weight="bold", color=DARK, va="center")
    cap = next((c for s, c in reversed(caps) if t >= s - 0.05), caps[0][1] if caps else "")
    fig.add_artist(Rectangle((0, 0), 1, 0.12, transform=fig.transFigure, color=DARK, zorder=5))
    fig.text(0.5, 0.06, textwrap.fill(cap, 92), ha="center", va="center", color="white", fontsize=25,
             zorder=6, linespacing=1.3)
    cards = [c for c in sc["why"] if t / length >= c[0]]
    if cards:
        _, title, body = cards[-1]
        fig.add_artist(FancyBboxPatch((0.695, 0.17), 0.28, 0.69, boxstyle="round,pad=0.008,rounding_size=0.015",
                                      transform=fig.transFigure, fc="#fff8e1", ec=ORANGE, lw=2.5))
        fig.text(0.715, 0.825, "WHY?", fontsize=19, weight="bold", color=ORANGE, va="top")
        fig.text(0.715, 0.775, textwrap.fill(title, 26), fontsize=23, weight="bold", color=DARK, va="top")
        fig.text(0.715, 0.66, textwrap.fill(body, 33), fontsize=18.5, color="#333333", va="top", linespacing=1.4)
    return bool(cards)


def stacked(ax, x, t=None, colors=CH_COLORS, labels=True, lw=1.6, alpha=1.0):
    """Six channels stacked, each scaled to its own range."""
    t = np.arange(x.shape[1]) if t is None else t
    for c in range(6):
        v = x[c] - x[c].mean()
        v = v / (np.abs(v).max() + 1e-9)
        ax.plot(t, v + 2.3 * (5 - c), color=colors[c], lw=lw, alpha=alpha)
    if labels:
        ax.set_yticks([2.3 * (5 - c) for c in range(6)], CHANNELS, fontsize=16)
    else:
        ax.set_yticks([])
    ax.set_ylim(-1.4, 2.3 * 5 + 1.4)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def scene0(fig, t, T, D):
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.7])
    ax.axis("off")
    g = D["seg"][:, 5]
    n = len(g)
    off = int(t * 50) % n
    v = np.roll(g, -off)
    ax.plot(np.arange(n), v, color=BLUE, alpha=0.18, lw=4)
    a = ramp(t / T, 0, 0.3)
    fig.text(0.5, 0.56, "From 6 sensor signals to a step count ±", ha="center", fontsize=40, color=DARK, alpha=a)
    fig.text(0.5, 0.45, "neural network  ·  PyTorch  ·  backpropagation  ·  5 networks", ha="center", fontsize=26,
             color=BLUE, alpha=ramp(t / T, 0.2, 0.5))


def scene1(fig, t, T, D):
    ax = fig.add_axes([0.03, 0.15, 0.27, 0.72])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    w = 2 * math.pi * 0.75
    th = math.radians(28) * math.sin(w * t)              # thigh
    sh = th - math.radians(30) * (1 + math.sin(w * t - 1.2)) / 2   # shin
    hip = np.array([0, 1.05])
    knee = hip + 0.5 * np.array([math.sin(th), -math.cos(th)])
    ank = knee + 0.5 * np.array([math.sin(sh), -math.cos(sh)])
    ax.plot(*zip(hip, knee, ank), color=DARK, lw=10, solid_capstyle="round")
    ax.plot([ank[0], ank[0] + 0.16], [ank[1], ank[1] - 0.02], color=DARK, lw=10, solid_capstyle="round")
    dsh = abs(math.cos(w * t) * 28 - 15 * math.cos(w * t - 1.2))
    ax.add_patch(Rectangle((ank[0] - 0.07, ank[1] + 0.03), 0.14, 0.12, color=BLUE, zorder=5))
    ax.text(ank[0] - 0.15, ank[1] + 0.09, "UNO Q +\nMPU-6050", ha="right", va="center", fontsize=15, color=BLUE)
    ax.add_patch(plt.Circle((ank[0], ank[1] + 0.09), 0.1 + 0.002 * dsh, color=ORANGE,
                            alpha=min(0.6, 0.012 * dsh), zorder=4))
    ax.text(0, -0.08, "right leg", ha="center", fontsize=16, color="#666666")
    a2 = fig.add_axes([0.36, 0.2, 0.6, 0.64])
    seg = D["seg"].T
    span = 4 * 50
    s0 = min(int(t * 50), seg.shape[1] - span - 1)
    stacked(a2, seg[:, s0:s0 + span], t=np.arange(span) / 50)
    a2.set_xlabel("seconds", fontsize=16)
    a2.set_title("real signals from your hallway walk (accel in g, gyro in °/s)", fontsize=18)
    clean(a2)
    fig.text(0.18, 0.86, "1 swing = 1 stride = 2 steps", ha="center", fontsize=20, color=ORANGE,
             alpha=ramp(t / T, 0.15, 0.3))


def scene2(fig, t, T, D):
    u = t / T
    ax = fig.add_axes([0.05, 0.24, 0.6, 0.6])
    seg = D["seg"].T
    tt = np.arange(seg.shape[1]) / 50
    stacked(ax, seg, t=tt)
    ax.set_xlabel("seconds", fontsize=16)
    clean(ax)
    ws = D["seg_win_start"]
    k = int(min(1, u / 0.7) * 10)
    x0 = ws - (10 - k) * 0.24
    ax.add_patch(Rectangle((x0, -1.4), 2.0, 2.3 * 5 + 2.8, fc=BLUE, alpha=0.12, ec=BLUE, lw=3))
    ax.text(x0 + 1.0, 2.3 * 5 + 1.6, "2 s window → network", ha="center", fontsize=17, color=BLUE)
    if u > 0.65:
        a = ramp(u, 0.65, 0.85)
        ins = fig.add_axes([0.38, 0.47, 0.27, 0.33])
        ins.imshow(D["x_typ"], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, alpha=a)
        ins.set_yticks(range(6), CHANNELS, fontsize=13)
        ins.set_xticks([0, 50, 99], ["0", "50", "99"], fontsize=12)
        ins.set_title("6 × 100 numbers, scaled to about −1…+1", fontsize=15, alpha=a)


def scene3(fig, t, T, D):
    u = t / T
    x, k = D["x_typ"], D["k_best"]
    z = D["acts"]["conv1"][k]
    relu = D["acts"]["relu1"][k]
    pool = D["acts"]["pool1"][k]
    ax = fig.add_axes([0.05, 0.6, 0.6, 0.27])
    stacked(ax, x, lw=1.3)
    ax.set_xticks([])
    ax.set_xlim(0, 99)
    clean(ax)
    p = int(min(1.0, u / 0.4) * (len(z) - 1))
    if u < 0.42:
        ax.add_patch(Rectangle((p - 0.5, -1.4), 7, 2.3 * 5 + 2.8, fc=ORANGE, alpha=0.25, ec=ORANGE, lw=2.5))
    kin = fig.add_axes([0.05, 0.43, 0.07, 0.13])
    kin.imshow(D["kernel"], cmap="coolwarm", aspect="auto")
    kin.set_xticks([])
    kin.set_yticks([])
    kin.set_title(f"filter {k + 1}: W (6×7)", fontsize=13)
    ao = fig.add_axes([0.16, 0.2, 0.49, 0.33])
    clean(ao)
    eq = r"$z_t = b + \Sigma_i\,\Sigma_k\; W_{i,k}\, x_{i,t+k}$"
    if u < 0.42:
        ao.plot(np.arange(p + 1), z[:p + 1], color=ORANGE, lw=2.5)
        ao.set_xlim(0, len(z))
        ao.set_ylim(z.min() * 1.1, z.max() * 1.1)
        ao.axhline(0, color="#999999", lw=1)
        ao.set_title("filter output z, one value per position", fontsize=17)
    elif u < 0.72:
        a = ramp(u, 0.42, 0.55)
        ao.plot(z, color="#bbbbbb", lw=2)
        ao.fill_between(np.arange(len(z)), np.minimum(z, 0), 0, color=RED, alpha=0.35 * (1 - a))
        ao.plot(relu, color=ORANGE, lw=3, alpha=a)
        ao.axhline(0, color="#999999", lw=1)
        ao.set_xlim(0, len(z))
        ao.set_title(f"ReLU: negatives → 0   ({D['relu_off']:.0%} of all values in layers 1–3 switched off)",
                     fontsize=17)
        eq = r"$a_t = \mathrm{max}(0,\; z_t)$"
    elif u < 0.86:
        a = ramp(u, 0.72, 0.8)
        ao.plot(relu, color="#cccccc", lw=2)
        ao.step(np.arange(len(pool)) * 2 + 1, pool, where="mid", color=PURPLE, lw=3, alpha=a)
        ao.set_xlim(0, len(z))
        ao.set_title("max-pool: keep the stronger of each pair  (94 → 47 values)", fontsize=17)
        eq = r"$p_t = \mathrm{max}(a_{2t},\; a_{2t+1})$"
    else:
        a = ramp(u, 0.86, 0.95)
        ao.imshow(D["acts"]["relu1"], aspect="auto", cmap="magma", alpha=a)
        ao.set_title("all 16 filters together: layer 1's output (16 × 94)", fontsize=17)
        ao.set_yticks([])
        eq = ""
    ao.tick_params(labelsize=13)
    if eq:
        ao.text(0.01, 0.97, eq, fontsize=24, color=DARK, va="top", transform=ao.transAxes,
                bbox={"fc": "white", "ec": "none", "alpha": 0.85})


def scene4(fig, t, T, D):
    u = t / T
    acts = D["acts"]
    ax = fig.add_axes([0.05, 0.66, 0.6, 0.17])
    ax.plot(D["x_typ"][5], color=CH_COLORS[5], lw=2)
    ax.set_xlim(0, 99)
    ax.axis("off")
    ax.set_title("the input (gz shown); brackets = how much time one value sees", fontsize=16)
    specs = [("relu1", "layer 1: 16 × 94", 7, "0.14 s"), ("relu2", "layer 2: 32 × 43", 16, "0.32 s"),
             ("relu3", "layer 3: 32 × 17", 34, "0.68 s")]
    for i, (key, label, rf, secs) in enumerate(specs):
        a = ramp(u, 0.05 + 0.28 * i, 0.2 + 0.28 * i)
        if a <= 0:
            continue
        hm = fig.add_axes([0.05 + 0.205 * i, 0.22, 0.19, 0.36])
        hm.imshow(acts[key], aspect="auto", cmap="magma", alpha=a)
        hm.set_xticks([])
        hm.set_yticks([])
        hm.set_title(label, fontsize=17, alpha=a)
        hm.set_xlabel(f"one value sees {secs}", fontsize=16, alpha=a)
        y = -0.15 - 0.35 * i
        mid = 50
        ax.plot([mid - rf / 2, mid + rf / 2], [y, y], color=[ORANGE, PURPLE, RED][i], lw=6, alpha=a,
                solid_capstyle="butt", clip_on=False)
        ax.text(mid + rf / 2 + 2, y, secs, fontsize=14, va="center", color=[ORANGE, PURPLE, RED][i], alpha=a)
    ax.set_ylim(-1.2, None)


def scene5(fig, t, T, D):
    u = t / T
    hm = fig.add_axes([0.05, 0.22, 0.25, 0.6])
    hm.imshow(D["acts"]["relu3"], aspect="auto", cmap="magma")
    hm.set_title("layer 3: 32 × 17", fontsize=17)
    hm.set_yticks([])
    hm.set_xlabel("time →", fontsize=15)
    f = D["acts"]["features"]
    g = ramp(u, 0.1, 0.45)
    b1 = fig.add_axes([0.34, 0.22, 0.14, 0.6])
    b1.barh(np.arange(32), f * g, color=GREEN)
    b1.set_ylim(31.5, -0.5)
    b1.set_xlim(0, max(f.max(), D["feat_stand"].max()) * 1.1)
    b1.set_yticks([])
    b1.set_title("walking: 32 numbers", fontsize=17)
    clean(b1)
    a = ramp(u, 0.5, 0.75)
    if a > 0:
        b2 = fig.add_axes([0.51, 0.22, 0.14, 0.6])
        b2.barh(np.arange(32), D["feat_stand"] * a, color="#999999")
        b2.set_ylim(31.5, -0.5)
        b2.set_xlim(0, max(f.max(), D["feat_stand"].max()) * 1.1)
        b2.set_yticks([])
        b2.set_title("standing: almost all 0", fontsize=17)
        clean(b2)
    fig.text(0.05, 0.86, r"$f_c = \frac{1}{17}\,\Sigma_t\, a_{c,t}$", fontsize=26, color=DARK)


def softplus(z):
    return np.log1p(np.exp(z))


def scene6(fig, t, T, D):
    u = t / T
    m = D["out1"]
    z1, z2 = D["acts"]["rate_head"]
    ax = fig.add_axes([0.06, 0.24, 0.27, 0.56])
    zz = np.linspace(-5, 3, 200)
    ax.plot(zz, softplus(zz), color="black", lw=3, label="softplus")
    ax.plot(zz, np.maximum(0, zz), color="#aaaaaa", ls="--", lw=1.5, label="ReLU")
    a = ramp(u, 0.1, 0.35)
    zc = -5 + (z1 + 5) * a
    ax.plot([zc], [softplus(zc)], "o", color=NET[0], ms=16)
    if a >= 1:
        ax.text(-4.8, 2.6, f"z₁ = {z1:.2f} → μ = {SPM * m[0]:.0f} steps/min", fontsize=16, color=NET[0])
    b = ramp(u, 0.4, 0.6)
    if b > 0:
        zc2 = -5 + (z2 + 5) * b
        ax.plot([zc2], [softplus(zc2)], "s", color=NET[1], ms=14)
        if b >= 1:
            ax.text(-4.8, 2.15, f"z₂ = {z2:.2f} → σ = ±{SPM * m[1]:.1f} steps/min", fontsize=16, color=NET[1])
    ax.set_ylim(-0.2, 3.1)
    ax.legend(fontsize=14, loc="lower right")
    ax.set_title(r"$\mathrm{softplus}(z) = \ln(1 + e^{z})$", fontsize=22)
    ax.tick_params(labelsize=13)
    clean(ax)
    a2 = fig.add_axes([0.38, 0.24, 0.27, 0.56])
    g = np.linspace(60, 130, 400)
    h = ramp(u, 0.55, 0.85)
    a2.fill_between(g, h * gauss(g, SPM * m[0], SPM * m[1]), color=NET[0], alpha=0.35)
    a2.plot(g, h * gauss(g, SPM * m[0], SPM * m[1]), color=NET[0], lw=3)
    a2.axvline(D["true_typ"], color=RED, ls="--", lw=2.5)
    a2.text(D["true_typ"] + 1, 1.02, f"true {D['true_typ']:.0f}", color=RED, fontsize=15)
    a2.set_ylim(0, 1.12)
    a2.set_yticks([])
    a2.set_xlabel("cadence (steps/min)", fontsize=16)
    a2.set_title(f"cadence: {SPM * m[0]:.0f} ± {SPM * m[1]:.1f}     speed: {m[2]:.2f} ± {m[3]:.2f} m/s",
                 fontsize=18, alpha=h)
    a2.tick_params(labelsize=13)
    clean(a2)


def net_boxes(ax, glow=None, pulse=None):
    names = ["input", "layer 1", "layer 2", "layer 3", "average", "heads"]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, n in enumerate(names):
        g = 0 if glow is None else glow.get(n, 0)
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.2), 0.84, 0.6, boxstyle="round,pad=0.02",
                                    fc=(1, 0.55 + 0.45 * (1 - g), 0.2 + 0.8 * (1 - g)), ec=DARK, lw=2))
        ax.text(i + 0.5, 0.5, n, ha="center", va="center", fontsize=18, weight="bold", color=DARK)
        if i:
            ax.annotate("", (i + 0.08, 0.5), (i - 0.08, 0.5), arrowprops={"arrowstyle": "-|>", "lw": 2})
    if pulse is not None:
        ax.plot([pulse], [0.5], "o", ms=30, color=RED, alpha=0.6)


def scene7(fig, t, T, D):
    u = t / T
    top = fig.add_axes([0.04, 0.66, 0.63, 0.17])
    fig.text(0.04, 0.855, "PyTorch, on the laptop", fontsize=17, color="#ee4c2c", weight="bold")
    o, ch = D["init_out"], D["chain"]
    y = D["true_typ"]
    low = fig.add_axes([0.06, 0.2, 0.3, 0.38])
    rt = fig.add_axes([0.42, 0.2, 0.25, 0.38])
    rt.axis("off")
    g = np.linspace(0, 200, 500)
    if u < 0.16:
        net_boxes(top, pulse=0.5 + 5 * u / 0.16)
        low.plot(g, gauss(g, SPM * o[0], SPM * o[1]), color=NET[0], lw=3)
        low.axvline(y, color=RED, ls="--", lw=2.5)
        low.set_title(f"before training: {SPM * o[0]:.0f} ± {SPM * o[1]:.0f} steps/min (true {y:.0f})", fontsize=16)
        rt.text(0, 0.8, "1. Forward pass", fontsize=24, weight="bold", color=DARK)
        rt.text(0, 0.6, "a batch of windows flows\nthrough the network", fontsize=18)
    elif u < 0.36:
        net_boxes(top)
        low.plot(g, gauss(g, SPM * o[0], SPM * o[1]), color=NET[0], lw=3)
        low.axvline(y, color=RED, ls="--", lw=2.5)
        low.annotate("", (y, 0.5), (SPM * o[0], 0.5), arrowprops={"arrowstyle": "<->", "color": RED, "lw": 3})
        low.text((y + SPM * o[0]) / 2, 0.56, "error", color=RED, fontsize=16, ha="center")
        rt.text(0, 0.8, "2. Loss", fontsize=24, weight="bold", color=DARK)
        rt.text(0, 0.55, r"$L = \ln\sigma + \frac{(y-\mu)^2}{2\sigma^2}$", fontsize=26)
        rt.text(0, 0.3, f"= {D['init_loss']:.2f} for this window", fontsize=18)
    elif u < 0.58:
        v = (u - 0.36) / 0.22
        order = ["heads", "average", "layer 3", "layer 2", "layer 1"]
        gmax = max(D["grad_norm"].values())
        glow = {}
        for i, n in enumerate(order):
            if v > i / 5:
                key = n if n in D["grad_norm"] else None
                glow[n] = 0.35 + 0.65 * (D["grad_norm"].get(key, gmax * 0.5) / gmax) ** 0.5 if key else 0.4
        net_boxes(top, glow=glow, pulse=5.5 - 5 * v)
        low.axis("off")
        rt.text(0, 0.8, "3. loss.backward()", fontsize=24, weight="bold", color=DARK)
        rt.text(0, 0.35, "the chain rule runs backwards:\nbrighter = bigger gradient\n"
                         f"(all {D['n_weights']:,} weights get one)", fontsize=18)
        y0 = 0.95
        for n in ["heads", "layer 3", "layer 2", "layer 1"]:
            if n in glow:
                low.text(0.02, y0, f"|∂L/∂w| {n}: {D['grad_norm'][n]:.2f}", fontsize=17, transform=low.transAxes)
                y0 -= 0.16
    elif u < 0.74:
        net_boxes(top, glow={"heads": 1.0})
        low.axis("off")
        rt.axis("off")
        lines = [f"one weight in the cadence head (from detector {ch['j'] + 1}):",
                 rf"$\partial L/\partial\mu = -(y-\mu)/\sigma^2 = {ch['dl_dmu']:.2f}$",
                 rf"$\partial\mu/\partial z = \mathrm{{sigmoid}}(z) = {ch['sigmoid']:.3f}$",
                 rf"$\partial z/\partial w = f_{{{ch['j'] + 1}}} = {ch['feature']:.3f}$",
                 rf"$\partial L/\partial w = {ch['product']:.3f}$   (PyTorch: {ch['grad']:.3f})",
                 rf"$w \leftarrow w - \eta\,\partial L/\partial w$:  {ch['w']:.4f} → {ch['w_new_sgd']:.4f}"]
        for i, s in enumerate(lines):
            fig.text(0.06, 0.55 - 0.065 * i, s, fontsize=22 if i else 18, color=DARK if i < 5 else BLUE,
                     alpha=ramp(u, 0.58 + 0.025 * i, 0.6 + 0.025 * i))
        fig.text(0.06, 0.6, "4. Chain rule on one weight, then Adam's step", fontsize=24, weight="bold", color=DARK)
    else:
        v = ramp(u, 0.74, 0.97)
        net_boxes(top)
        curve = D["loss_curve"]
        n = max(1, int(v * len(curve)))
        low.plot(np.arange(1, n + 1), curve[:n], color=BLUE, lw=3)
        low.set_xlim(1, len(curve))
        low.set_ylim(curve.min() - 0.1, curve[:3].max() + 0.1)
        low.set_xlabel("epoch", fontsize=16)
        low.set_title(f"training loss, epoch {n} of {len(curve)}", fontsize=17)
        f1 = D["out1"]
        mu = (1 - v) * o[0] + v * f1[0]
        sd = math.exp((1 - v) * math.log(o[1]) + v * math.log(f1[1]))
        ax = fig.add_axes([0.42, 0.2, 0.25, 0.38])
        ax.plot(g, gauss(g, SPM * mu, SPM * sd), color=NET[0], lw=3)
        ax.axvline(y, color=RED, ls="--", lw=2.5)
        ax.set_yticks([])
        ax.set_xlim(40, 160)
        ax.set_title(f"guess: {SPM * mu:.0f} ± {SPM * sd:.0f} steps/min", fontsize=17)
        ax.tick_params(labelsize=13)
        clean(ax)
        fig.text(0.06, 0.6, "5. Repeat for every batch, every epoch", fontsize=24, weight="bold", color=DARK)
    low.tick_params(labelsize=13)
    if u < 0.36:
        low.set_yticks([])
        low.set_xlim(0, 200)
        low.set_xlabel("cadence (steps/min)", fontsize=15)
    clean(low)


def scene8(fig, t, T, D):
    u = t / T
    n = len(NET)
    for k in range(n):
        a = ramp(u, 0.04 * k, 0.04 * k + 0.12)
        if a <= 0:
            continue
        ax = fig.add_axes([0.05 + 0.125 * k, 0.63, 0.11, 0.19])
        ax.bar(range(len(D["trust"][k])), D["trust"][k], color=NET[k], alpha=a)
        ax.axhline(1, color="#999999", ls="--", lw=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, max(3, D["trust"].max()))
        ax.set_title(f"network {k + 1}", fontsize=17, color=NET[k], weight="bold", alpha=a)
        ax.set_xlabel("trust in each recording", fontsize=12, alpha=a)
    for col, (key, true, lo_u, title) in enumerate((("typ", D["true_typ"], 0.3, "normal walking: they agree"),
                                                    ("unf", D["true_unf"], 0.62, "unusual moment: they disagree"))):
        a = ramp(u, lo_u, lo_u + 0.12)
        if a <= 0:
            continue
        ax = fig.add_axes([0.06 + 0.32 * col, 0.19, 0.28, 0.24])
        mem, e = D[f"members_{key}"], D[f"ens_{key}"]
        lo = min(SPM * (e["rate"] - 4 * e["rate_sd"]), true - 10)
        hi = max(SPM * (e["rate"] + 4 * e["rate_sd"]), true + 10)
        g = np.linspace(max(0, lo), hi, 400)
        for k in range(n):
            ax.plot(g, gauss(g, SPM * mem[k][0], SPM * mem[k][1]) / mem[k][1], color=NET[k], lw=2, alpha=a)
        ax.plot(g, gauss(g, SPM * e["rate"], SPM * e["rate_sd"]) / e["rate_sd"], color="black", lw=4, alpha=a)
        ax.axvline(true, color=RED, ls="--", lw=2.5, alpha=a)
        ax.set_yticks([])
        ax.tick_params(labelsize=13)
        ax.set_title(f"{title}\ncombined {SPM * e['rate']:.0f} ± {SPM * e['rate_sd']:.0f} (true {true:.0f})",
                     fontsize=16, alpha=a)
        clean(ax)
    if u > 0.85:
        fig.text(0.06, 0.54, r"$\bar\sigma^2 = \frac{1}{5}\Sigma_k\,\sigma_k^2 + \frac{1}{5}\Sigma_k\,(\mu_k-\bar\mu)^2$"
                             "    own unsureness + disagreement", fontsize=21, color=DARK,
                 alpha=ramp(u, 0.85, 0.93))


def scene9(fig, t, T, D):
    w = D["walk"]
    u = t / T
    v = ramp(u, 0.05, 0.85)
    n = max(2, int(v * len(w["t"])))
    ax = fig.add_axes([0.33, 0.22, 0.63, 0.6])
    tt, st, sd = w["t"][:n], w["steps"][:n], w["sd"][:n]
    ax.fill_between(tt, st - 1.96 * sd, st + 1.96 * sd, color=BLUE, alpha=0.15, lw=0, label="95%")
    ax.fill_between(tt, st - sd, st + sd, color=BLUE, alpha=0.3, lw=0, label="68%")
    ax.plot(tt, st, color=BLUE, lw=3, label="network")
    ax.plot(tt, w["true"][:n], color=RED, ls="--", lw=2, label="true (from your count)")
    ax.set_xlim(0, w["t"][-1])
    ax.set_ylim(0, max(w["steps"][-1] + 2 * 1.96 * w["sd"][-1], w["counted"]) * 1.05)
    ax.set_xlabel("seconds into your long walk", fontsize=16)
    ax.legend(fontsize=15, loc="upper left")
    ax.tick_params(labelsize=13)
    clean(ax)
    fig.text(0.05, 0.62, f"{st[-1]:.0f}", fontsize=110, weight="bold", color=BLUE)
    fig.text(0.05, 0.55, f"± {1.96 * sd[-1]:.1f} steps", fontsize=30, color=BLUE)
    if u > 0.85:
        f = w["final"]
        a = ramp(u, 0.85, 0.92)
        fig.text(0.05, 0.4, f"you counted {w['counted']}", fontsize=28, color=RED, alpha=a)
        fig.text(0.05, 0.33, f"95% range {f['steps_lo']}–{f['steps_hi']}", fontsize=24, color=DARK, alpha=a)


def scene10(fig, t, T, D):
    u = t / T
    ax = fig.add_axes([0.03, 0.17, 0.94, 0.7])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 1.2), 2.8, 2.6, boxstyle="round,pad=0.1", fc="#fde9d9", ec=DARK, lw=2))
    ax.text(1.7, 3.3, "laptop", ha="center", fontsize=24, weight="bold")
    ax.text(1.7, 2.3, "PyTorch:\ntraining +\nbackpropagation", ha="center", fontsize=18)
    x = 3.4 + 2.3 * ramp(u, 0.1, 0.55)
    ax.add_patch(Rectangle((x, 2.0), 0.9, 1.1, fc="white", ec=BLUE, lw=2.5))
    ax.text(x + 0.45, 2.55, "{ }", ha="center", va="center", fontsize=26, color=BLUE)
    ax.text(x + 0.45, 1.7, "stepnet_model.json\n5 × 8,564 weights", ha="center", fontsize=13, color=BLUE)
    ax.add_patch(FancyBboxPatch((6.8, 1.2), 2.9, 2.6, boxstyle="round,pad=0.1", fc="#e6f4ea", ec=DARK, lw=2))
    ax.text(8.25, 3.45, "UNO Q (board)", ha="center", fontsize=22, weight="bold")
    ax.text(8.25, 1.35, f"numpy forward pass only\nmatches PyTorch to {D['numpy_diff']:.0e}", ha="center",
            fontsize=14)
    if u > 0.5:
        count = f"{D['walk']['steps'][-1]:.0f}" if D["walk"] else "0"
        cols = []
        for ch_ in count:
            cols += D["font"][int(ch_)] + [0]
        lit = []
        for ci, col in enumerate(cols):
            for r in range(7):
                if col >> r & 1:
                    lit.append((ci, r))
        width = len(cols)
        off = int((u - 0.5) * 30) % (width + 13)
        for ci in range(13):
            for r in range(8):
                src = ci + off - 13
                on = (src, r) in lit
                ax.plot(7.25 + ci * 0.155, 3.0 - r * 0.155, "s", ms=8, color=RED if on else "#e0e0e0")


DRAW = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10]
MIN_LEN = [3.0, 5.0, 4.0, 8.0, 4.0, 4.0, 5.0, 11.0, 8.0, 6.0, 4.0]


# ---------------------------------------------------------------- rendering ---
def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_scene(args):
    i, length, caps, fps, data_path, out = args
    D = pickle.loads(Path(data_path).read_bytes())
    sc = D["scenes"][i]
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_PX}x{H_PX}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
           str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(length * fps))
    for f in range(n):
        t = f / fps
        base(fig, sc, t, length, caps)
        DRAW[i](fig, t, length, D)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, nargs="*", help="only these scenes (preview); default all")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)

    print("Preparing data from the installed model...")
    D = prepare()
    scenes = [dict(s, lines=list(s["lines"])) for s in SCENES]
    if D["walk"]:
        f = D["walk"]["final"]
        n, pm, counted = int(round(f["steps"])), int(round(1.96 * f["steps_sd"])), D["walk"]["counted"]
        scenes[9]["lines"][1] = (f"You counted {number_words(counted)}. The network said {number_words(n)}, "
                                 f"plus or minus {number_words(pm)}.",
                                 f"You counted {counted} steps. The network said {n} ± {pm}.")
    D["scenes"] = scenes
    data_path = BUILD / "data.pkl"
    data_path.write_bytes(pickle.dumps(D))

    print("Synthesizing the narration...")
    spoken = [s for sc in scenes for s, _ in sc["lines"]]
    paths, durs = synthesize(spoken, BUILD / "voice")
    timeline = build_timeline(scenes, durs, MIN_LEN)
    total = timeline[-1][0] + timeline[-1][1]
    print(f"  {len(spoken)} sentences, {sum(durs):.1f} s of speech; video {total:.1f} s")
    write_audio(paths, scenes, timeline, BUILD / "narration.wav")
    write_srt(timeline, OUT.with_suffix(".srt"))

    which = args.scenes if args.scenes else list(range(len(scenes)))
    jobs = [(i, timeline[i][1], timeline[i][2], args.fps, str(data_path), BUILD / f"scene{i:02d}.mp4") for i in which]
    print(f"Rendering {len(jobs)} scenes in parallel...")
    with ProcessPoolExecutor(max_workers=min(len(jobs), 11)) as ex:
        for out in ex.map(render_scene, jobs):
            print(f"  done {Path(out).name}")
    if args.scenes:
        print(f"Preview scenes are in {BUILD}")
        return

    listing = BUILD / "scenes.txt"
    listing.write_text("".join(f"file '{(BUILD / f'scene{i:02d}.mp4').as_posix()}'\n" for i in range(len(scenes))),
                       encoding="utf-8")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-i", str(BUILD / "narration.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", str(OUT)], check=True)
    print(f"Wrote {OUT} ({total:.0f} s) and {OUT.with_suffix('.srt').name}")


if __name__ == "__main__":
    main()
