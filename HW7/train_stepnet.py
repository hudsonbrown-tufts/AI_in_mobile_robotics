"""Train the HW7 step network ensemble with PyTorch (backpropagation).

Steps (HOW_THE_NET_WORKS.md explains each one):
  1. Labels: stride times for every recording (labels.py).
  2. Leave-one-walk-out check: for each hallway walk, train FOLD_MEMBERS networks
     without it and run the board's exact adding-up code on it. This gives
     honest errors for walks the networks never saw, used to
       - pick the number of training epochs,
       - correct the overall bias: k_rate / k_speed = true / predicted totals,
       - calibrate the uncertainty: c_rate / c_speed scale the window spreads so
         held-out errors are 1 spread on average (RMS); s_steps / s_dist are the
         systematic per-walk errors left over after the random part,
       - print a reliability table (claimed vs. actual coverage).
  3. Final ensemble: N_MEMBERS networks on all the training data. Each starts
     from different random weights and sees the recordings with different
     random weights (a Bayesian bootstrap), so they disagree where the data
     doesn't pin the answer down.
  4. Gradient check (backprop computed by PyTorch vs. nudging weights by hand),
     numpy-vs-PyTorch check, and the final test on the long walk(s), which are
     never trained on.

Training a member = backpropagation: forward pass -> beta-NLL loss ->
loss.backward() (chain rule through every layer gives d loss / d weight for all
~9k weights) -> Adam step. Repeated for every batch of every epoch.

Normally the GUI runs this (Record tab -> Train) and installs the result.
From the command line:

    .venv/Scripts/python HW7/train_stepnet.py --out model.json      # train, write elsewhere
    .venv/Scripts/python HW7/train_stepnet.py --install             # train and install in the app
    .venv/Scripts/python HW7/train_stepnet.py --data HW7/testing/data_synthetic --no-hw6 --out m.json
"""

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# The networks are tiny, so training runs on the CPU (several processes side
# by side). Hiding the GPU stops worker processes from trying to start CUDA
# (which failed with "CUDA-capable device is busy" once).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import config  # noqa: E402
import dataset  # noqa: E402
import labels as L  # noqa: E402
import stepnet as S  # noqa: E402

# Fixed input scales: accel / 2 g, gyro / 500 deg/s (same as HW6-NN).
SCALES = [2.0, 2.0, 2.0, 500.0, 500.0, 500.0]

# Augmentation (one person, one mounting, so smaller than HW6-NN's)
ROTATE_DEG = 8            # strap sits a little differently each time
AMPLITUDE = (0.92, 1.08)  # a bit softer / harder
STRETCH = (0.85, 1.15)    # play the walk slower / faster: the labels change with it
NOISE_ACCEL_G, NOISE_GYRO_DPS = 0.01, 1.5

N_MEMBERS = 5             # final ensemble
FOLD_MEMBERS = 3          # networks per leave-one-walk-out fold (fewer, for speed)
MAX_EPOCHS, MIN_EPOCHS = 80, 15
BATCH, LR, WEIGHT_DECAY = 64, 2e-3, 1e-4
BETA = 0.5                # beta-NLL
MOVING_RATE = 0.05        # windows above this stride rate count as "walking" for balancing
S_STEPS_FLOOR, S_DIST_FLOOR = 0.01, 0.03
SEED = 0


class Cancelled(Exception):
    """Raised inside training when should_stop() returns True."""


class StrideNet(nn.Module):
    """Must match stepnet.StrideNet (same layers, same names)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, 7)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.rate_head = nn.Linear(32, 2)
        self.speed_head = nn.Linear(32, 2)

    def forward(self, x):                      # x: (batch, 6, WINDOW)
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = torch.relu(self.conv3(h))
        feat = h.mean(dim=2)                   # average over time: (batch, 32)
        r, s = self.rate_head(feat), self.speed_head(feat)
        return torch.stack([F.softplus(r[:, 0]), F.softplus(r[:, 1]) + S.SIGMA_FLOOR,
                            F.softplus(s[:, 0]), F.softplus(s[:, 1]) + S.SIGMA_FLOOR], dim=1)


def beta_nll(mu, sd, y, beta=BETA):
    """Gaussian negative log-likelihood, each window weighted by its variance^beta.

    log(sd) + (y - mu)^2 / (2 sd^2): a confident wrong answer costs a lot, and so
    does a huge spread everywhere. The detached var^beta factor (Seitzer et al.
    2022) keeps hard windows from just getting a big spread and being ignored.
    """
    var = sd ** 2
    nll = 0.5 * torch.log(var) + (y - mu) ** 2 / (2 * var)
    return nll * var.detach() ** beta


# ---------------------------------------------------------------- data -------

class Source:
    """One continuous piece of a recording with its stride phase at every row boundary."""

    def __init__(self, rec, lab, a, b):
        t = rec.t[a:b]
        self.name = rec.name
        self.x6 = rec.rows[a:b][:, [S.COL[c] for c in S.NET_CHANNELS]]
        # phase at the start of every row, plus at the end of the last row
        self.pb = lab.phase(np.append(t, t[-1] + 1 / S.SAMPLE_HZ))
        self.length = lab.stride_length_m           # None = distance unknown
        self.weight = 1.0

    def __len__(self):
        return len(self.x6)


def sources_for(recs, labs):
    out = []
    for r in recs:
        lab = labs[r.name]
        if not lab.ok:
            continue
        for a, b in r.segments():
            if lab.valid is not None:                # only between the first and last stride
                t = r.t
                a = max(a, int(np.searchsorted(t, lab.valid[0])))
                b = min(b, int(np.searchsorted(t, lab.valid[1], side="right")))
            if b - a >= int(S.WINDOW * STRETCH[1]) + 1:
                out.append(Source(r, lab, a, b))
    return out


def random_rotations(rng, k):
    """k random small rotations (k, 3, 3) of up to ROTATE_DEG about random axes (Rodrigues' formula)."""
    axis = rng.normal(size=(k, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = np.radians(rng.uniform(-ROTATE_DEG, ROTATE_DEG, k))[:, None, None]
    z = np.zeros(k)
    kx = np.stack([np.stack([z, -axis[:, 2], axis[:, 1]], 1),
                   np.stack([axis[:, 2], z, -axis[:, 0]], 1),
                   np.stack([-axis[:, 1], axis[:, 0], z], 1)], 1)
    return np.eye(3)[None] + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def augmented_windows(src, k, rng):
    """k random 2 s windows from a source with all augmentations.

    -> windows (k, WINDOW, 6), stride rate (k,), speed (k,)
    """
    stretch = rng.uniform(*STRETCH, k)               # >1: more than 2 s of walking squeezed into 2 s
    src_len = np.round(S.WINDOW * stretch).astype(int)
    start = (rng.random(k) * (len(src) - src_len + 1)).astype(int)
    pos = np.linspace(0, 1, S.WINDOW)[None, :] * (src_len - 1)[:, None]
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, (src_len - 1)[:, None])
    frac = (pos - i0)[..., None]
    w = src.x6[start[:, None] + i0] * (1 - frac) + src.x6[start[:, None] + i1] * frac
    r = random_rotations(rng, k)
    acc = np.einsum("kij,ktj->kti", r, w[..., :3])
    gyro = np.einsum("kij,ktj->kti", r, w[..., 3:])
    gain = (rng.uniform(*AMPLITUDE, k) * stretch)[:, None, None]
    mean = acc.mean(axis=1, keepdims=True)
    acc = mean + gain * (acc - mean)                 # faster playback = faster, harder motion; gravity stays
    gyro = gyro * gain                               # same angles in less time = faster rotation
    acc = acc + rng.normal(0, NOISE_ACCEL_G, acc.shape)
    gyro = gyro + rng.normal(0, NOISE_GYRO_DPS, gyro.shape)
    strides = src.pb[start + src_len] - src.pb[start]
    rate = strides / S.WINDOW_S                      # strides per second as the network sees it
    speed = rate * src.length if src.length is not None else np.zeros(k)
    return np.concatenate([acc, gyro], axis=2), rate, speed


def to_input(windows6):
    """(k, WINDOW, 6) -> (k, 6, WINDOW) float32 scaled."""
    return (np.transpose(windows6, (0, 2, 1)) / np.array(SCALES)[None, :, None]).astype(np.float32)


def epoch_data(sources, rng):
    """One epoch of augmented windows (about one per HOP of data) with labels and weights.

    -> x (k, 6, WINDOW), y (k, 3) = (rate, speed, speed known), weights (k,)
    """
    xs, ys, ws = [], [], []
    for src in sources:
        k = max(1, (len(src) - S.WINDOW) // S.HOP + 1)
        w, rate, speed = augmented_windows(src, k, rng)
        xs.append(w)
        ys.append(np.stack([rate, speed, np.full(k, float(src.length is not None))], axis=1))
        ws.append(np.full(k, src.weight))
    y = np.concatenate(ys).astype(np.float32)
    w = np.concatenate(ws).astype(np.float32)
    xs = np.concatenate(xs)
    moving = y[:, 0] > MOVING_RATE
    if moving.any() and (~moving).any():             # walking and standing windows count equally
        w[moving] *= 0.5 * len(w) / w[moving].sum()
        w[~moving] *= 0.5 * len(w) / w[~moving].sum()
    return to_input(xs), y, w / w.mean()


def board_windows(rec, lab):
    """The windows the board would see for a recording, in order, with their labels.

    -> X (k, 6, WINDOW), y (k, 2) = (rate, speed), new_rows (k,), first (k,)
    """
    win = S.Windower(config.MAX_GAP_ROWS)
    xs, ys, news, firsts = [], [], [], []
    t = rec.t
    seen = 0
    for seq, rows in rec.replies():
        seen += len(rows)
        for window, new_rows, first in win.add_rows(seq, rows):
            a, b = seen - S.WINDOW, seen
            strides = lab.phase(t[b - 1] + 1 / S.SAMPLE_HZ) - lab.phase(t[a])
            rate = strides / S.WINDOW_S
            speed = rate * lab.stride_length_m if lab.stride_length_m is not None else np.nan
            xs.append(window[:, [S.COL[c] for c in S.NET_CHANNELS]])
            ys.append((rate, speed))
            news.append(new_rows)
            firsts.append(first)
    return to_input(np.stack(xs)), np.array(ys), np.array(news), np.array(firsts)


# ---------------------------------------------------------------- training ---

def train_member(sources, epochs, seed, held_out=None, log=None, should_stop=None, tag=""):
    """Train one network by backpropagation.

    held_out: optional input tensor; its outputs are kept after every epoch.
    -> (model, train loss per epoch, held-out outputs per epoch (epochs, k, 4) or None)
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for src in sources:                                # Bayesian bootstrap: random weight per recording
        src.weight = float(rng.exponential())
    model = StrideNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses, outs = [], []
    for epoch in range(1, epochs + 1):
        if should_stop and should_stop():
            raise Cancelled()
        model.train()
        x, y, w = epoch_data(sources, rng)
        perm = rng.permutation(len(y))
        x, y, w = torch.from_numpy(x[perm]), torch.from_numpy(y[perm]), torch.from_numpy(w[perm])
        total = 0.0
        for i in range(0, len(y), BATCH):
            xb, yb, wb = x[i:i + BATCH], y[i:i + BATCH], w[i:i + BATCH]
            out = model(xb)                                            # 1. forward pass
            per_window = (beta_nll(out[:, 0], out[:, 1], yb[:, 0])     # 2. loss
                          + yb[:, 2] * beta_nll(out[:, 2], out[:, 3], yb[:, 1]))
            loss = (wb * per_window).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()                                            # 3. backpropagation
            opt.step()                                                 # 4. Adam updates every weight
            total += loss.item() * len(yb)
        losses.append(total / len(y))
        if held_out is not None:
            model.eval()
            with torch.no_grad():
                outs.append(model(held_out).numpy())
        if log and (epoch % 20 == 0 or epoch == epochs):
            log(f"    {tag} epoch {epoch:3d}  train loss {losses[-1]:.3f}")
    for src in sources:
        src.weight = 1.0
    model.eval()
    return model, losses, (np.stack(outs) if outs else None)


def gaussian_nll(mu, sd, y):
    return float(np.mean(np.log(sd) + (y - mu) ** 2 / (2 * sd ** 2) + 0.5 * np.log(2 * np.pi)))


def ensemble_outputs(member_outs):
    """List of (k, 4) member outputs -> per-window combined predictions (list of dicts)."""
    stacked = np.stack(member_outs, axis=1)             # (k, members, 4)
    return [S.combine(stacked[i]) for i in range(len(stacked))]


class _Cal:
    """Just enough of StepNetEnsemble for StepAccumulator."""

    def __init__(self, s_steps=0.0, s_dist=0.0, epistemic_warn=1e9, k_rate=1.0, k_speed=1.0):
        self.s_steps, self.s_dist, self.epistemic_warn = s_steps, s_dist, epistemic_warn
        self.k_rate, self.k_speed = k_rate, k_speed


def run_accumulator(preds, news, firsts, cal, c_rate=1.0, c_speed=1.0):
    """The board's adding-up over a walk's window predictions -> (StepAccumulator, history)."""
    acc = S.StepAccumulator(cal)
    hist = []
    for p, n, f in zip(preds, news, firsts):
        p = dict(p, rate_sd=p["rate_sd"] * c_rate, speed_sd=p["speed_sd"] * c_speed)
        acc.add(p, int(n), bool(f))
        hist.append(acc.summary())
    return acc, hist


def coverage(z, levels=(0.5, 0.6827, 0.95)):
    """Share of |z| inside each two-sided Gaussian interval."""
    z = np.abs(np.asarray(z))
    return {f"{lv:.0%}": round(float(np.mean(z <= math.sqrt(2) * erfinv(lv))), 3) for lv in levels}


def erfinv(p):
    from scipy.special import erfinv as _erfinv
    return float(_erfinv(p))


def _member_job(sources, epochs, seed, held_out):
    """One network, in a worker process (one CPU thread each; several run side by side)."""
    torch.set_num_threads(1)
    model, losses, outs = train_member(sources, epochs, seed,
                                       held_out=None if held_out is None else torch.from_numpy(held_out))
    return {k: v.numpy() for k, v in model.state_dict().items()}, losses, outs


def run_jobs(jobs, log, should_stop, describe):
    """Run _member_job(*args) for every job in parallel processes -> results in job order."""
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    results = [None] * len(jobs)
    with ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(_member_job, *job): i for i, job in enumerate(jobs)}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in done:
                results[futures[f]] = f.result()
                log(describe(futures[f]))
            if should_stop and should_stop():
                ex.shutdown(cancel_futures=True)
                raise Cancelled()
    return results


def to_model(state):
    model = StrideNet()
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})
    model.eval()
    return model


def leave_one_walk_out(train_recs, labs, log, should_stop):
    """Train without each hallway walk in turn; collect the held-out window outputs per epoch."""
    halls = [r for r in train_recs if r.kind == "hall" and labs[r.name].ok]
    folds, jobs = [], []
    for i, held in enumerate(halls):
        sources = sources_for([r for r in train_recs if r is not held], labs)
        x, y, news, firsts = board_windows(held, labs[held.name])
        folds.append({"rec": held, "y": y, "news": news, "firsts": firsts, "outs": []})
        jobs += [(sources, MAX_EPOCHS, SEED + 100 * i + m, x) for m in range(FOLD_MEMBERS)]
    log(f"  training {len(jobs)} networks in parallel...")
    results = run_jobs(jobs, log, should_stop,
                       lambda j: f"  fold {j // FOLD_MEMBERS + 1}/{len(halls)} "
                                 f"(without {halls[j // FOLD_MEMBERS].name}), network {j % FOLD_MEMBERS + 1} done")
    for j, (_, _, outs) in enumerate(results):
        folds[j // FOLD_MEMBERS]["outs"].append(outs)
    return folds


def pick_epochs(folds):
    """Epoch count with the lowest held-out NLL averaged over folds (5-epoch running mean)."""
    curve = np.zeros(MAX_EPOCHS)
    for fold in folds:
        y = fold["y"]
        known = ~np.isnan(y[:, 1])
        for e in range(MAX_EPOCHS):
            preds = ensemble_outputs([o[e] for o in fold["outs"]])
            mu = np.array([[p["rate"], p["speed"]] for p in preds])
            sd = np.array([[p["rate_sd"], p["speed_sd"]] for p in preds])
            curve[e] += gaussian_nll(mu[:, 0], sd[:, 0], y[:, 0])
            if known.any():
                curve[e] += gaussian_nll(mu[known, 1], sd[known, 1], y[known, 1])
    curve /= max(1, len(folds))
    smooth = np.convolve(curve, np.ones(5) / 5, mode="same")
    smooth[:2], smooth[-2:] = np.inf, np.inf                 # edges of the running mean are biased
    best = int(np.argmin(smooth[MIN_EPOCHS - 1:])) + MIN_EPOCHS
    return best, curve


def calibrate(folds, epochs, log):
    """Fit c_rate, c_speed, s_steps, s_dist on the held-out folds and print the reliability table."""
    e = epochs - 1
    z_rate, z_speed, walks = [], [], []
    fold_preds = []
    for fold in folds:
        preds = ensemble_outputs([o[e] for o in fold["outs"]])
        fold_preds.append(preds)
        y = fold["y"]
        for p, (yr, ys) in zip(preds, y):
            z_rate.append((p["rate"] - yr) / p["rate_sd"])
            if not np.isnan(ys):
                z_speed.append((p["speed"] - ys) / p["speed_sd"])
    # c = root-mean-square z: after dividing by it, held-out errors have spread 1
    # (the Gaussian-best single scale factor).
    c_rate = float(np.clip(np.sqrt(np.mean(np.square(z_rate))), 0.2, 5.0))
    c_speed = float(np.clip(np.sqrt(np.mean(np.square(z_speed))), 0.2, 5.0)) if z_speed else 1.0

    # Overall bias: k = true total / predicted total over all held-out walks
    # (one number for steps, one for distance).
    raw = [run_accumulator(p, f["news"], f["firsts"], _Cal(), c_rate, c_speed)[0] for f, p in zip(folds, fold_preds)]
    k_rate = float(np.clip(sum(f["rec"].steps / 2 for f in folds) / max(1e-9, sum(a.strides for a in raw)),
                           0.85, 1.15))
    with_m = [(f, a) for f, a in zip(folds, raw) if f["rec"].distance_m]
    k_speed = float(np.clip(sum(f["rec"].distance_m for f, _ in with_m) / max(1e-9, sum(a.dist for _, a in with_m)),
                            0.85, 1.15)) if with_m else 1.0

    # Walk totals with the bias correction and calibrated spreads, no systematic part yet.
    for fold, preds in zip(folds, fold_preds):
        rec = fold["rec"]
        acc, _ = run_accumulator(preds, fold["news"], fold["firsts"], _Cal(k_rate=k_rate, k_speed=k_speed),
                                 c_rate, c_speed)
        walks.append({"name": rec.name, "true_steps": rec.steps, "true_m": rec.distance_m,
                      "pred_strides": acc.strides, "var_strides": acc.var_strides,
                      "pred_m": acc.dist, "var_m": acc.var_dist})
    # Systematic part: what's left of each walk's error after the random part, relative to its size.
    s_steps_sq = [max(0.0, (w["pred_strides"] - w["true_steps"] / 2) ** 2 - w["var_strides"] - 1 / 12)
                  / (w["true_steps"] / 2) ** 2 for w in walks]
    s_dist_sq = [max(0.0, (w["pred_m"] - w["true_m"]) ** 2 - w["var_m"]) / w["true_m"] ** 2
                 for w in walks if w["true_m"]]
    s_steps_fit = math.sqrt(np.mean(s_steps_sq)) if walks else 0.0
    s_dist_fit = math.sqrt(np.mean(s_dist_sq)) if s_dist_sq else 0.0
    s_steps = max(S_STEPS_FLOOR, s_steps_fit)
    s_dist = max(S_DIST_FLOOR, s_dist_fit)

    cal = _Cal(s_steps, s_dist, k_rate=k_rate, k_speed=k_speed)
    z_walk_steps, z_walk_m, rows = [], [], []
    for fold, preds, w in zip(folds, fold_preds, walks):
        acc, _ = run_accumulator(preds, fold["news"], fold["firsts"], cal, c_rate, c_speed)
        s = acc.summary()
        z_walk_steps.append((s["steps"] - w["true_steps"]) / s["steps_sd"])
        z_walk_m.append((s["dist_m"] - w["true_m"]) / s["dist_sd"])
        rows.append({"walk": w["name"], "true_steps": w["true_steps"], "pred_steps": round(s["steps"], 1),
                     "steps_95": round(S.Z95 * s["steps_sd"], 1), "true_m": w["true_m"],
                     "pred_m": round(s["dist_m"], 2), "m_95": round(S.Z95 * s["dist_sd"], 2)})

    log(f"\nHeld-out hallway walks (each one predicted by networks that never saw it), epoch {epochs}:")
    log(f"  {'walk':<28}{'true':>6}{'predicted':>14}{'true m':>9}{'predicted m':>16}")
    for r in rows:
        log(f"  {r['walk']:<28}{r['true_steps']:>6}{r['pred_steps']:>8.1f} ±{r['steps_95']:<5.1f}"
            f"{r['true_m']:>9.1f}{r['pred_m']:>9.1f} ±{r['m_95']:<5.1f}")
    errs = [abs(r["pred_steps"] - r["true_steps"]) / r["true_steps"] for r in rows]
    derrs = [abs(r["pred_m"] - r["true_m"]) / r["true_m"] for r in rows if r["true_m"]]
    log(f"  mean step error {np.mean(errs):.1%}, mean distance error {np.mean(derrs):.1%}" if derrs else "")
    log(f"  (after the bias correction below, which was fit on these same walks: totals x k_rate {k_rate:.3f}, "
        f"distances x k_speed {k_speed:.3f})")
    log(f"\nUncertainty calibration: window spreads x c_rate {c_rate:.2f}, x c_speed {c_speed:.2f}; "
        f"systematic {s_steps:.1%} of steps (fit {s_steps_fit:.1%}, floor {S_STEPS_FLOOR:.0%}), "
        f"{s_dist:.1%} of distance (fit {s_dist_fit:.1%}, floor {S_DIST_FLOOR:.0%})")
    rel = {"windows_rate": coverage(np.array(z_rate) / c_rate),
           "windows_speed": coverage(np.array(z_speed) / c_speed) if z_speed else None,
           "walks_steps": coverage(z_walk_steps), "walks_distance": coverage(z_walk_m)}
    log("Reliability (share of held-out cases inside the claimed range):")
    log(f"  {'claimed':<12}" + "".join(f"{k:>8}" for k in rel["walks_steps"]))
    names = {"windows_rate": f"windows, stride rate ({len(z_rate)})",
             "windows_speed": f"windows, speed ({len(z_speed)})",
             "walks_steps": f"walks, step count ({len(rows)})", "walks_distance": f"walks, distance ({len(rows)})"}
    for k, v in rel.items():
        if v:
            log(f"  {names[k]:<34}" + "".join(f"{x:>8.0%}" for x in v.values()))
    log("  (with only a few walks, the walk rows are rough: each walk is a big share)")
    calib = {"k_rate": round(k_rate, 4), "k_speed": round(k_speed, 4),
             "c_rate": round(c_rate, 4), "c_speed": round(c_speed, 4), "s_steps": round(s_steps, 4),
             "s_dist": round(s_dist, 4)}
    return calib, {"walks": rows, "reliability": rel, "s_steps_fit": s_steps_fit, "s_dist_fit": s_dist_fit,
                   "z_rate": [round(float(z), 3) for z in z_rate]}


def gradient_check(model, sources, log):
    """Backprop's gradient vs. a finite difference for a few weights (in float64).

    Uses the plain NLL (beta = 0): beta-NLL deliberately holds its var^beta
    weight constant during backprop, which a finite difference can't do.
    """
    rng = np.random.default_rng(123)
    x, y, w = epoch_data(sources[:3], rng)
    x, y = torch.from_numpy(x[:16]).double(), torch.from_numpy(y[:16]).double()
    m = StrideNet().double()
    m.load_state_dict({k: v.double() for k, v in model.state_dict().items()})

    def loss_fn():
        out = m(x)
        return (beta_nll(out[:, 0], out[:, 1], y[:, 0], beta=0)
                + y[:, 2] * beta_nll(out[:, 2], out[:, 3], y[:, 1], beta=0)).mean()

    m.zero_grad()
    loss_fn().backward()
    checks = []
    for name, idx in (("conv1.weight", (3, 4, 3)), ("conv2.weight", (7, 2, 1)), ("conv3.weight", (5, 7, 2)),
                      ("rate_head.weight", (0, 4)), ("speed_head.bias", (1,))):
        p = dict(m.named_parameters())[name]
        grad = float(p.grad[idx])
        eps = 1e-6
        with torch.no_grad():
            old = float(p[idx])
            p[idx] = old + eps
            up = float(loss_fn())
            p[idx] = old - eps
            down = float(loss_fn())
            p[idx] = old
        numeric = (up - down) / (2 * eps)
        checks.append({"weight": f"{name}{list(idx)}", "backprop": grad, "nudge": numeric})
    log("\nGradient check (d loss / d weight: backpropagation vs. nudging the weight by +-1e-6):")
    for c in checks:
        log(f"  {c['weight']:<24} backprop {c['backprop']:+.6e}   nudge {c['nudge']:+.6e}")
    worst = max(abs(c["backprop"] - c["nudge"]) / max(1e-12, abs(c["nudge"])) for c in checks)
    log(f"  largest relative difference {worst:.1e}")
    if worst > 1e-4:
        raise RuntimeError("Backprop gradients don't match the finite differences; something is wrong.")
    return checks


def export_member(model):
    return {k: v.detach().cpu().numpy().astype(float).round(7).tolist() for k, v in model.state_dict().items()}


def check_numpy_matches(models, model_json, x, log):
    ens = S.StepNetEnsemble(model_json)
    worst = 0.0
    for model, net in zip(models, ens.members):
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        ours = np.array([net.forward(xi.astype(np.float64)) for xi in x])
        worst = max(worst, float(np.abs(ref - ours).max()))
    log(f"numpy (board) vs PyTorch: max difference {worst:.1e} over {len(x)} windows x {len(models)} networks")
    if worst > 1e-4:
        raise RuntimeError("The numpy forward pass doesn't match PyTorch; not saving.")


def test_walk(rec, lab, ens, label_info):
    """Run a recording through the exact board pipeline with the final ensemble."""
    win = S.Windower(config.MAX_GAP_ROWS)
    acc = S.StepAccumulator(ens)
    hist = []
    for seq, rows in rec.replies():
        for window, new_rows, first in win.add_rows(seq, rows):
            acc.add(ens.predict(window), new_rows, first)
            s = acc.summary()
            hist.append((win.rows_total / S.SAMPLE_HZ, s["steps"], s["steps_sd"], s["dist_m"], s["dist_sd"]))
    s = acc.summary()
    base = baseline_steps(rec, label_info)
    return s, hist, base


def baseline_steps(rec, label_info):
    """Plain peak counter with one fixed threshold (no network), for comparison."""
    sig = L.swing_signal(rec.rows, label_info["axis"], label_info["sign"])
    return 2 * len(L.peaks(sig, rec.t, label_info["threshold"]))


def train_all(data_dir=config.DATA_DIR, include_hw6=True, log=print, should_stop=None):
    """Recordings -> (model JSON for the board, training report)."""
    torch.set_num_threads(4)
    t_start = time.time()
    recs = dataset.load_recordings(data_dir, include_hw6=include_hw6, log=log)
    log("Recordings:")
    dataset.summarize(recs, log)
    labs, label_info = L.label_all(recs, log=log)
    train_recs = [r for r in recs if r.kind != "long"]
    tests = [r for r in recs if r.kind == "long" and labs[r.name].ok]
    halls = [r for r in train_recs if r.kind == "hall" and labs[r.name].ok]
    if len(halls) < 2:
        raise ValueError(f"Need at least 2 usable hallway walks, have {len(halls)}. Record more first.")

    log(f"\n1. Leave-one-walk-out: {len(halls)} folds x {FOLD_MEMBERS} networks x {MAX_EPOCHS} epochs")
    folds = leave_one_walk_out(train_recs, labs, log, should_stop)
    epochs, curve = pick_epochs(folds)
    log(f"   held-out loss is lowest at epoch {epochs} -> the final networks train for {epochs} epochs")
    calib, val = calibrate(folds, epochs, log)

    log(f"\n2. Final ensemble: {N_MEMBERS} networks on all training recordings, {epochs} epochs each")
    sources = sources_for(train_recs, labs)
    results = run_jobs([(sources, epochs, SEED + 1000 + m, None) for m in range(N_MEMBERS)], log, should_stop,
                       lambda j: f"  network {j + 1}/{N_MEMBERS} done")
    models = [to_model(state) for state, _, _ in results]
    train_curves = [[round(v, 4) for v in losses] for _, losses, _ in results]

    checks = gradient_check(models[0], sources, log)

    # Unfamiliar-motion warning: well above the members' disagreement on training data.
    plain = np.concatenate([board_windows(r, labs[r.name])[0] for r in train_recs if labs[r.name].ok])
    with torch.no_grad():
        member_rates = np.stack([m(torch.from_numpy(plain)).numpy()[:, 0] for m in models])
    epi = member_rates.std(axis=0)
    calib["epistemic_warn"] = round(float(1.5 * np.quantile(epi, 0.99)), 4)
    log(f"Unfamiliar-motion light: networks disagree by more than {calib['epistemic_warn']:.3f} strides/s "
        f"(1.5 x the 99th percentile on training windows)")

    model_json = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scales": SCALES, "window": S.WINDOW, "hop": S.HOP, "sample_hz": S.SAMPLE_HZ,
        "calibration": calib,
        "epochs": epochs,
        "parameters_per_network": int(sum(p.numel() for p in models[0].parameters())),
        "label_info": label_info,
        "trained_on": [r.name for r in train_recs if labs[r.name].ok],
        "validation": {"walks": val["walks"], "reliability": val["reliability"]},
        "members": [export_member(m) for m in models],
    }
    check_numpy_matches(models, model_json, plain[:40], log)

    ens = S.StepNetEnsemble(model_json)
    test_results = []
    if tests:
        log("\n3. Final test on the long walk(s) (never trained on):")
    for r in tests:
        s, hist, base = test_walk(r, labs[r.name], ens, label_info)
        inside = s["steps_lo"] <= r.steps <= s["steps_hi"]
        d_inside = s["dist_lo"] <= r.distance_m <= s["dist_hi"] if r.distance_m else None
        log(f"  {r.name}: counted {r.steps} steps, network {s['steps_shown']} ± {S.Z95 * s['steps_sd']:.0f} "
            f"(95%: {s['steps_lo']}-{s['steps_hi']}) {'inside' if inside else 'OUTSIDE'} the range; "
            f"plain peak counter {base}")
        if r.distance_m:
            log(f"  {'':<{len(r.name)}}  walked {r.distance_m:.0f} m, network {s['dist_m']:.0f} ± "
                f"{S.Z95 * s['dist_sd']:.0f} m {'inside' if d_inside else 'OUTSIDE'} the range")
        test_results.append({"walk": r.name, "true_steps": r.steps, "steps": s["steps"], "steps_sd": s["steps_sd"],
                             "inside_95": inside, "baseline_steps": base, "true_m": r.distance_m,
                             "dist_m": s["dist_m"], "dist_sd": s["dist_sd"], "dist_inside_95": d_inside,
                             "history": [[round(v, 2) for v in h] for h in hist]})
    model_json["test"] = [{k: v for k, v in t.items() if k != "history"} for t in test_results]

    base_rows = [(r.name, r.steps, baseline_steps(r, label_info)) for r in halls]
    report = {
        "created": model_json["created"],
        "label_info": label_info,
        "labels": {r.name: {"ok": labs[r.name].ok, "note": labs[r.name].note,
                            "strides": [round(float(v), 3) for v in labs[r.name].stride_times]} for r in recs},
        "epochs": epochs,
        "heldout_nll_curve": [round(float(v), 4) for v in curve],
        "train_loss_curves": train_curves,
        "calibration": calib,
        "validation": val,
        "gradient_check": checks,
        "test": test_results,
        "baseline_hall": [{"walk": n, "true_steps": t, "baseline_steps": b} for n, t, b in base_rows],
        "seconds": round(time.time() - t_start, 1),
    }
    log(f"\nDone in {report['seconds']:.0f} s.")
    return model_json, report


def install(model_json, report):
    """Make this model the one the app uses; the old one is kept in model_history/."""
    folder = config.MODEL_HISTORY_DIR / time.strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    if S.MODEL_FILE.exists():
        shutil.copy2(S.MODEL_FILE, folder / "replaced_stepnet_model.json")
    text = json.dumps(model_json, separators=(",", ":"))
    (folder / S.MODEL_FILE.name).write_text(text, encoding="utf-8")
    S.MODEL_FILE.write_text(text, encoding="utf-8")
    config.TRAINING_REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--no-hw6", action="store_true", help="don't add HW6's standing/walking recordings")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="write the model here (and the report next to it)")
    where.add_argument("--install", action="store_true", help="install in the app (old one kept in model_history/)")
    args = ap.parse_args()
    try:
        model_json, report = train_all(args.data, include_hw6=not args.no_hw6)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
    if args.install:
        folder = install(model_json, report)
        print(f"Installed {S.MODEL_FILE} (history: {folder}). Deploy: python tools/deploy.py step-counter-wearable")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(model_json, f, separators=(",", ":"))
        report_path = args.out.rsplit(".", 1)[0] + "_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"Saved {args.out} and {report_path}")


if __name__ == "__main__":
    main()
