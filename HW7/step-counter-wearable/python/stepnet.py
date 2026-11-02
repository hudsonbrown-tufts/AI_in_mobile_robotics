"""Step network + step/distance adding with uncertainty, for the HW7 ankle step counter.

Used by the board (main.py) and by the laptop scripts (train_stepnet.py,
replay.py, make_figures.py), so the board and the laptop run exactly the same
code. Only numpy: the network is trained with PyTorch on the laptop
(train_stepnet.py) and its weights are saved in stepnet_model.json.

Sections:
    SHARED       - sample layout, window size
    NETWORK      - forward pass of one StrideNet and of the 5-network ensemble
    WINDOWS      - turning the stream of rows into 2 s windows every 0.24 s
    ADDING UP    - StepAccumulator: steps, distance and their uncertainty

HOW_THE_NET_WORKS.md explains all of it in depth.
"""

import json
import math
from pathlib import Path

import numpy as np


# =============================================================================
# SHARED
# =============================================================================

SAMPLE_HZ = 50          # rate of the rows the sketch sends (it reads the IMU at 200 Hz)
WINDOW = 100            # rows per window = 2.0 s
HOP = 12                # a new window every 12 rows = 0.24 s
WINDOW_S = WINDOW / SAMPLE_HZ

# One sample row (g and deg/s). The sketch sends 12 values (mean of 4 readings
# for all 6 channels, plus the max/min accel HW6 Part 1 needed); the network
# uses the 6 means.
COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz",
           "ax_max", "ay_max", "az_max", "ax_min", "ay_min", "az_min"]
COL = {name: i for i, name in enumerate(COLUMNS)}
NET_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "stepnet_model.json"

# "How sure" = the chance that the true count is within this many steps of the
# shown count: 1% of the count, but at least 3 steps.
SURE_TOLERANCE_FRAC = 0.01
SURE_TOLERANCE_MIN = 3
Z95 = 1.96


# =============================================================================
# NETWORK (one member: StrideNet)
# =============================================================================
#
#   input (6, 100): ax ay az / 2 g, gx gy gz / 500 deg/s
#   Conv1d(6->16, k7) ReLU MaxPool2 -> Conv1d(16->32, k5) ReLU MaxPool2
#   -> Conv1d(32->32, k5) ReLU -> average over time (32 numbers)
#   -> rate head  Linear(32->2): stride rate mean, spread (strides/s)
#   -> speed head Linear(32->2): walking speed mean, spread (m/s)
# Means and spreads go through softplus so they are never negative; the
# spreads get + SIGMA_FLOOR so they are never exactly 0.

SIGMA_FLOOR = 0.01


def net_input(window, scales):
    """(WINDOW, 12) sample rows -> (6, WINDOW) scaled network input."""
    x = window[:, [COL[c] for c in NET_CHANNELS]].T
    return x / np.asarray(scales, dtype=np.float64)[:, None]


def conv1d(x, weight, bias):
    """Valid 1D convolution. x: (C_in, T), weight: (C_out, C_in, K)."""
    k = weight.shape[2]
    patches = np.lib.stride_tricks.sliding_window_view(x, k, axis=1)   # (C_in, T-K+1, K)
    return np.einsum("oik,itk->ot", weight, patches) + bias[:, None]


def maxpool2(x):
    t = x.shape[1] // 2 * 2
    return x[:, :t].reshape(x.shape[0], t // 2, 2).max(axis=2)


def softplus(z):
    return np.logaddexp(0.0, z)


class StrideNet:
    """Numpy forward pass of one trained member (same layer names as PyTorch)."""

    def __init__(self, weights):
        w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.convs = [(w["conv1.weight"], w["conv1.bias"], True),
                      (w["conv2.weight"], w["conv2.bias"], True),
                      (w["conv3.weight"], w["conv3.bias"], False)]
        self.rate_w, self.rate_b = w["rate_head.weight"], w["rate_head.bias"]
        self.speed_w, self.speed_b = w["speed_head.weight"], w["speed_head.bias"]

    def forward(self, x, keep=False):
        """x: (6, WINDOW) scaled input -> (rate mu, rate sd, speed mu, speed sd).

        With keep=True also returns every layer's output (for figures/animation).
        """
        acts = {"input": x}
        for i, (weight, bias, pool) in enumerate(self.convs, start=1):
            x = conv1d(x, weight, bias)
            if keep:
                acts[f"conv{i}"] = x
            x = np.maximum(x, 0.0)
            if keep:
                acts[f"relu{i}"] = x
            if pool:
                x = maxpool2(x)
                if keep:
                    acts[f"pool{i}"] = x
        features = x.mean(axis=1)                        # average over time: 32 numbers
        r = self.rate_w @ features + self.rate_b
        s = self.speed_w @ features + self.speed_b
        out = (float(softplus(r[0])), float(softplus(r[1]) + SIGMA_FLOOR),
               float(softplus(s[0])), float(softplus(s[1]) + SIGMA_FLOOR))
        if keep:
            acts["features"] = features
            acts["rate_head"] = r
            acts["speed_head"] = s
            return out, acts
        return out


class StepNetEnsemble:
    """The 5 trained members + the uncertainty calibration from stepnet_model.json."""

    def __init__(self, model):
        self.scales = model["scales"]
        self.members = [StrideNet(m) for m in model["members"]]
        cal = model["calibration"]
        self.k_rate = cal["k_rate"]                # scale on the totals (true / predicted on held-out walks)
        self.k_speed = cal["k_speed"]
        self.c_rate = cal["c_rate"]                # scale on the window spreads (fit on held-out walks)
        self.c_speed = cal["c_speed"]
        self.s_steps = cal["s_steps"]              # systematic relative error of a whole walk
        self.s_dist = cal["s_dist"]
        self.epistemic_warn = cal["epistemic_warn"]  # rate disagreement above this = unfamiliar motion
        self.created = model.get("created")
        self.info = {k: v for k, v in model.items() if k != "members"}

    @classmethod
    def load(cls, path=MODEL_FILE):
        with open(path) as f:
            return cls(json.load(f))

    def member_outputs(self, window):
        """(n_members, 4): rate mu, rate sd, speed mu, speed sd for each member."""
        x = net_input(window, self.scales)
        return np.array([m.forward(x) for m in self.members])

    def predict(self, window):
        """Combine the members for one (WINDOW, 12) window.

        mean     = average of the members' means
        spread^2 = average of their spreads^2   (aleatoric: "this window is ambiguous")
                 + variance of their means      (epistemic: "the members disagree")
        then times the calibration factor c.
        """
        out = self.member_outputs(window)
        return combine(out, self.c_rate, self.c_speed)


def combine(out, c_rate=1.0, c_speed=1.0):
    """Member outputs (n, 4) -> one prediction dict (also used by training)."""
    rate_mu, rate_sd, speed_mu, speed_sd = out.T
    rate_alea = float(np.sqrt(np.mean(rate_sd ** 2)))
    rate_epi = float(np.std(rate_mu))
    speed_alea = float(np.sqrt(np.mean(speed_sd ** 2)))
    speed_epi = float(np.std(speed_mu))
    return {
        "rate": float(rate_mu.mean()),
        "rate_sd": c_rate * math.hypot(rate_alea, rate_epi),
        "rate_aleatoric": rate_alea,
        "rate_epistemic": rate_epi,
        "speed": float(speed_mu.mean()),
        "speed_sd": c_speed * math.hypot(speed_alea, speed_epi),
        "speed_aleatoric": speed_alea,
        "speed_epistemic": speed_epi,
        "members_rate": [round(float(v), 4) for v in rate_mu],
    }


# =============================================================================
# WINDOWS
# =============================================================================

class Windower:
    """Keeps the last WINDOW rows and hands out a window every HOP new rows.

    A hole of more than max_gap rows (Bridge or WiFi stall) restarts the window,
    so no window mixes data from both sides of the hole.
    """

    def __init__(self, max_gap=10):
        self.max_gap = max_gap
        self.window = np.zeros((0, len(COLUMNS)))
        self.new_rows = 0
        self.expected_seq = None
        self.dropped = 0
        self.rows_total = 0
        self.restarted = True        # the next window is the first of a run

    def add_rows(self, seq, rows):
        """Add rows from one reply. Returns a list of (window, new_rows, first)."""
        if self.expected_seq is not None:
            gap = (seq - self.expected_seq) % 65536
            if gap:
                self.dropped += gap
                if gap > self.max_gap:     # too much missing: start the window again
                    self.window = self.window[:0]
                    self.new_rows = 0
                    self.restarted = True
        self.expected_seq = (seq + len(rows)) % 65536
        self.rows_total += len(rows)
        self.window = np.vstack([self.window, rows])[-WINDOW:]
        self.new_rows += len(rows)
        if len(self.window) == WINDOW and self.new_rows >= HOP:
            out = [(self.window.copy(), self.new_rows, self.restarted)]
            self.new_rows = 0
            self.restarted = False
            return out
        return []


# =============================================================================
# ADDING UP: steps, distance and their uncertainty
# =============================================================================
#
# Every window credits `dt` seconds of walking: normally its newest 0.24 s.
#   strides  += k_rate  * rate  * dt          distance += k_speed * speed * dt
# k_rate / k_speed correct the small overall bias measured on held-out walks
# (e.g. the network reacts a little late when a walk starts); about 1.0.
# The first window of a run credits WINDOW_S / 2 = 1 s. (Each moment is seen by
# 8 overlapping windows; the first window stands in for the ones that never
# happened before it, which on average saw half of its 2 s.)
#
# Uncertainty, three parts (in strides; steps = 2 x strides):
#   random      the window spreads. Windows overlap, so neighbouring credits
#               share their error: treat credits within one 2 s window as fully
#               correlated and separate windows as independent:
#                   var_random += WINDOW_S * dt * rate_sd^2
#               (grows like the square root of time)
#   systematic  a whole walk can be off by a few percent: (s_steps * steps)^2
#               (grows in proportion to the count)
#   rounding    which foot moved first; the ankle can't see the left foot's
#               first or last step: +-1 step, uniform -> 1/3 steps^2
# Distance: the same random and systematic parts (no rounding).
#
# Standing gate: the network's rate can't be exactly 0 (softplus), and while
# standing it says about 0.02 strides/s, which would add about 2 steps per
# minute of standing. Windows below MIN_RATE (18 steps/min; real walking is
# 90+) count as standing: nothing is added, not even uncertainty.

MIN_RATE = 0.15            # strides/s


class StepAccumulator:
    def __init__(self, ensemble):
        self.ens = ensemble
        self.reset()

    def reset(self):
        self.strides = 0.0
        self.var_strides = 0.0
        self.dist = 0.0
        self.var_dist = 0.0
        self.walk_s = 0.0          # seconds credited with a real stride rate
        self.last = None           # the last window's prediction

    def add(self, pred, new_rows, first):
        self.last = pred
        if pred["rate"] < MIN_RATE:            # standing
            return
        dt = WINDOW_S / 2 if first else new_rows / SAMPLE_HZ
        k, ks = self.ens.k_rate, self.ens.k_speed
        self.strides += k * pred["rate"] * dt
        self.var_strides += WINDOW_S * dt * (k * pred["rate_sd"]) ** 2
        self.dist += ks * pred["speed"] * dt
        self.var_dist += WINDOW_S * dt * (ks * pred["speed_sd"]) ** 2
        self.walk_s += dt

    def summary(self):
        steps = 2 * self.strides
        sd_random = 2 * math.sqrt(self.var_strides)
        sd_system = self.ens.s_steps * steps
        sd_round = math.sqrt(1 / 3) if steps >= 0.5 else 0.0
        sd = math.sqrt(sd_random ** 2 + sd_system ** 2 + sd_round ** 2)
        tol = max(SURE_TOLERANCE_MIN, SURE_TOLERANCE_FRAC * steps)
        sure = 1.0 if sd < 1e-9 else math.erf(tol / (sd * math.sqrt(2)))

        d_random = math.sqrt(self.var_dist)
        d_system = self.ens.s_dist * self.dist
        d_sd = math.hypot(d_random, d_system)

        p = self.last or {}
        rate, rate_sd = p.get("rate", 0.0), p.get("rate_sd", 0.0)
        return {
            "steps": round(steps, 2),
            "steps_shown": int(round(steps)),
            "steps_sd": round(sd, 2),
            "steps_lo": max(0, int(math.floor(steps - Z95 * sd))),
            "steps_hi": int(math.ceil(steps + Z95 * sd)),
            "sure": round(sure, 3),
            "sure_tol": round(tol, 1),
            "steps_sd_parts": {"random": round(sd_random, 2), "systematic": round(sd_system, 2),
                               "rounding": round(sd_round, 2)},
            "dist_m": round(self.dist, 2),
            "dist_sd": round(d_sd, 2),
            "dist_lo": round(max(0.0, self.dist - Z95 * d_sd), 1),
            "dist_hi": round(self.dist + Z95 * d_sd, 1),
            "dist_sd_parts": {"random": round(d_random, 2), "systematic": round(d_system, 2)},
            "cadence_spm": round(2 * 60 * rate, 1),          # steps per minute
            "cadence_sd": round(2 * 60 * rate_sd, 1),
            "speed": round(p.get("speed", 0.0), 3),
            "speed_sd": round(p.get("speed_sd", 0.0), 3),
            "epistemic": round(p.get("rate_epistemic", 0.0), 4),
            "aleatoric": round(p.get("rate_aleatoric", 0.0), 4),
            "members_rate": p.get("members_rate"),
            "unfamiliar": bool(p) and p.get("rate_epistemic", 0.0) > self.ens.epistemic_warn,
            "walk_s": round(self.walk_s, 1),
        }
