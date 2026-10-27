"""Synthetic ankle IMU data, for testing the pipeline without the board.

NOT used for the real calibration/training: the shapes are rough guesses
(heel-strike spikes, shin swing, pedaling rotation), only good enough to check
that every piece of code runs and the parts can tell the activities apart.

Sensor frame like the real mounting: +x up the shin, y forward, z sideways.
`rows_counts` reproduces the sketch: 200 Hz readings in raw counts, packed into
50 Hz rows of [mean ax..gz, max ax..az, min ax..az].
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402  (puts the board code on the path)
import classifiers as C  # noqa: E402

FS = 200
ACCEL_COUNTS_PER_G = 2048.0
GYRO_COUNTS_PER_DPS = 16.4
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def make_person(seed):
    """Random per-person differences: mounting angle, cadence, intensity."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(-10, 10))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k
    return {"seed": seed, "rot": rot, "cadence": rng.uniform(0.85, 1.15), "intensity": rng.uniform(0.8, 1.2)}


def _stride_amp(k, seed, lo, hi):
    """Repeatable random amplitude for stride number k."""
    out = np.empty(len(k))
    for i, kk in enumerate(k):
        out[i] = np.random.default_rng((seed * 100003 + int(kk)) & 0x7FFFFFFF).uniform(lo, hi)
    return out


def simulate(label, t, person, rng):
    """IMU readings at times t (s): (len(t), 6) in g and deg/s."""
    cad, inten, seed = person["cadence"], person["intensity"], person["seed"]
    n = len(t)
    dyn = np.zeros((n, 3))
    wx = wy = np.zeros(n)
    if label == "standing":
        theta = np.radians(2.0 * np.sin(2 * np.pi * 0.2 * t) + 1.0 * np.sin(2 * np.pi * 0.07 * t + 1))
        w = 2 * np.pi * 0.2 * np.radians(2.0) * np.cos(2 * np.pi * 0.2 * t)
        dyn[:, 1] = 0.02 * np.sin(2 * np.pi * 0.3 * t)
    elif label in ("walking", "running"):
        run = label == "running"
        f = (1.4 if run else 0.9) * cad
        amp = np.radians((45 if run else 25) * inten)
        ph = 2 * np.pi * f * t
        theta = amp * np.sin(ph) + 0.3 * amp * np.sin(2 * ph)
        w = amp * 2 * np.pi * f * (np.cos(ph) + 0.6 * np.cos(2 * ph))
        stride = np.floor(f * t)
        local = (f * t - stride) / f                       # seconds since this stride began
        uniq, inv = np.unique(stride, return_inverse=True)
        lo, hi = ((3.5, 6.0) if run else (1.2, 2.2))
        peak = _stride_amp(uniq, seed, lo * inten, hi * inten)[inv]
        width = 0.012 if run else 0.018
        dyn[:, 0] = peak * np.exp(-0.5 * ((local - 0.05) / width) ** 2)
        dyn[:, 0] += 0.3 * peak * np.exp(-0.5 * ((local - 0.11) / (2 * width)) ** 2)
        dyn[:, 1] = (1.2 if run else 0.4) * inten * np.sin(ph + 0.5)
        if run:
            dyn[:, 0] += 0.6 * inten * np.sin(2 * ph)       # body bounce
        wx = 20 * inten * np.sin(ph + 1.0)
        wy = 15 * inten * np.sin(2 * ph)
    elif label == "biking":
        f = 1.0 * cad
        ph = 2 * np.pi * f * t
        amp = np.radians(35 * inten)
        theta = np.radians(90) + amp * np.sin(ph)           # lying down: shin around horizontal
        w = amp * 2 * np.pi * f * np.cos(ph)
        dyn[:, 0] = 0.3 * inten * np.cos(ph)
        dyn[:, 1] = 0.3 * inten * np.sin(ph)
        wx = 10 * np.sin(ph)
    else:
        raise ValueError(label)
    grav = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    acc = grav + dyn
    gyro = np.stack([wx * np.ones(n), wy * np.ones(n), np.degrees(w)], axis=1)
    acc, gyro = acc @ person["rot"].T, gyro @ person["rot"].T
    acc += rng.normal(0, 0.015, acc.shape)
    gyro += rng.normal(0, 1.0, gyro.shape)
    return np.hstack([acc, gyro])


def to_counts(readings):
    c = np.empty_like(readings)
    c[:, :3] = np.clip(np.round(readings[:, :3] * ACCEL_COUNTS_PER_G), -32768, 32767)
    c[:, 3:] = np.clip(np.round(readings[:, 3:] * GYRO_COUNTS_PER_DPS), -32768, 32767)
    return c.astype(np.int64)


def rows_counts(counts):
    """200 Hz counts (4k, 6) -> sketch rows (k, 12) in counts, same math as pushRow()."""
    g = counts[: len(counts) // 4 * 4].reshape(-1, 4, 6)
    mean = np.trunc(g.sum(axis=1) / 4).astype(np.int64)    # C integer division truncates toward 0
    return np.hstack([mean, g[:, :, :3].max(axis=1), g[:, :, :3].min(axis=1)])


def counts_to_physical(rows):
    scale = np.array([1 / ACCEL_COUNTS_PER_G] * 3 + [1 / GYRO_COUNTS_PER_DPS] * 3 + [1 / ACCEL_COUNTS_PER_G] * 6)
    return rows * scale


def encode_reply(seq, rows):
    """Same string the sketch's get_samples() builds."""
    def put3(u):
        return B64[(u >> 12) & 0x0F] + B64[(u >> 6) & 0x3F] + B64[u & 0x3F]
    return put3(seq & 0xFFFF) + "".join(put3(int(v) + 32768) for row in rows for v in row)


def recording_rows(label, seconds, person, seed):
    """One synthetic recording: physical rows (n, 12) at 50 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS + rng.uniform(0, 100)
    return counts_to_physical(rows_counts(to_counts(simulate(label, t, person, rng))))
