"""Synthetic ankle IMU walks with KNOWN stride times, for testing HW7 without the board.

NOT used for the real training: the shapes are rough guesses (a shin swing per
stride, a heel-strike spike, gravity tilting with the shin), only good enough
to check that every piece of code runs and that the labels, the network and
the uncertainty come out right when the truth is known exactly.

The walk is driven by a stride phase phi(t): it goes up by 1 per stride, and
the shin swing peaks at phi = k + 0.5 (the same convention as labels.py). The
sensor frame is like HW6's synthetic one: +x up the shin, y forward, z sideways.
`rows_counts` reproduces the sketch: 200 Hz readings in raw counts, packed
into 50 Hz rows of [mean ax..gz, max ax..az, min ax..az].

    .venv/Scripts/python HW7/testing/synth.py        # writes HW7/testing/data_synthetic/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402  (puts the board code on the path)
import dataset  # noqa: E402

FS = 200
ACCEL_COUNTS_PER_G = 2048.0
GYRO_COUNTS_PER_DPS = 16.4
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
OUT_DIR = Path(__file__).resolve().parent / "data_synthetic"
HALL_M = 25.0

# Paces: (strides per second, stride length in m). One stride = 2 steps.
PACES = {"slow": (0.80, 1.15), "normal": (0.93, 1.38), "fast": (1.05, 1.58)}


def amp_for_length(length_m):
    """Shin swing amplitude (rad): longer strides swing the shin further."""
    return np.radians(32.0 * length_m / 1.38)


def imu_from_phase(t, phi, f, amp, yaw, rng):
    """IMU readings (len(t), 6) in g and deg/s from the stride phase.

    phi: strides so far; f: stride rate (1/s, 0 when standing); amp: swing
    amplitude (rad); yaw: extra rotation about the shin (deg/s, for turns).
    """
    u = phi - np.floor(phi)                               # where in the stride (0..1)
    ang = 2 * np.pi * (u - 0.5)
    theta = amp * (np.sin(ang) + 0.25 * np.sin(2 * ang))  # shin angle, 0 at u = 0
    w = amp * 2 * np.pi * f * (np.cos(ang) + 0.5 * np.cos(2 * ang))   # d theta / dt, peak at u = 0.5
    strike = np.where(f > 0, np.exp(-0.5 * ((u - 0.78) / 0.015) ** 2), 0.0)
    walking = (f > 0).astype(float)
    acc = np.stack([np.cos(theta), np.sin(theta), np.zeros(len(t))], axis=1)
    acc[:, 0] += strike * (1.2 + 0.8 * f) + walking * 0.15 * np.sin(4 * np.pi * u)
    acc[:, 1] += walking * 0.5 * f * np.sin(2 * np.pi * u + 0.5)
    acc[:, 2] += walking * 0.1 * np.sin(2 * np.pi * u + 1.3)
    gyro = np.stack([yaw + walking * 12 * np.sin(2 * np.pi * u + 1.0),
                     walking * 10 * np.sin(4 * np.pi * u),
                     np.degrees(w)], axis=1)
    acc += rng.normal(0, 0.015, acc.shape)
    gyro += rng.normal(0, 1.0, gyro.shape)
    return np.hstack([acc, gyro])


def timeline(parts, rng):
    """parts: ("stand", s) | ("walk", strides, f, length) | ("turn", strides, f, length).

    -> 200 Hz arrays t, phi, f, amp, yaw, plus the true stride times and distance.
    """
    t_all, phi_all, f_all, amp_all, yaw_all = [], [], [], [], []
    t0, phi0, dist = 0.0, 0.0, 0.0
    for part in parts:
        if part[0] == "stand":
            n = int(part[1] * FS)
            t = t0 + np.arange(n) / FS
            t_all.append(t)
            phi_all.append(np.full(n, phi0))
            f_all.append(np.zeros(n))
            amp_all.append(np.zeros(n))
            yaw_all.append(np.zeros(n))
        else:
            kind, strides, f, length = part
            # each stride gets its own rate and length (+-4%), like a real walk
            rates = f * rng.uniform(0.96, 1.04, strides)
            lengths = length * rng.uniform(0.96, 1.04, strides)
            for rate, ln in zip(rates, lengths):
                n = int(round(FS / rate))
                t = t0 + np.arange(n) / FS
                t_all.append(t)
                phi_all.append(phi0 + np.arange(n) / n)
                f_all.append(np.full(n, rate))
                amp_all.append(np.full(n, amp_for_length(ln)))
                yaw_all.append(np.full(n, 140.0 if kind == "turn" else 0.0))
                t0 += n / FS
                phi0 += 1.0
                dist += ln
            continue
        t0 += part[1]
    t = np.concatenate(t_all)
    phi = np.concatenate(phi_all)
    true_strides = np.interp(np.arange(int(phi0)) + 0.5, phi, t)
    return t, phi, np.concatenate(f_all), np.concatenate(amp_all), np.concatenate(yaw_all), true_strides, dist


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


def recording(parts, seed):
    """-> physical rows (n, 12) at 50 Hz, true stride times (s), true distance (m)."""
    rng = np.random.default_rng(seed)
    t, phi, f, amp, yaw, strides, dist = timeline(parts, rng)
    rows = counts_to_physical(rows_counts(to_counts(imu_from_phase(t, phi, f, amp, yaw, rng))))
    return rows, strides, dist


def walk_parts(pace, lengths, hall_m=HALL_M):
    """Back and forth along the hallway `lengths` times, with a turn at each end."""
    f, length = PACES[pace]
    parts = [("stand", 3.0)]
    for i in range(lengths):
        parts.append(("walk", int(round(hall_m / length)), f, length))
        if i < lengths - 1:
            parts.append(("turn", 2, 0.85 * f, 0.4))
    parts.append(("stand", 3.0))
    return parts


def make_dataset(out_dir=OUT_DIR, seed=1, log=print):
    """6 there-and-back hallway walks (2 slow, 2 normal, 2 fast), one long walk and 30 s standing."""
    rng = np.random.default_rng(seed)
    folder = out_dir / config.PERSON
    if folder.exists():
        for p in folder.glob("*"):
            p.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    plan = [("hall", "slow", 2), ("hall", "normal", 2), ("hall", "fast", 2),
            ("hall", "slow", 2), ("hall", "normal", 2), ("hall", "fast", 2), ("long", "normal", 8),
            ("standing", None, 30)]
    for i, (kind, pace, n) in enumerate(plan):
        if kind == "standing":
            rows, strides, dist = recording([("stand", n)], seed * 1000 + i)
            meta = None
        else:
            rows, strides, dist = recording(walk_parts(pace, n), seed * 1000 + i)
            steps = 2 * len(strides) - (1 if rng.random() < 0.5 else 0)   # odd counts happen
            meta = {"steps": int(steps), "lengths": n, "hall_m": HALL_M, "distance_m": HALL_M * n,
                    "true_strides": len(strides), "true_distance_m": round(dist, 2), "pace": pace}
        path = folder / f"{kind}_20260101_{i:06d}.csv"
        seqs = 1000 * i + np.arange(len(rows))
        dataset.save_recording(path, seqs, rows, meta)
        if log:
            log(f"  {path.name}: {len(rows) / 50:.0f} s" + (f", {meta['steps']} steps, {meta['distance_m']} m "
                                                            f"(true {meta['true_distance_m']} m)" if meta else ""))
    return folder


if __name__ == "__main__":
    print(f"Writing synthetic recordings to {OUT_DIR}")
    make_dataset()
