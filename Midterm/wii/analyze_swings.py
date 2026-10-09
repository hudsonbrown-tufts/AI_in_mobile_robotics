"""Break a `wii_remote.py --log` recording into swings and summarize each one.

For every swing (gyro speed above SWING_DPS) it prints, in Wii Remote axes
(left, back, up), the strongest acceleration each way, the rotation
direction, and whether the raw accelerometer looked clipped -- enough to see
why the game reads a swing as forehand or backhand.

Usage:
    python analyze_swings.py                # wii/swing_log.csv
    python analyze_swings.py path/to/log.csv
"""

import csv
import sys
from pathlib import Path

SWING_DPS = 250.0       # a swing is where the gyro speed goes above this
MIN_GAP_S = 0.35        # samples closer than this belong to the same swing


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("swing_log.csv")
    rows = load(path)
    if not rows:
        sys.exit("empty log")
    t0 = rows[0]["t"]
    dur = rows[-1]["t"] - t0
    print(f"{path.name}: {len(rows)} samples over {dur:.1f} s ({len(rows) / max(dur, 1e-9):.0f} Hz)")

    raw_max = max(max(abs(r["ax_raw"]), abs(r["ay_raw"]), abs(r["az_raw"])) for r in rows)
    at_max = sum(1 for r in rows if max(abs(r["ax_raw"]), abs(r["ay_raw"]), abs(r["az_raw"])) >= raw_max * 0.995)
    print(f"largest raw accel reading: {raw_max:.0f} (hit by {at_max} samples"
          f"{' -> likely the sensor limit' if at_max > 5 else ''})")

    # Group swing samples.
    swings, cur, last_t = [], [], None
    for r in rows:
        w = (r["gyro_left_dps"] ** 2 + r["gyro_back_dps"] ** 2 + r["gyro_up_dps"] ** 2) ** 0.5
        if w > SWING_DPS:
            if last_t is not None and r["t"] - last_t > MIN_GAP_S and cur:
                swings.append(cur)
                cur = []
            cur.append(r)
            last_t = r["t"]
    if cur:
        swings.append(cur)

    print(f"\n{len(swings)} swings (gyro > {SWING_DPS:.0f} dps)\n")
    print(f"{'#':>2} {'t(s)':>6} {'n':>3}  {'accel left  min/max':>20} {'back min/max':>14} {'up min/max':>14}  "
          f"{'yaw dps (up axis)':>17} {'roll dps (back)':>15}  first strong push")
    for i, s in enumerate(swings, 1):
        def mm(k):
            vals = [r[k] for r in s]
            return min(vals), max(vals)
        (l0, l1), (b0, b1), (u0, u1) = mm("acc_left_g"), mm("acc_back_g"), mm("acc_up_g")
        yaw = sum(r["gyro_up_dps"] for r in s) / len(s)
        roll = sum(r["gyro_back_dps"] for r in s) / len(s)
        first = next((r for r in s if abs(r["acc_left_g"]) > 1.0), None)
        push = "none" if first is None else ("LEFT" if first["acc_left_g"] > 0 else "RIGHT")
        print(f"{i:>2} {s[0]['t'] - t0:6.1f} {len(s):3d}  {l0:+8.2f} / {l1:+5.2f} g {b0:+6.2f}/{b1:+5.2f} "
              f"{u0:+6.2f}/{u1:+5.2f}  {yaw:+17.0f} {roll:+15.0f}  {push}")

    still = [r for r in rows if (r["gyro_left_dps"] ** 2 + r["gyro_back_dps"] ** 2 + r["gyro_up_dps"] ** 2) ** 0.5 < 30]
    if still:
        n = len(still)
        print(f"\nresting average (Wii axes): left {sum(r['acc_left_g'] for r in still) / n:+.2f} g, "
              f"back {sum(r['acc_back_g'] for r in still) / n:+.2f} g, up {sum(r['acc_up_g'] for r in still) / n:+.2f} g")
    print("flipped (F) was", "ON" if rows[-1]["flipped"] else "off")


if __name__ == "__main__":
    main()
