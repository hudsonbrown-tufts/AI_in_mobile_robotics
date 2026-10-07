"""The LEGO Double Motor as a Wii Remote: IMU in Wii Remote axes + shaft buttons.

Axes. The hub's IMU axes depend on how it sits in your hand, so a two-pose
calibration finds them:
    1. hold it like a Wii Remote aimed at the screen, top facing up, still
       -> the accelerometer's gravity reading is "up"; its size is 1 g
    2. tip it to point straight up at the ceiling, still
       -> gravity is now along the remote's "forward" axis
The gyro integrated during that 90-degree tip gives the gyro's units (deg/s
per raw unit) and confirms the accelerometer's sign. From up and forward we
build a rotation into Dolphin's Wii Remote frame:
    x = left, y = backward (toward you), z = up      (right-handed)
so a remote lying flat reads accel (0, 0, +1 g), like a real one.

Buttons. With the motors coasting, their shafts turn freely by hand. Turning
a shaft past SHAFT_PRESS_DEG from where it rests is a press (hold it there to
keep holding the button); turning it back releases. Attach a small LEGO
beam to each shaft as a lever/trigger.
"""

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import legoeducation as le

import _paths  # noqa: F401  (lets us import the ping pong modules one folder up)
from paddle_imu import HAPTICS, STILL_W, connect

CALIBRATION_PATH = Path(__file__).with_name("wii_calibration.json")

POLL_S = 0.003
STILL_S = 1.0                  # hold each calibration pose still this long
SETUP_READ_S = 1.0             # ...counted from this long after the step starts
TIP_MIN_ANGLE = 60.0           # pose 2 must be at least this far from pose 1 (degrees)
# +1: the accelerometer reads +1 g on its UP axis at rest (the usual convention). If, after
# calibrating, rolling the remote to the right makes Dolphin's preview roll LEFT, set -1 and
# recalibrate. (Two still poses can't tell these apart; only "which way is really up" can.)
ACCEL_SIGN = 1

# --- Shaft buttons ---
SHAFT_BUTTONS = {le.MOTOR_LEFT: "B", le.MOTOR_RIGHT: "A"}   # left shaft = trigger B, right = A
SHAFT_PRESS_DEG = 15.0         # turned this far from rest = pressed
SHAFT_RELEASE_DEG = 7.0        # back within this = released
SHAFT_STUCK_S = 6.0            # held this long without moving = treat as the new rest (auto-release)


def vadd(a, b): return [x + y for x, y in zip(a, b)]
def vsub(a, b): return [x - y for x, y in zip(a, b)]
def vscale(a, k): return [x * k for x in a]
def vdot(a, b): return sum(x * y for x, y in zip(a, b))
def vnorm(a): return math.sqrt(vdot(a, a))
def vunit(a):
    n = vnorm(a)
    return [x / n for x in a] if n else [0.0, 0.0, 0.0]
def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, vdot(vunit(a), vunit(b))))))


def snap_scale(estimate):
    """Gyro units are almost certainly a power of ten of deg/s; snap a noisy estimate to it."""
    if estimate <= 0:
        return 1.0
    p = 10 ** round(math.log10(estimate))
    return p if 0.6 < estimate / p < 1.6 else estimate


class Hub:
    def __init__(self):
        self.motor = None
        self.status = "connecting..."
        self.sample_rate = 0.0
        self.calibration = self._load()        # {"rows": 3x3, "g_raw": float, "gyro_scale": float}
        self.accel = (0.0, 0.0, 1.0)           # Wii Remote frame, g
        self.gyro = (0.0, 0.0, 0.0)            # Wii Remote frame, deg/s
        self.motion_t = 0.0
        self.raw_accel = None
        self.raw_gyro = None
        self.gyro_raw_mag = 0.0
        self.buttons = set()                   # from the shafts
        self.shaft_offsets = {}                # degrees turned from rest, per motor
        self._still_since = None
        self._still_accel = deque(maxlen=200)  # (t, accel) while still
        self._gyro_integral = [0.0, 0.0, 0.0]  # raw gyro integrated since the last reset
        self._last_t = None
        self._rest = {}
        self._pressed_since = {}
        self._haptic_until = 0.0
        self._stop = threading.Event()

    # ---------------- lifecycle ----------------

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    @property
    def connected(self):
        return self.motor is not None

    def close(self):
        self._stop.set()
        if self.motor is not None:
            try:
                self.motor.movement_stop()
                self.motor.disconnect()
            except Exception:
                pass

    def _run(self):
        try:
            self.motor = connect()
        except Exception as exc:
            self.status = f"not connected ({exc})"
            print(f"[hub] {self.status}")
            return
        self.status = "connected"
        print("[hub] Double Motor connected.")
        self._loop()

    # ---------------- sampling ----------------

    def _loop(self):
        last = None
        count, rate_t = 0, time.monotonic()
        while not self._stop.is_set():
            self._update_shafts(time.monotonic())
            imu = self.motor.imu_device
            if imu is last or math.isnan(imu.gyroscopeX) or math.isnan(imu.accelerometerX):
                time.sleep(POLL_S)
                continue
            last = imu
            now = time.monotonic()
            count += 1
            if now - rate_t >= 1.0:
                self.sample_rate, count, rate_t = count / (now - rate_t), 0, now

            a = [float(imu.accelerometerX), float(imu.accelerometerY), float(imu.accelerometerZ)]
            g = [float(imu.gyroscopeX), float(imu.gyroscopeY), float(imu.gyroscopeZ)]
            dt = 0.0 if self._last_t is None else min(0.1, now - self._last_t)
            self._last_t = now
            self._gyro_integral = vadd(self._gyro_integral, vscale(g, dt))
            self.raw_accel, self.raw_gyro = a, g
            self.gyro_raw_mag = vnorm(g)

            if self.gyro_raw_mag < STILL_W:
                if self._still_since is None:
                    self._still_since = now
                    self._still_accel.clear()
                self._still_accel.append((now, a))
            else:
                self._still_since = None

            cal = self.calibration
            if cal is not None:
                rows = cal["rows"]
                self.accel = tuple(vdot(r, a) / cal["g_raw"] for r in rows)
                self.gyro = tuple(vdot(r, g) * cal["gyro_scale"] for r in rows)
            self.motion_t = now

    def _update_shafts(self, now):
        pressed = set()
        for idx, name in SHAFT_BUTTONS.items():
            pos = self.motor.motor[idx].position
            if pos is None or (isinstance(pos, float) and math.isnan(pos)):
                continue
            if now < self._haptic_until:
                self._rest[idx] = pos   # buzzing moves the shaft: not a press
                continue
            rest = self._rest.setdefault(idx, pos)
            off = pos - rest
            self.shaft_offsets[idx] = off
            if idx in self._pressed_since:
                if abs(off) <= SHAFT_RELEASE_DEG:
                    del self._pressed_since[idx]
                elif now - self._pressed_since[idx][0] > SHAFT_STUCK_S and abs(pos - self._pressed_since[idx][1]) < 2:
                    # Left turned and not touched for a while: that's its new rest position.
                    self._rest[idx] = pos
                    del self._pressed_since[idx]
                else:
                    if abs(pos - self._pressed_since[idx][1]) >= 2:
                        self._pressed_since[idx] = (now, pos)
                    pressed.add(name)
            elif abs(off) >= SHAFT_PRESS_DEG:
                self._pressed_since[idx] = (now, pos)
                pressed.add(name)
        self.buttons = pressed

    # ---------------- calibration helpers ----------------

    def still_for(self, now):
        return 0.0 if self._still_since is None else now - self._still_since

    def held_still_since(self, now, step_start):
        return max(0.0, min(self.still_for(now), now - step_start - SETUP_READ_S))

    def still_average(self, seconds=0.6):
        """Mean raw accel over the last `seconds` of the current still period."""
        if not self._still_accel:
            return None
        t_end = self._still_accel[-1][0]
        pts = [a for t, a in self._still_accel if t_end - t <= seconds]
        return [sum(c) / len(pts) for c in zip(*pts)]

    def reset_gyro_integral(self):
        self._gyro_integral = [0.0, 0.0, 0.0]

    @property
    def gyro_integral(self):
        return list(self._gyro_integral)

    def solve_calibration(self, flat_accel, up_accel, gyro_integral):
        """Build the hub -> Wii Remote rotation from the two still poses. Returns a summary."""
        flat_accel = vscale(flat_accel, ACCEL_SIGN)
        up_accel = vscale(up_accel, ACCEL_SIGN)
        g_raw = (vnorm(flat_accel) + vnorm(up_accel)) / 2
        tip_angle = angle_deg(flat_accel, up_accel)
        up = vunit(flat_accel)
        forward = vunit(vsub(up_accel, vscale(up, vdot(up_accel, up))))
        left = vcross(up, forward)
        back = vscale(forward, -1)
        # The tip turned the remote ~90 degrees; the gyro's raw integral over it gives its units.
        turned = vnorm(gyro_integral)
        gyro_scale = snap_scale(tip_angle / turned) if turned > 0 else 1.0
        self.calibration = {"rows": [left, back, up], "g_raw": g_raw * ACCEL_SIGN,
                            "gyro_scale": gyro_scale, "tip_angle": tip_angle, "gyro_raw_turned": turned}
        self._save()
        return self.calibration

    def _load(self):
        try:
            return json.loads(CALIBRATION_PATH.read_text())
        except (OSError, ValueError):
            return None

    def _save(self):
        CALIBRATION_PATH.write_text(json.dumps(self.calibration, indent=2))

    # ---------------- haptics (optional) ----------------

    def haptic(self, name):
        """Fire-and-forget buzz; shaft buttons ignore the motion it causes."""
        if self.motor is None:
            return
        pattern = HAPTICS[name]
        self._haptic_until = time.monotonic() + sum(ms for *_, ms in pattern) / 1000 + 0.2

        def play():
            for direction, speed, ms in pattern:
                if direction:
                    try:
                        self.motor.motor_run_for_time(
                            ms, motor=le.MOTOR_BOTH, speed=speed, blocking=False,
                            direction=(le.MOTOR_MOVE_DIRECTION_CLOCKWISE if direction > 0
                                       else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE))
                    except Exception:
                        return
                time.sleep(ms / 1000)
        threading.Thread(target=play, daemon=True).start()


# ---------------- Wii Remote frame -> DSU frame ----------------
# Dolphin's DSU client turns DSU readings into its own Wii Remote frame as
#   accel_wm = (dsu_x, -dsu_z, dsu_y)      gyro_wm = (pitch, roll, -yaw)
# so the inverse below makes "Accel Up" etc. in Dolphin match the real remote.
# If Dolphin's Motion Input preview moves the wrong way, flip a sign here --
# or skip the question entirely with `python wii_remote.py --map`.
DSU_ACCEL_SIGNS = (1, 1, 1)
DSU_GYRO_SIGNS = (1, 1, 1)


def wiimote_to_dsu(accel, gyro):
    ax, ay, az = accel           # left, back, up (g)
    gx, gy, gz = gyro            # about left, back, up (deg/s)
    sa, sg = DSU_ACCEL_SIGNS, DSU_GYRO_SIGNS
    dsu_accel = (sa[0] * ax, sa[1] * az, sa[2] * -ay)
    dsu_gyro = (sg[0] * gx, sg[1] * -gz, sg[2] * gy)   # (pitch, yaw, roll)
    return dsu_accel, dsu_gyro
