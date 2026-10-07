"""The LEGO Double Motor as a Wii-Remote-style paddle: swing detection and
paddle orientation from its IMU, and haptic "buzz" feedback by pulsing its
motors.

Hold the Double Motor by itself in your right hand.

Swing detection: angular speed w = sqrt(gx^2 + gy^2 + gz^2). When w rises
above SWING_THRESHOLD a swing starts; the swing is reported at its peak
(the moment w falls back below PEAK_DROP of the running max), which is
roughly when the paddle would meet the ball. The peak sets the strength.

Forehand vs backhand: a forehand and a backhand spin the hub in opposite
directions, so the gyro readings summed over the swing point opposite
ways. `calibrate` records a few of each; every swing is then classified by
which recording its direction is closer to. Uncalibrated, it falls back on
which way the IMU yaw turned during the swing (FOREHAND_YAW_SIGN).

Orientation: the hub's yaw/pitch/roll, relative to a neutral "paddle
facing the screen" pose captured during the countdown, become the paddle's
  aim   (face turned left/right -> where the shot goes),
  tilt  (face opened up/closed down -> how high and deep it goes),
  twist (rotation of the handle; drawn on screen).
Yaw drifts, so it is re-zeroed at the start of every point.

Haptics: short alternating-direction motor pulses on both sides make the
hub buzz in your hand. While a pattern plays (and briefly after), swing
detection is muted so the motor's own shaking isn't read as a swing.

Usage:
    python paddle_imu.py check       # live gyro / orientation + swing events, buzzes on every swing
    python paddle_imu.py calibrate   # record forehand + backhand swings
    python paddle_imu.py buzz        # play each haptic pattern once
"""

import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import legoeducation as le

from game import Swing

# Update these to match the Connection Card printed on/with your Double Motor.
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "5164"

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

# --- Swing detection (gyro units as reported by the hub; tune with `check`) ---
SWING_THRESHOLD = 550.0     # w above this starts a possible swing
SWING_MIN_PEAK = 650.0      # ...and it only counts if w peaks at least this high (no small flicks)
SWING_FULL = 1600.0         # w at (or above) this = full-strength swing
PEAK_DROP = 0.85            # swing peak = w falls below this fraction of its max
MAX_SWING_S = 0.35          # report the swing after this long even if w hasn't dropped
SWING_COOLDOWN_S = 0.40     # ignore new swings for this long after one
RETURN_WINDOW_S = 0.90      # a swing the OPPOSITE way this soon after a real one is just the
                            # arm coming back (follow-through), not a new swing
RETURN_SIMILARITY = -0.3    # "opposite way" = direction dot product below this
STILL_W = 140.0             # w below this counts as holding still
CALM_W = 300.0              # w below this = not swinging; the hub's hold is read from here
HOLD_STILL_S = 1.5          # game setup: hold the paddle upright and still this long to zero the angles
SETUP_READ_S = 1.0          # ...counted only from this long after the step appears (time to read it
                            # and get the paddle into position), so being still beforehand can't skip it
HOLD_STILL_TIMEOUT_S = 10.0  # ...or take whatever it's doing after this long
HAPTIC_MUTE_S = 0.15        # keep ignoring the gyro this long after a haptic pattern
POLL_S = 0.004

# --- Forehand / backhand ---
CALIBRATION_SWINGS = 3      # swings recorded per stroke by `calibrate`
# On a backhand the hub is flipped -- the OTHER face of the motor leads toward the
# camera. The gyro is in the hub's own axes, so that flip cancels out the reversed
# swing direction and the spin alone looks about the same for both strokes. How the
# hub is held (its gravity direction from the accelerometer, just before the swing)
# tells the two faces apart, so it is weighted more than the spin.
HOLD_WEIGHT = 1.5
FOREHAND_YAW_SIGN = -1      # uncalibrated fallback: a righty's forehand turns yaw this way
                            # (hub yaw is + clockwise seen from above; flip if it's backwards)

# --- Orientation: which hub angle drives each paddle angle, and its sign ---
# Watch `python paddle_imu.py check` while you turn the paddle face left/right
# (aim), open/close it (tilt) and rotate the handle (twist); swap these
# around if the hub sits differently in your hand.
ANGLE_SOURCES = {
    "aim": ("yaw", +1),
    "tilt": ("pitch", +1),
    "twist": ("roll", +1),
}
IMU_ANGLE_PER_DEGREE = 10.0  # yaw/pitch/roll are reported in tenths of a degree

# --- Haptic patterns: (direction +1/-1/0 for pause, speed %, milliseconds) ---
BUZZ = [(+1, 100, 60), (-1, 100, 60)]
HAPTICS = {
    "hit": BUZZ,
    "good": BUZZ + [(+1, 100, 50)],
    "perfect": BUZZ + [(0, 0, 50)] + BUZZ,
    "soft": [(+1, 60, 50)],
    "miss": [(+1, 45, 110), (-1, 45, 110)] * 3,
    "start": [(+1, 80, 90), (0, 0, 120), (+1, 80, 90)],
    "win": [(+1, 100, 70), (-1, 100, 70)] * 4 + [(0, 0, 100), (+1, 100, 200)],
    "lose": [(+1, 40, 300), (0, 0, 100), (-1, 40, 400)],
}


def connect():
    motor = le.DoubleMotor()
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not motor.connected:
        raise RuntimeError("Could not connect to the Double Motor "
                           f"(card {CARD_SERIAL}). Is it on and broadcasting?")
    # Coast, so the motors don't lock up in your hand between pulses.
    motor.movement_set_end_state(le.MOTOR_END_STATE_COAST)
    motor.movement_stop()
    return motor


def read_imu(motor):
    """Latest IMU notification (object identity changes on each new sample)."""
    imu = motor.imu_device
    if math.isnan(imu.gyroscopeX):
        return None
    return imu


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 0 else [0.0, 0.0, 0.0]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Paddle:
    """Background IMU swing detector + orientation + haptics player for the Double Motor."""

    def __init__(self):
        self.motor = None
        self.status = "connecting..."
        self.omega = 0.0               # latest angular speed (for the GUI meter)
        self.sample_rate = 0.0
        self.last_swing = None         # (Swing, peak w)
        self._swings = queue.Queue()
        self._haptics = queue.Queue()
        self._haptic_until = 0.0
        self._stop = threading.Event()
        self._neutral = None           # {"yaw": deg, "pitch": deg, "roll": deg}
        self._cal_step = None          # None, or ["forehand"/"backhand", [recorded unit vectors]]
        self._cal_data = {}
        self._still_since = None
        self.calibration = self._load_calibration()   # {"forehand": [x,y,z], "backhand": [x,y,z]}

    # ---------------- public API ----------------

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    @property
    def connected(self):
        return self.motor is not None

    def get_swings(self):
        """All swings detected since the last call."""
        out = []
        while True:
            try:
                out.append(self._swings.get_nowait())
            except queue.Empty:
                return out

    def haptic(self, name):
        """Queue a haptic pattern. Anything still waiting is dropped, so feedback never lags."""
        if self.motor is None or name not in HAPTICS:
            return
        while not self._haptics.empty():
            try:
                self._haptics.get_nowait()
            except queue.Empty:
                break
        self._haptics.put(name)

    # --- orientation ---

    def _raw_angles(self):
        if self.motor is None:
            return None
        imu = self.motor.imu_device
        if math.isnan(imu.yaw):
            return None
        return {"yaw": imu.yaw / IMU_ANGLE_PER_DEGREE,
                "pitch": imu.pitch / IMU_ANGLE_PER_DEGREE,
                "roll": imu.roll / IMU_ANGLE_PER_DEGREE}

    def zero_orientation(self, yaw_only=False):
        """Take the current pose as 'paddle facing the screen'. yaw_only just removes yaw drift."""
        raw = self._raw_angles()
        if raw is None:
            return
        if yaw_only and self._neutral is not None:
            self._neutral["yaw"] = raw["yaw"]
        else:
            self._neutral = dict(raw)

    def orientation(self, forehand=None):
        """Paddle {"aim", "tilt", "twist"} in degrees relative to neutral, or None.

        With forehand=True/False, aim and tilt are measured from that stroke's
        calibrated "straight" angle instead: the hub turns over for a backhand, so
        its raw angles there are offset from (and can be flipped against) a forehand's.
        """
        raw = self._raw_angles()
        if raw is None:
            return None
        if self._neutral is None:
            self._neutral = dict(raw)
        rel = {k: wrap180(raw[k] - self._neutral[k]) for k in raw}
        out = {name: sign * rel[src] for name, (src, sign) in ANGLE_SOURCES.items()}
        if forehand is not None:
            out = self._stroke_relative(out, forehand)
        return out

    def _stroke_relative(self, o, forehand):
        stroke = "forehand" if forehand else "backhand"
        cal = self.calibration or {}
        o = dict(o)
        for k in ("aim", "tilt"):
            o[k] = wrap180(o[k] - cal.get(f"{stroke}_{k}", 0.0))
        if cal.get(f"{stroke}_tilt_sign", 1) < 0:
            o["tilt"] = -o["tilt"]
        return o

    # --- forehand/backhand calibration ---

    def calibrate(self):
        """Record CALIBRATION_SWINGS forehands, then the same number of backhands."""
        self._cal_step = ["forehand", []]
        self._cal_data = {}

    def cancel_calibration(self):
        self._cal_step = None

    def still_for(self, now):
        """Seconds the paddle has been held still (0 if it's moving)."""
        return 0.0 if self._still_since is None else now - self._still_since

    def held_still_since(self, now, step_start):
        """Seconds held still, counting only from SETUP_READ_S after step_start."""
        return max(0.0, min(self.still_for(now), now - step_start - SETUP_READ_S))

    @property
    def calibrating(self):
        return self._cal_step is not None

    @property
    def calibration_prompt(self):
        if self._cal_step is None:
            return None
        stroke, done = self._cal_step
        return f"CALIBRATING: swing a {stroke.upper()} ({len(done) + 1}/{CALIBRATION_SWINGS})"

    def close(self):
        self._stop.set()
        if self.motor is not None:
            try:
                self.motor.movement_stop()
                self.motor.disconnect()
            except Exception:
                pass

    # ---------------- threads ----------------

    def _run(self):
        try:
            self.motor = connect()
        except Exception as exc:
            self.status = f"not connected ({exc})"
            print(f"[paddle] {self.status}")
            return
        self.status = "connected"
        print("[paddle] Double Motor connected.")
        threading.Thread(target=self._haptics_loop, daemon=True).start()
        self._imu_loop()

    def _imu_loop(self):
        last = None
        in_swing = False
        peak = peak_t = start_t = 0.0
        gyro_sum = [0.0, 0.0, 0.0]
        start_yaw = 0.0
        peak_orient = None
        calm_accel = None                  # accelerometer while not swinging = how the hub is held
        held = (0.0, 0.0, 1.0)
        cooldown_until = 0.0
        last_dir, last_t = None, -1e9      # direction/time of the last real swing
        rate_count, rate_t = 0, time.monotonic()

        while not self._stop.is_set():
            imu = read_imu(self.motor)
            if imu is None or imu is last:
                time.sleep(POLL_S)
                continue
            last = imu
            now = time.monotonic()
            rate_count += 1
            if now - rate_t >= 1.0:
                self.sample_rate = rate_count / (now - rate_t)
                rate_count, rate_t = 0, now

            g = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
            w = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
            self.omega = w
            if w < CALM_W and not math.isnan(imu.accelerometerX):
                calm_accel = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
            if w < STILL_W:
                if self._still_since is None:
                    self._still_since = now
            else:
                self._still_since = None

            if now < self._haptic_until or now < cooldown_until:
                in_swing = False
                continue

            if not in_swing:
                if w > SWING_THRESHOLD:
                    in_swing, peak, peak_t, start_t = True, w, now, now
                    gyro_sum = list(g)
                    held = calm_accel or (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
                    if any(math.isnan(c) for c in held):
                        held = (0.0, 0.0, 1.0)
                    start_yaw = imu.yaw / IMU_ANGLE_PER_DEGREE
                    peak_orient = self.orientation()
                continue

            gyro_sum = [s + c for s, c in zip(gyro_sum, g)]
            if w > peak:
                peak, peak_t = w, now
                peak_orient = self.orientation()
            if w < PEAK_DROP * peak or now - start_t > MAX_SWING_S:
                in_swing = False
                if peak < SWING_MIN_PEAK:
                    continue   # too small to be a real swing
                cooldown_until = now + SWING_COOLDOWN_S
                direction = unit(gyro_sum)
                if (last_dir is not None and peak_t - last_t < RETURN_WINDOW_S
                        and dot(direction, last_dir) < RETURN_SIMILARITY):
                    continue   # the arm swinging back after the last swing -- ignore it
                last_dir, last_t = direction, peak_t
                yaw_turn = wrap180(imu.yaw / IMU_ANGLE_PER_DEGREE - start_yaw)
                self._emit(peak_t, peak, direction, unit(held), yaw_turn, peak_orient)

    def _classify(self, direction, held, yaw_turn):
        """True = forehand, False = backhand.

        Calibrated: nearest stroke by spin direction AND by which face of the hub
        is toward the camera (gravity direction), the latter weighted more.
        """
        cal = self.calibration
        if cal:
            def score(stroke):
                sc = dot(direction, cal[stroke])
                if f"{stroke}_hold" in cal:
                    sc += HOLD_WEIGHT * dot(held, cal[f"{stroke}_hold"])
                return sc
            return score("forehand") >= score("backhand")
        return yaw_turn * FOREHAND_YAW_SIGN > 0

    def _emit(self, t, peak, direction, held, yaw_turn, orient):
        strength = min(1.0, max(0.0, (peak - SWING_MIN_PEAK) / (SWING_FULL - SWING_MIN_PEAK)))
        orient = orient or {"aim": 0.0, "tilt": 0.0, "twist": 0.0}

        if self._cal_step is not None:
            stroke, done = self._cal_step
            done.append((direction, held, orient["aim"], orient["tilt"]))
            print(f"[paddle] {stroke} {len(done)}/{CALIBRATION_SWINGS} recorded")
            self.haptic("soft")
            if len(done) >= CALIBRATION_SWINGS:
                self._cal_data[stroke] = unit([sum(d[0][i] for d in done) for i in range(3)])
                self._cal_data[f"{stroke}_hold"] = unit([sum(d[1][i] for d in done) for i in range(3)])
                # This stroke's natural "straight" face angles (circular mean of aim, mean of tilt).
                aims = [math.radians(d[2]) for d in done]
                self._cal_data[f"{stroke}_aim"] = math.degrees(math.atan2(
                    sum(math.sin(a) for a in aims), sum(math.cos(a) for a in aims)))
                self._cal_data[f"{stroke}_tilt"] = sum(d[3] for d in done) / len(done)
                if stroke == "forehand":
                    self._cal_step = ["backhand", []]
                else:
                    # If the hub is upside down for backhands, opening the face reads backwards.
                    flipped = dot(self._cal_data["forehand_hold"], self._cal_data["backhand_hold"]) < 0
                    self._cal_data["backhand_tilt_sign"] = -1 if flipped else 1
                    self.calibration = self._cal_data
                    self._save_calibration()
                    self._cal_step = None
                    spin = dot(self.calibration["forehand"], self.calibration["backhand"])
                    hold = dot(self.calibration["forehand_hold"], self.calibration["backhand_hold"])
                    print(f"[paddle] calibrated. similarity: spin {spin:+.2f}, hold {hold:+.2f} "
                          "(lower = more distinct; if both are above +0.8, redo it)")
                    self.haptic("start")
            return

        forehand = self._classify(direction, held, yaw_turn)
        if self.calibration:
            orient = self._stroke_relative(orient, forehand)
        swing = Swing(t=t, strength=strength, forehand=forehand,
                      aim_deg=orient["aim"], tilt_deg=orient["tilt"])
        self.last_swing = (swing, peak)
        self._swings.put(swing)

    def _haptics_loop(self):
        while not self._stop.is_set():
            try:
                name = self._haptics.get(timeout=0.2)
            except queue.Empty:
                continue
            pattern = HAPTICS[name]
            print(f"[paddle] haptic: {name}")
            total = sum(ms for _, _, ms in pattern) / 1000
            self._haptic_until = time.monotonic() + total + HAPTIC_MUTE_S
            for direction, speed, ms in pattern:
                if direction != 0:
                    try:
                        self.motor.motor_run_for_time(
                            ms, motor=le.MOTOR_BOTH, speed=speed, blocking=False,
                            direction=(le.MOTOR_MOVE_DIRECTION_CLOCKWISE if direction > 0
                                       else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE))
                    except Exception as exc:
                        print(f"[paddle] haptic failed: {exc}")
                        break
                time.sleep(ms / 1000)
            self._haptic_until = time.monotonic() + HAPTIC_MUTE_S

    # ---------------- calibration file ----------------

    def _load_calibration(self):
        try:
            data = json.loads(CALIBRATION_PATH.read_text())
        except (OSError, ValueError):
            return None
        # Older files only stored a forehand axis; those need re-calibrating.
        return data if "forehand" in data and "backhand" in data else None

    def _save_calibration(self):
        CALIBRATION_PATH.write_text(json.dumps(self.calibration, indent=2))


# ---------------- command line tools ----------------

def _wait_for_connection(paddle):
    while not paddle.connected and not paddle.status.startswith("not"):
        time.sleep(0.1)
    return paddle.connected


def check():
    paddle = Paddle().start()
    print("Connecting... hold the paddle facing the screen (that becomes neutral), then swing.")
    print("Ctrl+C to stop.")
    if not _wait_for_connection(paddle):
        return
    time.sleep(1.0)
    paddle.zero_orientation()
    try:
        while True:
            time.sleep(0.1)
            o = paddle.orientation() or {"aim": 0, "tilt": 0, "twist": 0}
            bar = "#" * int(min(paddle.omega, 2 * SWING_FULL) / (2 * SWING_FULL) * 30)
            print(f"w={paddle.omega:6.0f} {paddle.sample_rate:3.0f} Hz |{bar:<30}| "
                  f"aim {o['aim']:+6.1f}  tilt {o['tilt']:+6.1f}  twist {o['twist']:+6.1f}")
            for s in paddle.get_swings():
                print(f"   >>> {'FOREHAND' if s.forehand else 'BACKHAND'}  strength {s.strength:.2f}  "
                      f"aim {s.aim_deg:+.0f}  tilt {s.tilt_deg:+.0f}  -> buzz")
                paddle.haptic("perfect" if s.strength > 0.7 else "good" if s.strength > 0.3 else "hit")
    except KeyboardInterrupt:
        pass
    finally:
        paddle.close()


def calibrate():
    paddle = Paddle().start()
    if not _wait_for_connection(paddle):
        return
    paddle.calibrate()
    last = None
    try:
        while paddle.calibrating:
            if paddle.calibration_prompt != last:
                last = paddle.calibration_prompt
                print(last)
            time.sleep(0.05)
        time.sleep(0.6)
    except KeyboardInterrupt:
        pass
    finally:
        paddle.close()


def buzz():
    paddle = Paddle().start()
    if not _wait_for_connection(paddle):
        return
    for name in HAPTICS:
        print(f"haptic: {name}")
        paddle.haptic(name)
        time.sleep(1.8)
    paddle.close()


if __name__ == "__main__":
    {"check": check, "calibrate": calibrate, "buzz": buzz}.get(
        sys.argv[1] if len(sys.argv) > 1 else "check", check)()
