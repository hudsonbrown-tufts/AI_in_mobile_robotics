"""The sumo robot's LEGO hardware: Double Motor (tank drive + IMU + wheel encoders)
and the front Color Sensor, both over Bluetooth with the legoeducation package.

`Robot.snapshot()` returns one `RobotSnapshot` with everything fusion needs,
already converted to the robot frame (forward wheel degrees count up, yaw rate
counterclockwise positive). The conversions come from drive_calibration.json
(written by the `d` drive calibration in sumo.py) or config.DEFAULT_DRIVE.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field

import legoeducation as le

import config

BLUE, GRAY, UNKNOWN = "BLUE", "GRAY", "?"


def _run_with_timeout(func, *args, timeout, **kwargs):
    """Run func in a background thread; True if it finished in time (BLE calls can hang). From HW3."""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive()


def _ok(*values):
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in values)


def robot_to_motor(drive, left, right):
    """Our (left, right) wheel command -> the Double Motor's (left, right) ports."""
    if drive.get("reverse"):
        left, right = -right, -left        # the motor's front is our back
    if drive.get("swap_sides"):
        left, right = right, left          # its left port drives our right wheel (or a mirrored camera)
    return left, right


def motor_to_robot_wheels(drive, motor_left, motor_right):
    """The motor ports' readings (left, right) -> (our left wheel, our right wheel)."""
    if bool(drive.get("reverse")) != bool(drive.get("swap_sides")):
        return motor_right, motor_left
    return motor_left, motor_right


def load_drive_calibration():
    drive = dict(config.DEFAULT_DRIVE)
    try:
        drive.update(json.loads(config.DRIVE_CALIBRATION_PATH.read_text()))
    except (OSError, ValueError):
        pass
    return drive


# ---------------- color classification ----------------

@dataclass
class ColorReading:
    label: str = UNKNOWN
    t: float = 0.0
    hue: float = math.nan
    saturation: float = math.nan
    value: float = math.nan
    reflection: float = math.nan
    rgb: tuple = (math.nan, math.nan, math.nan)
    lego_color: int = -1


class ColorClassifier:
    """BLUE tape vs GRAY floor.

    The floor is one uniform dark gray, so anything that clearly doesn't look like
    the floor is treated as tape. That catches deep blue AND light blue painter's
    tape (light blue is mostly much brighter than the floor, not very blue):
      - brighter than the floor by more than COLOR_BRIGHTER (reflection), or
      - bluer than the floor by more than COLOR_BLUER (blue share of R+G+B), or
      - the sensor's own color ID says blue/azure, or
      - nearer the recorded BLUE than the recorded GRAY (after `k` calibration).
    The floor reference is the recorded GRAY if there is one, otherwise it is
    learned from what the sensor sees most of the time.
    """

    def __init__(self):
        self.centroids = {}
        try:
            self.centroids = json.loads(config.COLOR_CALIBRATION_PATH.read_text())
        except (OSError, ValueError):
            pass
        self.floor = None             # (reflection, blue share) of the gray floor
        self._n = 0
        if GRAY in self.centroids:
            c = self.centroids[GRAY]
            self.floor = (c[4] * 100.0, c[2])
            self._n = 1000

    @staticmethod
    def features(rgb, reflection):
        r, g, b = rgb
        s = max(1.0, r + g + b)
        return [r / s, g / s, b / s, min(1.0, s / 1500.0), (reflection or 0) / 100.0]

    def classify(self, reading):
        if not _ok(*reading.rgb):
            return UNKNOWN
        r, g, b = reading.rgb
        refl = reading.reflection if _ok(reading.reflection) else 0.0
        share = b / max(1.0, r + g + b)
        votes = reading.lego_color in (le.LEGO_COLOR_BLUE, le.LEGO_COLOR_AZURE)
        if BLUE in self.centroids and GRAY in self.centroids:
            f = self.features(reading.rgb, reading.reflection)
            d = {k: sum((a - c) ** 2 for a, c in zip(f, cen)) for k, cen in self.centroids.items()}
            votes = votes or min(d, key=d.get) == BLUE
        if self.floor is not None:
            f_refl, f_share = self.floor
            votes = votes or refl > f_refl + max(config.COLOR_BRIGHTER, 0.3 * f_refl)
            votes = votes or share > f_share + config.COLOR_BLUER
        else:
            votes = votes or (b > 1.25 * r and b > 1.05 * g and (r + g + b) > 30)
        label = BLUE if votes else GRAY
        if label == GRAY and self._n < 1000:
            # Learn the floor from the first readings (the robot starts on the floor).
            self._n += 1
            if self.floor is None:
                self.floor = (refl, share)
            else:
                k = 1.0 / min(self._n, 200)
                self.floor = (self.floor[0] + k * (refl - self.floor[0]), self.floor[1] + k * (share - self.floor[1]))
        return label

    def record(self, label, readings):
        """Save the average of `readings` as the centroid for `label` (BLUE or GRAY)."""
        feats = [self.features(r.rgb, r.reflection) for r in readings if _ok(*r.rgb)]
        if not feats:
            return False
        self.centroids[label] = [sum(c) / len(feats) for c in zip(*feats)]
        config.COLOR_CALIBRATION_PATH.write_text(json.dumps(self.centroids, indent=2))
        if label == GRAY:
            c = self.centroids[GRAY]
            self.floor, self._n = (c[4] * 100.0, c[2]), 1000
        return True


# ---------------- snapshot ----------------

@dataclass
class RobotSnapshot:
    t: float
    enc_left: float = math.nan       # wheel degrees, + = that wheel rolling forward
    enc_right: float = math.nan
    gyro_yaw_dps: float = math.nan   # + = turning counterclockwise (ring frame)
    gyro_raw: tuple = (math.nan, math.nan, math.nan)
    accel_mg: tuple = (math.nan, math.nan, math.nan)
    imu_t: float = 0.0               # when the IMU sample arrived
    color: ColorReading = field(default_factory=ColorReading)
    cmd_left: int = 0
    cmd_right: int = 0


class Robot:
    def __init__(self):
        self.motor = None
        self.sensor = None
        self.drive = load_drive_calibration()
        self.classifier = ColorClassifier()
        self.status = "not connected"
        self._lock = threading.Lock()
        self._last_cmd = None
        self._last_send = 0.0
        self.cmd = (0, 0)
        self._imu_obj = None
        self._imu_t = 0.0
        self._color_obj = None
        self._color = ColorReading()
        self.dropped_replies = 0          # commands that never got a reply (connection dropped)
        self._watchdog = None
        self._closing = threading.Event()

    # ---------- lifecycle ----------

    @property
    def connected(self):
        return (self.motor is not None and bool(self.motor.connected)
                and self.sensor is not None and bool(self.sensor.connected))

    def _quiet_dropped_replies(self):
        """When the Bluetooth link drops, the LEGO library fails every command still waiting for a
        reply, and asyncio prints "Future exception was never retrieved" for each one. Count
        those instead of flooding the terminal; the watchdog reports the disconnect once."""
        try:
            from legoeducation._platform import _get_persistent_loop
        except ImportError:
            return
        loop = _get_persistent_loop()
        default = loop.get_exception_handler()

        def handler(lp, context):
            if isinstance(context.get("exception"), ConnectionError):
                self.dropped_replies += 1
                return
            if default is None:
                lp.default_exception_handler(context)
            else:
                default(lp, context)
        loop.set_exception_handler(handler)

    def connect(self):
        self._connect_devices()
        self._quiet_dropped_replies()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="ble-watchdog")
        self._watchdog.start()
        return self

    def _watch(self):
        """Reconnect a device that drops out (battery, range, or interference)."""
        while not self._closing.wait(config.RECONNECT_EVERY_S):
            if self.connected:
                continue
            which = [n for d, n in ((self.motor, "Double Motor"), (self.sensor, "Color Sensor"))
                     if d is None or not d.connected]
            self.status = f"{' + '.join(which)} disconnected: reconnecting..."
            print(f"[robot] {self.status}")
            try:
                self._connect_devices(only_missing=True)
                print("[robot] reconnected.")
            except Exception as exc:
                self.status = f"{' + '.join(which)} disconnected ({exc}); retrying"

    def _connect_devices(self, only_missing=False):
        if not (only_missing and self.motor is not None and self.motor.connected):
            print("[robot] connecting Double Motor (purple card)...")
            motor = le.DoubleMotor()
            motor.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
                          device_notification_delay=config.NOTIFICATION_MS)
            if not motor.connected:
                raise RuntimeError(f"Could not connect to the Double Motor (card {config.CARD_SERIAL}). Is it on?")
            motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
            motor.movement_stop()
            with self._lock:
                self.motor = motor
                self._last_cmd = (0, 0)
            self._imu_obj = None
        if not (only_missing and self.sensor is not None and self.sensor.connected):
            print("[robot] connecting Color Sensor (purple card)...")
            sensor = le.ColorSensor()
            sensor.connect(card_color=config.CARD_COLOR, card_serial=config.CARD_SERIAL,
                           device_notification_delay=config.NOTIFICATION_MS)
            if not sensor.connected:
                raise RuntimeError(f"Could not connect to the Color Sensor (card {config.CARD_SERIAL}). Is it on?")
            self.sensor = sensor
            self._color_obj = None
        self.status = "connected"
        print("[robot] connected.")

    def close(self):
        self._closing.set()
        if self.motor is not None:
            try:
                self.motor.movement_stop(blocking=False)
            except Exception:
                pass
        for dev, name in ((self.motor, "motor"), (self.sensor, "color sensor")):
            if dev is not None and not _run_with_timeout(dev.disconnect, timeout=config.BLE_DISCONNECT_TIMEOUT_S):
                print(f"[robot] {name} BLE disconnect timed out.")
        self.motor = self.sensor = None
        self.status = "closed"

    # ---------- driving ----------

    def _should_send(self, left, right, now, force):
        """Rate-limit commands so the hub's Bluetooth isn't flooded (see MOTOR_MIN_INTERVAL_S)."""
        last = self._last_cmd
        if force or last is None:
            return True
        if (left, right) == last:
            return False                                   # nothing new to say
        urgent = ((left, right) == (0, 0)                  # stopping
                  or left * last[0] < 0 or right * last[1] < 0)   # reversing a side
        if urgent:
            return True
        if abs(left - last[0]) < config.MOTOR_MIN_CHANGE and abs(right - last[1]) < config.MOTOR_MIN_CHANGE:
            return False
        return now - self._last_send >= config.MOTOR_MIN_INTERVAL_S

    def _send(self, left, right):
        if self.motor is None or not self.motor.connected:
            return
        try:
            if left == 0 and right == 0:
                self.motor.movement_stop(blocking=False)
            else:
                self.motor.movement_move_tank(left, right, blocking=False)
        except Exception as exc:
            self.status = f"motor error: {exc}"

    def tank(self, left, right, force=False):
        """Drive with left/right speeds in -100..100. Thread-safe; rate-limited to spare Bluetooth.

        Call it every tick with what you want: a command skipped by the rate limit goes out on a
        later tick.
        """
        left = int(max(-100, min(100, round(left))))
        right = int(max(-100, min(100, round(right))))
        now = time.monotonic()
        with self._lock:
            self.cmd = (left, right)
            if not self._should_send(left, right, now, force):
                return
            self._last_cmd, self._last_send = (left, right), now
            self._send(*robot_to_motor(self.drive, left, right))

    def stop(self):
        self.tank(0, 0, force=True)

    def raw_motor(self, left, right):
        """Drive the motor directly, ignoring `reverse` (drive calibration only). Rate-limited too."""
        now = time.monotonic()
        with self._lock:
            self.cmd = (left, right)
            if not self._should_send(left, right, now, False):
                return
            self._last_cmd, self._last_send = (left, right), now
            self._send(left, right)

    def raw_encoders(self):
        """(motor-left, motor-right) wheel degrees as the hub reports them, or None."""
        if self.motor is None:
            return None
        pl = self.motor.motor[le.MOTOR_LEFT].position
        pr = self.motor.motor[le.MOTOR_RIGHT].position
        return (float(pl), float(pr)) if _ok(pl, pr) else None

    # ---------- sensing ----------

    def read_color(self):
        """Latest color reading, classified. Cheap enough for the 200 Hz reflex thread."""
        if self.sensor is None:
            return self._color
        s = self.sensor.sensor
        if s is self._color_obj:
            return self._color
        self._color_obj = s
        reading = ColorReading(
            t=time.monotonic(), hue=float(s.hue), saturation=float(s.saturation), value=float(s.value),
            reflection=float(s.reflection), rgb=(float(s.rawRed), float(s.rawGreen), float(s.rawBlue)),
            lego_color=int(s.color) if _ok(s.color) else -1)
        reading.label = self.classifier.classify(reading)
        self._color = reading
        return reading

    def snapshot(self):
        now = time.monotonic()
        snap = RobotSnapshot(t=now, color=self.read_color(), cmd_left=self.cmd[0], cmd_right=self.cmd[1])
        if self.motor is None:
            return snap
        d = self.drive
        try:
            raw = self.raw_encoders()
            if raw is not None:
                pl, pr = motor_to_robot_wheels(d, *raw)                   # our left/right wheels
                snap.enc_left = d["enc_left_sign"] * pl
                snap.enc_right = d["enc_right_sign"] * pr
        except (AttributeError, IndexError, TypeError):
            pass
        imu = self.motor.imu_device
        if _ok(imu.gyroscopeX, imu.accelerometerX):
            if imu is not self._imu_obj:
                self._imu_obj, self._imu_t = imu, now
            g = (float(imu.gyroscopeX), float(imu.gyroscopeY), float(imu.gyroscopeZ))
            snap.gyro_raw = g
            snap.gyro_yaw_dps = d["gyro_sign"] * d["gyro_scale"] * g[int(d["gyro_axis"])]
            snap.accel_mg = (float(imu.accelerometerX), float(imu.accelerometerY), float(imu.accelerometerZ))
            snap.imu_t = self._imu_t
        return snap
