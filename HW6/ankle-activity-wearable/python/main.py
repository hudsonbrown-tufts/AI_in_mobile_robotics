# Linux side of the HW6 ankle activity wearable.
#
# Shared by both parts:
#   - pulls 50 Hz IMU rows from the sketch over the Bridge
#   - keeps the last 2 s (classifiers.WINDOW rows) and makes a decision every
#     0.24 s with BOTH parts, each with its own 3-in-a-row smoothing
#   - the active part (mode 1 or 2, set from the GUI over MQTT) decides what the
#     LED matrix scrolls
#   - publishes the decisions (ME193/hudson/hw6/state) and the raw rows
#     (ME193/hudson/hw6/raw, used by the GUI for its plot and for recording)
#
# The decision code itself is in classifiers.py (PART 1 and PART 2 sections).

import json
import time
import uuid

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

import classifiers as C

# Our own Mosquitto broker runs on this board in a Docker container
# ("mqtt-broker", port 1883; README section 7). This app runs in its own
# container, where 172.17.0.1 is Docker's fixed address for the board itself,
# so this works on any WiFi network. The laptop reaches the same broker at the
# board's WiFi address.
BROKER = "172.17.0.1"
PORT = 1883
TOPIC_STATE = "ME193/hudson/hw6/state"
TOPIC_RAW = "ME193/hudson/hw6/raw"
TOPIC_CMD = "ME193/hudson/hw6/cmd"      # "mode:1" / "mode:2" (retained, so a restart keeps the mode)

ACCEL_COUNTS_PER_G = 2048.0             # sketch: +-16 g
GYRO_COUNTS_PER_DPS = 16.4              # sketch: +-2000 deg/s
ROWS_PER_REPLY = 6                      # sketch: MAX_ROWS_PER_REPLY
RAW_PUBLISH_S = 0.2
DISPLAY_RESEND_S = 2.0                  # re-send the word in case the sketch restarted
PRINT_S = 2.0
MAX_GAP_ROWS = 10                       # bigger gaps in the data restart the window

# Physical units for each of the 12 values in a row (see classifiers.COLUMNS).
_ROW_SCALE = np.array([1 / ACCEL_COUNTS_PER_G] * 3 + [1 / GYRO_COUNTS_PER_DPS] * 3
                      + [1 / ACCEL_COUNTS_PER_G] * 6)
_B64_LOOKUP = np.zeros(256, dtype=np.int32)
for _i, _c in enumerate(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"):
    _B64_LOOKUP[_c] = _i


def decode_reply(reply):
    """Sketch reply -> (first sequence number, rows (n, 12) in g / deg/s), or None if "E"."""
    if reply.startswith("E"):
        return None
    d = _B64_LOOKUP[np.frombuffer(reply.encode("ascii"), dtype=np.uint8)].reshape(-1, 3)
    u16 = (d[:, 0] << 12) | (d[:, 1] << 6) | d[:, 2]
    seq = int(u16[0])
    values = (u16[1:].astype(np.int64) - 32768).reshape(-1, len(C.COLUMNS))
    return seq, values * _ROW_SCALE


class Wearable:
    """Windowing, both classifiers, smoothing and the matrix word."""

    def __init__(self):
        self.mode = 1
        # ---- PART 1: peak acceleration in one direction (part1_calibration.json)
        self.part1 = self._load(C.Part1PeakClassifier, C.PART1_CALIBRATION_FILE, "Part 1 calibration")
        self.smooth1 = C.Smoother()
        self.peak = None
        # ---- PART 2: decision rules on all IMU outputs (part2_rules.json)
        self.part2 = self._load(C.Part2Rules, C.PART2_RULES_FILE, "Part 2 rules")
        self.smooth2 = C.Smoother()
        self.why2 = None                 # Part 2's features and the questions it answered
        self.raw2 = None

        self.window = np.zeros((0, len(C.COLUMNS)))
        self.new_rows = 0
        self.expected_seq = None
        self.dropped = 0
        self.rows_total = 0
        self.decisions = 0

    @staticmethod
    def _load(cls, path, what):
        try:
            obj = cls.load(path)
            print(f"Loaded {what} from {path.name}")
            return obj
        except FileNotFoundError:
            print(f"No {what} ({path.name} missing): run the one-time calibration/training and redeploy")
        except Exception as exc:
            print(f"Couldn't load {what} from {path.name}: {exc}")
        return None

    def add_rows(self, seq, rows):
        """Add rows from one reply. Returns True if a new decision was made."""
        if self.expected_seq is not None:
            gap = (seq - self.expected_seq) % 65536
            if gap:
                self.dropped += gap
                if gap > MAX_GAP_ROWS:   # too much missing: don't classify across the hole
                    self.window = self.window[:0]
                    self.new_rows = 0
        self.expected_seq = (seq + len(rows)) % 65536
        self.rows_total += len(rows)
        self.window = np.vstack([self.window, rows])[-C.WINDOW:]
        self.new_rows += len(rows)
        if len(self.window) == C.WINDOW and self.new_rows >= C.HOP:
            self.new_rows = 0
            self.decide()
            return True
        return False

    def decide(self):
        self.decisions += 1
        # ---- PART 1: only the chosen direction's peak goes in
        if self.part1:
            raw1, self.peak = self.part1.classify(self.window)
            self.smooth1.update(raw1)
        # ---- PART 2: features of all 6 channels go through the questions
        if self.part2:
            self.raw2, self.why2 = self.part2.classify(self.window)
            self.smooth2.update(self.raw2)

    def active_state(self):
        return (self.smooth1 if self.mode == 1 else self.smooth2).state

    def display_word(self):
        if self.mode == 1 and not self.part1:
            return "NO CAL"
        if self.mode == 2 and not self.part2:
            return "NO CAL"
        state = self.active_state()
        return C.STATE_WORDS[state] if state else "HW6"

    def state_message(self, imu_ok):
        msg = {
            "t": round(time.time(), 2),
            "imu_ok": imu_ok,
            "mode": self.mode,
            "state": self.active_state(),
            "rows_total": self.rows_total,
            "dropped": self.dropped,
            "part1": None,
            "part2": None,
        }
        if self.part1:
            msg["part1"] = {
                "state": self.smooth1.state,
                "raw": self.part1.classify_peak(self.peak) if self.peak is not None else None,
                "peak": None if self.peak is None else round(self.peak, 3),
                "direction": self.part1.direction,
                "order": self.part1.order,
                "thresholds": self.part1.thresholds,
                "created": self.part1.created,
            }
        if self.part2:
            msg["part2"] = {
                "state": self.smooth2.state,
                "raw": self.raw2,
                "features": None if self.why2 is None else
                {k: round(v, 3) for k, v in self.why2["features"].items()},
                "path": None if self.why2 is None else self.why2["path"],
                "created": self.part2.created,
            }
        return msg


wearable = Wearable()
_raw_pending = []          # (seq, rows) waiting to go out on TOPIC_RAW
_last_raw_pub = 0.0
_last_display = ("", 0, 0.0)
_last_print = 0.0
_last_imu_err_pub = 0.0


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to {BROKER} ({reason_code})")
    client.subscribe(TOPIC_CMD)


def on_message(client, userdata, msg):
    text = msg.payload.decode(errors="replace").strip()
    if text in ("mode:1", "mode:2"):
        new_mode = int(text[-1])
        if new_mode != wearable.mode:
            wearable.mode = new_mode
            print(f"Mode -> Part {new_mode}")
    else:
        print(f"Ignoring command {text!r}")


def update_display(force=False):
    global _last_display
    word, mode = wearable.display_word(), wearable.mode
    old_word, old_mode, sent_at = _last_display
    now = time.monotonic()
    if force or word != old_word or mode != old_mode or now - sent_at > DISPLAY_RESEND_S:
        try:
            Bridge.call("set_display", word, mode)
            _last_display = (word, mode, now)
        except Exception as exc:
            print(f"Bridge call set_display failed: {exc}")


def publish_raw():
    global _raw_pending, _last_raw_pub
    now = time.monotonic()
    if not _raw_pending or now - _last_raw_pub < RAW_PUBLISH_S:
        return
    _last_raw_pub = now
    rows = np.vstack([r for _, r in _raw_pending])
    out = [[round(float(v), 3 if i < 3 or i >= 6 else 1) for i, v in enumerate(row)] for row in rows]
    client.publish(TOPIC_RAW, json.dumps({"seq": _raw_pending[0][0], "rows": out}, separators=(",", ":")))
    _raw_pending = []


def loop():
    global _last_print, _last_imu_err_pub
    imu_ok = True
    for _ in range(5):                       # drain up to 30 rows per pass
        try:
            reply = Bridge.call("get_samples")
        except Exception as exc:
            print(f"Bridge call get_samples failed: {exc}")
            time.sleep(0.5)
            return
        decoded = decode_reply(str(reply))
        if decoded is None:
            imu_ok = False
            break
        seq, rows = decoded
        if len(rows) == 0:
            break
        _raw_pending.append((seq, rows))
        if wearable.add_rows(seq, rows):
            client.publish(TOPIC_STATE, json.dumps(wearable.state_message(True), separators=(",", ":")))
        if len(rows) < ROWS_PER_REPLY:
            break

    now = time.monotonic()
    if not imu_ok and now - _last_imu_err_pub > 1.0:
        _last_imu_err_pub = now
        client.publish(TOPIC_STATE, json.dumps(wearable.state_message(False), separators=(",", ":")))
    update_display()
    publish_raw()

    if now - _last_print >= PRINT_S:
        _last_print = now
        if not imu_ok:
            print("MPU-6050 not answering: check wiring (VCC 3.3V, GND, SDA, SCL)")
        else:
            p1 = wearable.smooth1.state
            p2 = wearable.smooth2.state
            peak = "-" if wearable.peak is None else f"{wearable.peak:.2f} g"
            print(f"mode {wearable.mode} | part1 {p1} (peak {peak}) | part2 {p2} | "
                  f"rows {wearable.rows_total} dropped {wearable.dropped}")
    time.sleep(0.04)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hw6-wearable-{uuid.uuid4().hex[:6]}")
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT)
client.loop_start()  # network traffic + reconnects on a background thread

# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    update_display(force=True)
    App.run(user_loop=loop)
finally:
    client.loop_stop()
    client.disconnect()
