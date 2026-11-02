# Linux side of the HW7 ankle step counter.
#
#   - pulls 50 Hz IMU rows from the sketch over the Bridge
#   - keeps the last 2 s (stepnet.WINDOW rows); every 0.24 s the 5-network
#     ensemble predicts the stride rate and walking speed of that window, each
#     with a spread
#   - StepAccumulator adds them up into steps and distance, each with a
#     95% range and an uncertainty breakdown
#   - publishes everything on ME193/hudson/hw7/state (about 4 times a second)
#     and the raw rows on ME193/hudson/hw7/raw (GUI plot and recordings)
#   - scrolls the step count on the LED matrix
#   - "reset" on ME193/hudson/hw7/cmd sets the count back to 0
#
# The count is not saved: every app start begins at 0.
# The network and the adding-up code are in stepnet.py.

import json
import time
import uuid

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

import stepnet as S

# Our own Mosquitto broker runs on this board in a Docker container (set up in
# HW6, see HW6/README.md section 7). This app runs in its own container, where
# 172.17.0.1 is Docker's fixed address for the board itself, so this works on
# any WiFi network. The laptop reaches the same broker at the board's address.
BROKER = "172.17.0.1"
PORT = 1883
TOPIC_STATE = "ME193/hudson/hw7/state"
TOPIC_RAW = "ME193/hudson/hw7/raw"
TOPIC_CMD = "ME193/hudson/hw7/cmd"      # "reset"

ACCEL_COUNTS_PER_G = 2048.0             # sketch: +-16 g
GYRO_COUNTS_PER_DPS = 16.4              # sketch: +-2000 deg/s
ROWS_PER_REPLY = 6                      # sketch: MAX_ROWS_PER_REPLY
RAW_PUBLISH_S = 0.2
DISPLAY_EVERY_S = 2.0                   # how often the matrix gets the new count
PRINT_S = 2.0
MAX_GAP_ROWS = 10                       # bigger gaps in the data restart the window

# Physical units for each of the 12 values in a row (see stepnet.COLUMNS).
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
    values = (u16[1:].astype(np.int64) - 32768).reshape(-1, len(S.COLUMNS))
    return seq, values * _ROW_SCALE


def load_model():
    try:
        ens = S.StepNetEnsemble.load()
        print(f"Loaded {len(ens.members)}-network ensemble from {S.MODEL_FILE.name} (made {ens.created})")
        return ens
    except FileNotFoundError:
        print(f"No model ({S.MODEL_FILE.name} missing): train it in the GUI (Record tab -> Train) and deploy")
    except Exception as exc:
        print(f"Couldn't load {S.MODEL_FILE.name}: {exc}")
    return None


ensemble = load_model()
counter = S.StepAccumulator(ensemble) if ensemble else None
windower = S.Windower(MAX_GAP_ROWS)
_reset_requested = False
_raw_pending = []          # (seq, rows) waiting to go out on TOPIC_RAW
_last_raw_pub = 0.0
_last_display = ("", 0.0)
_last_print = 0.0
_last_imu_err_pub = 0.0
_decision_ms = 0.0


def state_message(imu_ok):
    msg = {
        "t": round(time.time(), 2),
        "imu_ok": imu_ok,
        "rows_total": windower.rows_total,
        "dropped": windower.dropped,
        "model": None,
        "decision_ms": round(_decision_ms, 1),
    }
    if counter:
        msg["model"] = {"created": ensemble.created, "members": len(ensemble.members)}
        msg.update(counter.summary())
    return msg


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to {BROKER} ({reason_code})")
    client.subscribe(TOPIC_CMD)


def on_message(client, userdata, msg):
    global _reset_requested
    text = msg.payload.decode(errors="replace").strip()
    if text == "reset":
        _reset_requested = True       # done in loop(), not on the MQTT thread
    else:
        print(f"Ignoring command {text!r}")


def display_text():
    if counter is None:
        return "NO MODEL"
    return str(counter.summary()["steps_shown"])


def update_display(force=False):
    global _last_display
    text = display_text()
    old_text, sent_at = _last_display
    now = time.monotonic()
    if force or (text != old_text and now - sent_at > DISPLAY_EVERY_S) or now - sent_at > 2 * DISPLAY_EVERY_S:
        try:
            Bridge.call("set_display", text, 0)
            _last_display = (text, now)
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
    global _last_print, _last_imu_err_pub, _reset_requested, _decision_ms
    if _reset_requested:
        _reset_requested = False
        if counter:
            counter.reset()
        print("Count reset to 0")
        update_display(force=True)

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
        for window, new_rows, first in windower.add_rows(seq, rows):
            if counter:
                t0 = time.perf_counter()
                counter.add(ensemble.predict(window), new_rows, first)
                _decision_ms = 1000 * (time.perf_counter() - t0)
            client.publish(TOPIC_STATE, json.dumps(state_message(True), separators=(",", ":")))
        if len(rows) < ROWS_PER_REPLY:
            break

    now = time.monotonic()
    if not imu_ok and now - _last_imu_err_pub > 1.0:
        _last_imu_err_pub = now
        client.publish(TOPIC_STATE, json.dumps(state_message(False), separators=(",", ":")))
    update_display()
    publish_raw()

    if now - _last_print >= PRINT_S:
        _last_print = now
        if not imu_ok:
            print("MPU-6050 not answering: check wiring (VCC 3.3V, GND, SDA, SCL)")
        elif counter:
            s = counter.summary()
            print(f"steps {s['steps_shown']} +-{Z(s['steps_sd'])} | {s['dist_m']:.1f} m +-{Z(s['dist_sd'])} | "
                  f"{s['cadence_spm']:.0f} steps/min | rows {windower.rows_total} dropped {windower.dropped} | "
                  f"{_decision_ms:.1f} ms/window")
        else:
            print(f"no model | rows {windower.rows_total} dropped {windower.dropped}")
    time.sleep(0.04)


def Z(sd):
    """95% half-width, for printing."""
    return f"{S.Z95 * sd:.1f}"


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hw7-stepcounter-{uuid.uuid4().hex[:6]}")
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
