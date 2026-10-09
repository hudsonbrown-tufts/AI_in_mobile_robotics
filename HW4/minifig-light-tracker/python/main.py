# Linux side of the UNO Q minifig tracker.
#
# Subscribes to the HW3 MQTT topic, picks out "minifig:<col>,<row>" /
# "minifig:none" messages sent by HW4/minifig_tracker.py on the computer,
# and forwards them over the Bridge to the sketch, which drives the matrix.
# Any other message on the topic (e.g. HW3's game messages) is ignored.
#
# The MQTT callback runs on paho's network thread, so it only records the
# latest position; loop() (run by App.run on the app's own thread) makes the
# Bridge calls. This also drops intermediate positions if messages arrive
# faster than the Bridge can forward them.

import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "test.mosquitto.org"
PORT = 1883
MQTT_TOPIC = "ME193/hudson"
MESSAGE_PREFIX = "minifig:"

MATRIX_COLS = 13
MATRIX_ROWS = 8

# Latest position from MQTT: (col, row), "none", or None if nothing received
# yet. `pending` is set when it changes and cleared once forwarded.
_lock = threading.Lock()
_latest = None
_pending = False


def parse(payload):
    """Return (col, row), "none", or None if the payload isn't ours / is malformed."""
    if not payload.startswith(MESSAGE_PREFIX):
        return None
    body = payload[len(MESSAGE_PREFIX):].strip()
    if body == "none":
        return "none"
    try:
        col, row = (int(v) for v in body.split(","))
    except ValueError:
        return None
    if 0 <= col < MATRIX_COLS and 0 <= row < MATRIX_ROWS:
        return col, row
    return None


def on_connect(client, userdata, flags, reason_code, properties):
    # Subscribing here means we re-subscribe automatically after a reconnect.
    print(f"Connected to {BROKER} ({reason_code}); subscribing to '{MQTT_TOPIC}'")
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    global _latest, _pending
    payload = msg.payload.decode(errors="replace")
    parsed = parse(payload)
    if parsed is None:
        return
    with _lock:
        _latest = parsed
        _pending = True


def loop():
    global _pending
    with _lock:
        position, send = _latest, _pending
        _pending = False

    if send:
        try:
            if position == "none":
                Bridge.call("clear_matrix")
                print("<- none (matrix cleared)")
            else:
                # Resends of the same position still go through, which
                # refreshes the sketch's stale-pixel timeout.
                Bridge.call("set_pixel", position[0], position[1])
                print(f"<- pixel {position[0]},{position[1]}")
        except Exception as exc:
            print(f"Bridge call for {position} failed: {exc}")

    time.sleep(0.02)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT)
client.loop_start()  # network traffic + reconnects on a background thread

# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    App.run(user_loop=loop)
finally:
    client.loop_stop()
    client.disconnect()
    try:
        Bridge.call("clear_matrix")
    except Exception:
        pass
