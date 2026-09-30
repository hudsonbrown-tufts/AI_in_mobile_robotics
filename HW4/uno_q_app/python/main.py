# Linux side of the UNO Q minifig tracker.
#
# Subscribes to the HW3 MQTT topic, picks out "minifig:<col>,<row>" /
# "minifig:none" messages sent by HW4/minifig_tracker.py on the computer,
# and forwards them over the Bridge to the sketch, which drives the matrix.
# Any other message on the topic (e.g. HW3's game messages) is ignored.

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "test.mosquitto.org"
PORT = 1883
MQTT_TOPIC = "ME193/hudson"
MESSAGE_PREFIX = "minifig:"

MATRIX_COLS = 13
MATRIX_ROWS = 8


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
    payload = msg.payload.decode(errors="replace")
    parsed = parse(payload)
    if parsed is None:
        return
    try:
        if parsed == "none":
            Bridge.call("clear_matrix")
        else:
            Bridge.call("set_pixel", parsed[0], parsed[1])
        print(f"<- {payload}")
    except Exception as exc:
        print(f"Bridge call for '{payload}' failed: {exc}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT)
client.loop_start()  # network traffic + reconnects on a background thread

App.run()
