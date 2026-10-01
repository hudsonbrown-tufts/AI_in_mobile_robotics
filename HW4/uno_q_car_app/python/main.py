# Linux side of the UNO Q minifig car.
#
# Subscribes to the HW3/HW4 MQTT topic, picks out "car:<speed>" /
# "car:stop" messages sent by HW4/minifig_car.py on the computer, and
# forwards the speed over the Bridge to the sketch, which drives the motors.
# Any other message on the topic (tracker, HW3 game) is ignored.
#
# The MQTT callback runs on paho's network thread, so it only records the
# latest speed; loop() (run by App.run on the app's own thread) makes the
# Bridge calls.

import threading
import time

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "test.mosquitto.org"
PORT = 1883
MQTT_TOPIC = "ME193/hudson"
MESSAGE_PREFIX = "car:"
MAX_SPEED = 100

_lock = threading.Lock()
_latest = None   # latest speed in -100..100
_pending = False


def parse(payload):
    """Return a speed in -100..100, or None if the payload isn't ours / is malformed."""
    if not payload.startswith(MESSAGE_PREFIX):
        return None
    body = payload[len(MESSAGE_PREFIX):].strip()
    if body == "stop":
        return 0
    try:
        speed = int(body)
    except ValueError:
        return None
    return max(-MAX_SPEED, min(MAX_SPEED, speed))


def on_connect(client, userdata, flags, reason_code, properties):
    # Subscribing here means we re-subscribe automatically after a reconnect.
    print(f"Connected to {BROKER} ({reason_code}); subscribing to '{MQTT_TOPIC}'")
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    global _latest, _pending
    speed = parse(msg.payload.decode(errors="replace"))
    if speed is None:
        return
    with _lock:
        _latest = speed
        _pending = True


def loop():
    global _pending
    with _lock:
        speed, send = _latest, _pending
        _pending = False

    if send:
        try:
            # Every message is forwarded, even repeats, because each one
            # refreshes the sketch's command-timeout safety stop.
            Bridge.call("set_speed", speed)
            print(f"<- speed {speed:+d}%")
        except Exception as exc:
            print(f"Bridge call for speed {speed} failed: {exc}")

    time.sleep(0.01)


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
        Bridge.call("set_speed", 0)
    except Exception:
        pass
