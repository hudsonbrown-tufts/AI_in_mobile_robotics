# Linux side of the HW5 sumo shake controller.
#
# Polls the sketch's shake intensity over the Bridge and publishes it on the
# private MQTT topic as "shake:<0..1>" (or "shake:err" if the MPU-6050 isn't
# answering). HW5/sumo.py on the computer turns it into the robot's
# aggression: shaking = attack, still = defense. If these messages stop, the
# robot falls back to defense by itself after 2 s.

import time

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "test.mosquitto.org"
PORT = 1883
MQTT_TOPIC = "ME193/hudson"
MESSAGE_PREFIX = "shake:"
PUBLISH_PERIOD_S = 0.05     # 20 Hz: aggression follows the shaking quickly
PRINT_PERIOD_S = 1.0

_last_print = 0.0


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to {BROKER} ({reason_code}); publishing on '{MQTT_TOPIC}'")


def loop():
    global _last_print
    try:
        value = float(Bridge.call("get_shake"))
    except Exception as exc:
        print(f"Bridge call get_shake failed: {exc}")
        time.sleep(0.5)
        return

    payload = MESSAGE_PREFIX + ("err" if value < 0 else f"{value:.2f}")
    client.publish(MQTT_TOPIC, payload)

    now = time.monotonic()
    if now - _last_print >= PRINT_PERIOD_S:
        _last_print = now
        if value < 0:
            print("MPU-6050 not answering: check wiring (VCC 3.3V, GND, SDA, SCL)")
        else:
            print(f"-> {payload}  {'#' * int(value * 30)}")
    time.sleep(PUBLISH_PERIOD_S)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.connect_async(BROKER, PORT)
client.loop_start()  # network traffic + reconnects on a background thread

# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    App.run(user_loop=loop)
finally:
    client.publish(MQTT_TOPIC, MESSAGE_PREFIX + "0.00")
    client.loop_stop()
    client.disconnect()
