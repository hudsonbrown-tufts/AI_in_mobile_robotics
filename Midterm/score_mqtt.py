"""Publishes my ping pong score to MQTT in real time.

Same broker as the earlier assignments (test.mosquitto.org, via mqttlib).
The score is my RECORD number of continuous hits (longest rally of my hits
in a row, all-time for this player card), sent as a floating point number,
e.g. "7.0". It goes out the moment it changes (including mid-rally, as a
rally beats the record) and is repeated every RESEND_S so a dashboard that
subscribes late still gets it.

Connecting happens on a background thread with retries, so a slow or
missing network never freezes the game.
"""

import threading
import time

from mqttlib import MQTTClient

MQTT_TOPIC = "ME193/Rogers/Hudson"
RESEND_S = 2.0
CONNECT_RETRIES = 3
RETRY_DELAY_S = 3


class ScorePublisher:
    def __init__(self, topic=MQTT_TOPIC):
        self.topic = topic
        self.status = "connecting..."
        self.last_sent = None
        self._client = None
        self._score = None
        self._sent_t = 0.0
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._connect, daemon=True).start()
        return self

    @property
    def connected(self):
        return self._client is not None

    def _connect(self):
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                client = MQTTClient()
                client.connect()
                self._client = client
                self.status = "connected"
                print(f"[mqtt] connected; publishing score to {self.topic}")
                return
            except OSError as exc:
                self.status = f"connect failed ({exc})"
                print(f"[mqtt] connect attempt {attempt}/{CONNECT_RETRIES} failed: {exc}")
                if attempt < CONNECT_RETRIES:
                    time.sleep(RETRY_DELAY_S)
        self.status = "offline (no broker)"

    def update(self, score, now):
        """Call every frame with the current record; sends on change or every RESEND_S."""
        if score is None or self._client is None:
            return
        score = float(score)
        if score != self._score or now - self._sent_t >= RESEND_S:
            message = f"{score:.1f}"
            try:
                self._client.publish(self.topic, message)
            except Exception as exc:
                self.status = f"publish failed ({exc})"
                return
            if score != self._score:
                print(f"[mqtt] {self.topic} <- {message}")
            self._score, self._sent_t, self.last_sent = score, now, message
            self.status = "connected"

    def close(self):
        if self._client is not None:
            try:
                time.sleep(0.3)   # let the last publish go out
                self._client.disconnect()
            except Exception:
                pass
