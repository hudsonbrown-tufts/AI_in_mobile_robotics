"""
Tracks the LEGO minifigure with the trained YOLO model and tells an Arduino
UNO Q, over MQTT, which pixel of its 13x8 LED matrix to light so the lit
pixel mirrors where the minifig is in the camera frame.

Messages go to the same broker/topic as HW3 (test.mosquitto.org,
"ME193/hudson"):
    minifig:<col>,<row>   col 0-12 (left->right), row 0-7 (top->bottom)
    minifig:none          no minifig detected -> matrix clears
The "minifig:" prefix lets the UNO Q ignore HW3's game messages on the topic.

Usage:
    python minifig_tracker.py            # track and publish
    python minifig_tracker.py --dry-run  # track and print messages only
Press 'q' in the preview window to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from cv2_enumerate_cameras import enumerate_cameras
from ultralytics import YOLO

from mqttlib import MQTTClient

HERE = Path(__file__).parent
WEIGHTS = HERE / "runs" / "detect" / "minifig" / "weights" / "best.pt"

MQTT_TOPIC = "ME193/hudson"
MESSAGE_PREFIX = "minifig:"
MQTT_CONNECT_RETRIES = 3
MQTT_RETRY_DELAY_SECONDS = 3

# UNO Q built-in LED matrix size.
MATRIX_COLS = 13
MATRIX_ROWS = 8

# Camera selection: the first camera whose name contains CAMERA_NAME_HINT is
# used (the Android phone shared through Windows Phone Link shows up as
# "<phone name> (Windows Virtual Camera)"). If no camera matches -- e.g. the
# phone isn't connected -- CAMERA_INDEX is used instead (0 = laptop webcam).
# Set CAMERA_NAME_HINT = None to always use CAMERA_INDEX.
# List cameras with: python minifig_tracker.py --list-cameras
CAMERA_NAME_HINT = "Windows Virtual Camera"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
MIRROR = False  # True flips left/right, if the matrix reads backwards from where you stand
MISSED_FRAMES_BEFORE_NONE = 10  # consecutive frames with no detection before sending "none"
RESEND_INTERVAL_SECONDS = 2.0  # repeat the current message so a newly started UNO Q catches up


def connect_mqtt(mqtt_client):
    """Connect with a few retries, since the public test.mosquitto.org broker
    occasionally fails a connection attempt with a transient DNS/network
    error rather than being genuinely unreachable."""
    for attempt in range(1, MQTT_CONNECT_RETRIES + 1):
        try:
            mqtt_client.connect()
            return
        except OSError as exc:
            if attempt == MQTT_CONNECT_RETRIES:
                print(f"Could not connect to the MQTT broker after {MQTT_CONNECT_RETRIES} attempts.")
                raise
            print(
                f"MQTT connect attempt {attempt}/{MQTT_CONNECT_RETRIES} failed ({exc}); "
                f"retrying in {MQTT_RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(MQTT_RETRY_DELAY_SECONDS)


def list_cameras():
    for cam in enumerate_cameras(cv2.CAP_MSMF):
        print(f"  [{cam.index}] {cam.name}")


def open_camera():
    """Open the camera matching CAMERA_NAME_HINT, falling back to CAMERA_INDEX."""
    if CAMERA_NAME_HINT:
        for cam in enumerate_cameras(cv2.CAP_MSMF):
            if CAMERA_NAME_HINT.lower() in cam.name.lower():
                cap = cv2.VideoCapture(cam.index, cam.backend)
                if cap.isOpened():
                    print(f"Using camera [{cam.index}] {cam.name}")
                    return cap
                print(f"Found '{cam.name}' but couldn't open it.")
        print(f"No camera matching '{CAMERA_NAME_HINT}' -- falling back to camera {CAMERA_INDEX}.")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}.")
    print(f"Using camera {CAMERA_INDEX}")
    return cap


def best_detection(result):
    """Return the (x1, y1, x2, y2) box of the most confident minifig, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    return boxes.xyxy[best].tolist()


def box_to_cell(box, frame_width, frame_height):
    """Map a box's center to a (col, row) cell of the LED matrix."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = min(int(cx / frame_width * MATRIX_COLS), MATRIX_COLS - 1)
    row = min(int(cy / frame_height * MATRIX_ROWS), MATRIX_ROWS - 1)
    if MIRROR:
        col = MATRIX_COLS - 1 - col
    return col, row


def draw_overlay(frame, box, cell, fps):
    h, w = frame.shape[:2]
    for c in range(1, MATRIX_COLS):
        x = int(c * w / MATRIX_COLS)
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    for r in range(1, MATRIX_ROWS):
        y = int(r * h / MATRIX_ROWS)
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

    if cell is not None:
        col, row = cell
        if MIRROR:
            col = MATRIX_COLS - 1 - col  # draw in image coordinates
        x1 = int(col * w / MATRIX_COLS)
        y1 = int(row * h / MATRIX_ROWS)
        x2 = int((col + 1) * w / MATRIX_COLS)
        y2 = int((row + 1) * h / MATRIX_ROWS)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    label = f"cell {cell[0]},{cell[1]}" if cell else "no minifig"
    cv2.putText(frame, f"{label}  {fps:.0f} fps", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print messages instead of publishing")
    parser.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    args = parser.parse_args()
    if args.list_cameras:
        list_cameras()
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    print(f"Loaded {WEIGHTS.name}, running on device={device!r}")

    cap = open_camera()

    mqtt_client = None
    if not args.dry_run:
        mqtt_client = MQTTClient()
        connect_mqtt(mqtt_client)
        print(f"Publishing to '{MQTT_TOPIC}' on test.mosquitto.org.")

    def send(message):
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC, message)
        print(f"-> {message}")

    last_message = None
    last_sent_at = 0.0
    missed_frames = 0
    cell = None
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break

            result = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)[0]
            box = best_detection(result)

            if box is not None:
                missed_frames = 0
                cell = box_to_cell(box, frame.shape[1], frame.shape[0])
            else:
                missed_frames += 1
                if missed_frames >= MISSED_FRAMES_BEFORE_NONE:
                    cell = None

            message = MESSAGE_PREFIX + (f"{cell[0]},{cell[1]}" if cell else "none")
            now = time.time()
            if message != last_message or now - last_sent_at >= RESEND_INTERVAL_SECONDS:
                send(message)
                last_message = message
                last_sent_at = now

            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_overlay(frame, box, cell, fps)
            cv2.imshow("Minifig tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        send(MESSAGE_PREFIX + "none")  # leave the matrix dark when we stop
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            time.sleep(0.5)  # let the final publish go out
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
