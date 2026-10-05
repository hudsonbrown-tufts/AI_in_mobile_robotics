"""
Drives the UNO Q car (two motors on a Maker Drive) along a line parallel to
the camera until the minifig riding on it is centered in the frame.

A PID controller turns the minifig's horizontal offset from the frame center
into a signed motor speed, sent over MQTT on the same broker/topic as
minifig_tracker.py (test.mosquitto.org, "ME193/hudson"):
    car:<speed>   speed -100..100 (% of full power; sign = direction)
    car:stop      stop both motors
The "car:" prefix lets the UNO Q ignore the tracker's and HW3's messages.

Tune the gains and the detection confidence with the constants below, or
live with the sliders in the "Tuning" window (changes are printed so you can
copy good values back).

Usage:
    python minifig_car.py            # track, run PID, drive the car
    python minifig_car.py --dry-run  # track and print commands only
Keys in the preview window:
    q   quit (the car is told to stop)
    s   save the current camera frame to captures/ (to label in Roboflow
        and retrain, so the model learns what the minifig looks like from here)
"""

import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from minifig_tracker import HERE, MQTT_TOPIC, WEIGHTS, connect_mqtt, list_cameras, open_camera
from mqttlib import MQTTClient

MESSAGE_PREFIX = "car:"

# --- Detection ---------------------------------------------------------------
# Minimum model confidence to accept a detection. Lower finds the minifig more
# often (especially small / far away) but risks locking onto look-alikes.
CONFIDENCE_THRESHOLD = 0.25
# Detections down to this confidence are drawn in grey with their score, so you
# can see what the model almost found and pick a good threshold.
SHOW_CONFIDENCE_FLOOR = 0.05
# Model input size. The training photos were close-ups; a minifig across the
# room is only a few pixels wide, and a bigger input size helps the model see
# it (try 960 or 1280). Larger = slower, especially without the GPU.
IMAGE_SIZE = 960
# Camera capture resolution (more pixels on a small minifig). Set to None to
# use the camera's default.
CAMERA_RESOLUTION = (1280, 720)
CAPTURE_DIR = HERE / "captures"

# --- PID tuning --------------------------------------------------------------
# Error is the minifig's horizontal offset from the frame center, as a
# fraction of half the frame width: -1 (left edge) .. 0 (center) .. +1 (right).
# Output is motor speed in % (-100..100).
KP = 60.0   # % speed per unit of error: 0.5 (quarter-frame off) -> 30%
KI = 5.0    # removes a steady offset (e.g. friction stopping the car short)
KD = 8.0    # damps overshoot; raise if the car oscillates around the center

MAX_SPEED = 70        # clamp on |output|, in %
MIN_SPEED = 20        # smallest % that actually moves the car (overcomes static friction)
INTEGRAL_LIMIT = 0.5  # anti-windup: clamp on the integral term's accumulated error*seconds
DEADBAND = 0.04       # |error| below this counts as centered -> motors stop
DERIVATIVE_SMOOTHING = 0.5  # 0..1 low-pass on the D term (higher = smoother, laggier)

# +1 or -1. Flip if the car drives *away* from the center instead of toward it.
DRIVE_DIRECTION = 1

MISSED_FRAMES_BEFORE_STOP = 5  # consecutive frames without a minifig before stopping
SEND_INTERVAL_SECONDS = 0.1    # send at 10 Hz; the UNO Q stops if commands stop arriving

# Slider ranges: Kp slider is the gain itself; Ki/Kd sliders are gain x10;
# confidence slider is in %.
KP_SLIDER_MAX = 200
KI_SLIDER_MAX = 500
KD_SLIDER_MAX = 500
TUNING_WINDOW = "Tuning"


class PID:
    def __init__(self, kp, ki, kd, output_limit, integral_limit, derivative_smoothing):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.derivative_smoothing = derivative_smoothing
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.derivative = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        if self.prev_error is not None and dt > 0:
            raw = (error - self.prev_error) / dt
            a = self.derivative_smoothing
            self.derivative = a * self.derivative + (1 - a) * raw
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * self.derivative
        return max(-self.output_limit, min(self.output_limit, output))


def split_detections(result, threshold):
    """Return ((box, conf) of the most confident detection at or above
    `threshold`, or None) and a list of (box, conf) below it."""
    best, rejected = None, []
    boxes = result.boxes
    if boxes is None:
        return best, rejected
    for box, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        if conf < threshold:
            rejected.append((box, conf))
        elif best is None or conf > best[1]:
            best = (box, conf)
    return best, rejected


def horizontal_error(box, frame_width):
    """Minifig center's offset from the frame center, in -1..1."""
    cx = (box[0] + box[2]) / 2
    half = frame_width / 2
    return (cx - half) / half


def apply_min_speed(output):
    """Bump small nonzero outputs up to MIN_SPEED so the motors actually turn."""
    if output == 0:
        return 0
    sign = 1 if output > 0 else -1
    return sign * max(abs(output), MIN_SPEED)


def create_tuning_sliders():
    cv2.namedWindow(TUNING_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNING_WINDOW, 400, 170)
    cv2.createTrackbar("Kp", TUNING_WINDOW, int(round(KP)), KP_SLIDER_MAX, lambda v: None)
    cv2.createTrackbar("Ki x10", TUNING_WINDOW, int(round(KI * 10)), KI_SLIDER_MAX, lambda v: None)
    cv2.createTrackbar("Kd x10", TUNING_WINDOW, int(round(KD * 10)), KD_SLIDER_MAX, lambda v: None)
    cv2.createTrackbar("Conf %", TUNING_WINDOW, int(round(CONFIDENCE_THRESHOLD * 100)), 95, lambda v: None)
    cv2.setTrackbarMin("Conf %", TUNING_WINDOW, 1)


def read_tuning_sliders():
    """Return ((kp, ki, kd), confidence threshold)."""
    kp = cv2.getTrackbarPos("Kp", TUNING_WINDOW)
    ki = cv2.getTrackbarPos("Ki x10", TUNING_WINDOW) / 10
    kd = cv2.getTrackbarPos("Kd x10", TUNING_WINDOW) / 10
    conf = cv2.getTrackbarPos("Conf %", TUNING_WINDOW) / 100
    return (kp, ki, kd), conf


def draw_overlay(frame, best, rejected, error, speed, pid, conf_threshold, fps):
    h, w = frame.shape[:2]
    cx = w // 2
    band = int(DEADBAND * w / 2)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 1)
    cv2.line(frame, (cx - band, 0), (cx - band, h), (0, 120, 0), 1)
    cv2.line(frame, (cx + band, 0), (cx + band, h), (0, 120, 0), 1)

    for box, conf in rejected:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    if best is not None:
        box, conf = best
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        bx = (x1 + x2) // 2
        cv2.line(frame, (cx, h // 2), (bx, h // 2), (0, 200, 255), 2)

    status = f"err {error:+.2f}" if error is not None else "no minifig"
    lines = [
        f"{status}  speed {speed:+.0f}%  {fps:.0f} fps",
        f"Kp {pid.kp:g}  Ki {pid.ki:g}  Kd {pid.kd:g}  conf >= {conf_threshold:.2f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print commands instead of publishing")
    parser.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    args = parser.parse_args()
    if args.list_cameras:
        list_cameras()
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    print(f"Loaded {WEIGHTS.name}, running on device={device!r}, imgsz={IMAGE_SIZE}")
    if device == "cpu":
        print("WARNING: CUDA GPU not available -- running on CPU, expect low fps. "
              "Plug the laptop in / switch its GPU mode off integrated-only.")

    cap = open_camera()
    if CAMERA_RESOLUTION:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    mqtt_client = None
    if not args.dry_run:
        mqtt_client = MQTTClient()
        connect_mqtt(mqtt_client)
        print(f"Publishing to '{MQTT_TOPIC}' on test.mosquitto.org.")

    def send(message):
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC, message)
        print(f"-> {message}")

    pid = PID(KP, KI, KD, MAX_SPEED, INTEGRAL_LIMIT, DERIVATIVE_SMOOTHING)
    conf_threshold = CONFIDENCE_THRESHOLD
    create_tuning_sliders()

    missed_frames = 0
    speed = 0
    last_sent_at = 0.0
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break
            now = time.time()
            dt = now - prev_time
            prev_time = now

            gains, conf = read_tuning_sliders()
            if gains != (pid.kp, pid.ki, pid.kd):
                pid.kp, pid.ki, pid.kd = gains
                print(f"Gains: KP = {pid.kp:g}, KI = {pid.ki:g}, KD = {pid.kd:g}")
            if conf != conf_threshold:
                conf_threshold = conf
                print(f"CONFIDENCE_THRESHOLD = {conf_threshold:.2f}")

            raw_frame = frame.copy()  # unannotated, for 's' captures
            result = model.predict(frame, conf=min(SHOW_CONFIDENCE_FLOOR, conf_threshold),
                                   imgsz=IMAGE_SIZE, device=device, verbose=False)[0]
            best, rejected = split_detections(result, conf_threshold)

            error = None
            if best is not None:
                missed_frames = 0
                error = horizontal_error(best[0], frame.shape[1])
                output = pid.update(error, dt)
                if abs(error) < DEADBAND:
                    speed = 0
                else:
                    speed = apply_min_speed(DRIVE_DIRECTION * output)
            else:
                missed_frames += 1
                if missed_frames >= MISSED_FRAMES_BEFORE_STOP:
                    speed = 0
                    pid.reset()  # don't carry stale integral/derivative into the next sighting

            if now - last_sent_at >= SEND_INTERVAL_SECONDS:
                send(f"{MESSAGE_PREFIX}{int(round(speed))}")
                last_sent_at = now

            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)
            draw_overlay(frame, best, rejected, error, speed, pid, conf_threshold, fps)
            cv2.imshow("Minifig car (q quit, s save frame)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(exist_ok=True)
                path = CAPTURE_DIR / f"car_{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}.jpg"
                cv2.imwrite(str(path), raw_frame)
                print(f"Saved {path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        send(MESSAGE_PREFIX + "stop")
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            time.sleep(0.5)  # let the stop command go out
            mqtt_client.disconnect()
        print(f"Final gains: KP = {pid.kp:g}, KI = {pid.ki:g}, KD = {pid.kd:g}, "
              f"CONFIDENCE_THRESHOLD = {conf_threshold:.2f}")


if __name__ == "__main__":
    main()
