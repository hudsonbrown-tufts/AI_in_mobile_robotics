# HW4 Progress Log — Minifig-centering PID car

A car carrying the minifig drives on a line parallel to the laptop. The webcam and the YOLO model find the minifig, and a PID controller on the computer drives the car until the minifig is horizontally centered in the frame. Speed commands go over MQTT to an Arduino UNO Q, which drives two motors through a Cytron Maker Drive.

It reuses the trained model, the MQTT topic and the UNO Q app pattern from the tracker. Read [PROGRESS_tracker.md](PROGRESS_tracker.md) first for the environment, model and App Lab gotchas (zip layout, `sketch.yaml` library list, **board clock must be correct**).

## Files
| File | Role |
|---|---|
| [minifig_car.py](minifig_car.py) | Computer side: webcam → YOLO → error → PID → `car:<speed>` over MQTT |
| [uno_q_car_app/](uno_q_car_app/) | UNO Q App Lab app: `app.yaml`, `python/main.py`, `python/requirements.txt` (paho-mqtt), `sketch/sketch.ino`, `sketch/sketch.yaml` (copied from the working tracker app) |
| `minifig_car_app.zip` | Zip of `uno_q_car_app/` for App Lab import (gitignored; rebuild with the same snippet as in PROGRESS_tracker.md, `src = Path("uno_q_car_app")`) |

## Protocol
- Same broker/topic as the tracker and HW3: `test.mosquitto.org:1883`, **`ME193/hudson`**.
- `car:<speed>` with an int from −100 to 100 (% power, sign = direction; one speed for both motors, so the car drives straight), or `car:stop`.
- The board ignores everything without the `car:` prefix (the tracker's `minifig:` and HW3's `Goal` / `I lost`), so both apps can run at once.

## Computer side (`minifig_car.py`)
- Imports `CAMERA_INDEX`, `CONFIDENCE_THRESHOLD`, `MQTT_TOPIC`, `WEIGHTS`, `best_detection`, `connect_mqtt` from `minifig_tracker.py` (keep those names stable).
- Error = (box center x − frame center) / (half width), from −1 to 1.
- `PID` class: integral clamped by `INTEGRAL_LIMIT` (anti-windup), derivative of error low-passed by `DERIVATIVE_SMOOTHING`, output clamped to ±`MAX_SPEED`. `reset()` runs after `MISSED_FRAMES_BEFORE_STOP` (5) frames with no detection, which also sends speed 0.
- Inside `DEADBAND` (0.04), speed = 0. Otherwise `apply_min_speed` raises small outputs to `MIN_SPEED` (20%) so the motors overcome static friction. `DRIVE_DIRECTION` (±1) flips the sign if the car drives away from center.
- Starting gains: **KP 60, KI 5, KD 8**; MAX_SPEED 70, MIN_SPEED 20.
- **Easy tuning:** constants at the top of the file, plus a "PID gains" OpenCV window with sliders (Kp = slider value; Ki and Kd = slider/10). Changes print, and the final gains print on exit so the user can copy them back into the constants.
- Sends at 10 Hz (`SEND_INTERVAL_SECONDS = 0.1`) even when the speed is unchanged, because each message refreshes the board's watchdog. Sends `car:stop` on exit. `--dry-run` prints instead of publishing.
- Overlay: center line, deadband lines, box, error, speed, gains, fps.

## UNO Q side
- `python/main.py`: same structure as the tracker app. MQTT `on_message` stores the latest parsed speed under a lock, and `loop()` (via `App.run(user_loop=loop)`) forwards **every** message with `Bridge.call("set_speed", speed)`, repeats included (watchdog). `try/finally` sends `set_speed(0)` on stop. Speeds are clamped to ±100 and junk is ignored.
- `sketch.ino`: Maker Drive, two inputs per motor: A = PWM and B = LOW → forward; A = LOW and B = PWM → reverse; both LOW → coast.
  - Pins: `M1A=3, M1B=5, M2A=6, M2B=9`. **Confirmed PWM on the UNO Q**, all at 500 Hz (ArduinoCore-zephyr `variants/arduino_uno_q_stm32u585xx` overlay: D3=TIM3_CH3, D5=TIM1_CH4, D6=TIM3_CH4, D9=TIM4_CH3; PWM also on D2, D7, D8, D10–D13, D20, D21; D0/D1 are not PWM).
  - `INVERT_M1=true, INVERT_M2=false` (2026-10-05: user asked to reverse both motors; was false/true). Mirror-mounted motors spin opposite ways, so exactly one is inverted. Inversion only affects direction, never speed.
  - Watchdog: motors stop if there's no `set_speed` for `COMMAND_TIMEOUT_MS = 500`.
  - The Bridge handler only stores state, and `loop()` calls `driveBoth(speed)`, which computes ONE duty (`map(|speed|, 0..100, 0..255)`) and writes it to both motors. The user requires that the motors never run at different speeds, so don't add per-motor trims or differential steering without asking.
  - Needs a common GND between the UNO Q and the Maker Drive; motor power comes from a separate battery.

## Testing done (2026-10-01, off-hardware)
- PID simulation (error rate ∝ speed, 100% ≈ 1.5 half-frames/s, 0.2 s command latency, 30 fps): starting at +0.8 and −0.6, it settled within 0.03 of center and stopped by 10 s.
- `horizontal_error` on test images gave sensible values (e.g. +0.71 for a minifig far right).
- Board `main.py` with stubbed `App`/`Bridge`/paho: `car:45` → 45, `car:-30` → −30, `car:250` → 100, `car:stop` → 0; `minifig:3,2`, `Goal` and `car:abc` were ignored; stop sends `set_speed(0)`.
- **Not yet tested on the real car.** Suggested first run: wheels off the ground, then check `DRIVE_DIRECTION`, `INVERT_M2`, MIN_SPEED and MAX_SPEED.
- PyTorch couldn't see the GPU during this session, so expect a lower fps on CPU. MQTT latency to the public broker plus a low fps limits how aggressive the gains can be. Raise KD or lower KP if it oscillates.

## 2026-10-05: detection trouble on the real car
The user reported the car program having trouble finding the minifig. Causes found:
- **Dataset mismatch.** *Correction:* I first claimed the training photos were close-ups (46% median width), but that came from misreading the labels. Roboflow exported **polygon** labels (`class x1 y1 x2 y2 ...`), not `cx cy w h`, so field 3 isn't the width. Ultralytics turns the polygons into boxes, so training was fine. The true median box width is **~11%** of the image (range 2.5–30%). The real gap is the **scene**: training = the minifig alone on a bare wooden table, mostly lying down, shot from above at a phone-in-hand angle. On the car it stands upright, seen from the side, surrounded by colorful jumper wires, a pink 3D-printed frame and yellow wheels. A frame from the phone camera with the minifig clearly visible (~90 px wide at 1280×720) got **no detection at all, even at conf 0.05**, at imgsz 640 or 960.
- **No GPU:** the RTX 3060 showed status "Unknown" in `Get-PnpDevice` (discrete GPU powered off) and the laptop was on battery, so inference ran on CPU.

Changes to `minifig_car.py` (it no longer imports `CONFIDENCE_THRESHOLD` / `best_detection` from the tracker):
- Its own `CONFIDENCE_THRESHOLD = 0.25` (was 0.5) plus a live **"Conf %"** slider (1–95). The sliders window is now "Tuning".
- `IMAGE_SIZE = 960` passed as `imgsz` (default was 640). `CAMERA_RESOLUTION = (1280, 720)` (the webcam supports it).
- Detections from `SHOW_CONFIDENCE_FLOOR` (0.05) up to the threshold are drawn in grey with their score. The accepted box shows its score.
- Pressing **`s`** saves the raw frame to `HW4/captures/` (gitignored) for labeling in Roboflow and retraining.
- A startup warning prints when running on CPU.
- Synthetic test (test image shrunk onto a 1280×720 grey canvas): at 50% scale, confidence was 0.54 at imgsz 640 vs **0.84** at 960. At 15% scale, rejected (0.13) at 640 vs **accepted 0.39** at 960. At 25% scale there was no detection at either size, so the model itself is fragile at small scales, and **retraining with captures from the real setup is the real fix**. CPU inference at 960 took about 50 ms per frame.

## 2026-10-05: Android phone as the webcam (Windows Phone Link)
- Phone Link exposes the phone (a Pixel 10 Pro Fold) as **"Pixel 10 Pro Fold (Windows Virtual Camera)"**. Laptop webcam = "USB2.0 HD UVC WebCam" (index 0), phone = index 1 (MSMF). DirectShow also lists an "OBS Virtual Camera".
- Added the `cv2-enumerate-cameras` package (in `.venv` and `requirements.txt`) to map camera names to OpenCV indices.
- `minifig_tracker.py` now has `CAMERA_NAME_HINT = "Windows Virtual Camera"` + `open_camera()` (first MSMF camera whose name contains the hint, else falls back to `CAMERA_INDEX`) + `list_cameras()`. Both programs use `open_camera()` and accept `--list-cameras`. `minifig_car.py` imports `open_camera` / `list_cameras` from the tracker.
- Verified: opens the phone at 1280×720, ~33 fps read rate. **The first frames are black** for a moment while Phone Link starts streaming, then real images arrive.

## 2026-10-05: retrained (v2 model)
- Retrained with 44 new photos of the minifig on the car (details in PROGRESS_tracker.md). The new model detects it in 13/13 held-out car photos at 0.89–0.99 confidence (old model: max 0.36), and at 0.97 on the phone-camera frame that previously got nothing.
- The user set `DRIVE_DIRECTION = -1` in `minifig_car.py` (on top of reversing both motors in the sketch).
- With the new model's high scores, `CONFIDENCE_THRESHOLD = 0.25` may let false positives through. The user can raise it with the slider if the car chases something else. `IMAGE_SIZE = 960` isn't needed for this model (640 scored the same on the phone frame) and 640 would be faster; both are left as is for now.

## Next steps / open questions
- Tune the PID on the real car with the v2 model.
- Confirm the Maker Drive model (MDD3A vs. the original Maker Drive).
- Tune the gains on the real car and record the final values here.
