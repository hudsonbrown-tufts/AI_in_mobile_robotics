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
  - Pins: `M1A=3, M1B=5, M2A=6, M2B=9`. These are PWM pins on the classic UNO layout; **not yet confirmed against the UNO Q pinout**.
  - `INVERT_M1=false, INVERT_M2=true`, since mirror-mounted motors spin opposite ways.
  - Watchdog: motors stop if there's no `set_speed` for `COMMAND_TIMEOUT_MS = 500`.
  - The Bridge handler only stores state, and `loop()` does the `analogWrite`s (`map(|speed|, 0..100, 0..255)`).
  - Needs a common GND between the UNO Q and the Maker Drive; motor power comes from a separate battery.

## Testing done (2026-10-01, off-hardware)
- PID simulation (error rate ∝ speed, 100% ≈ 1.5 half-frames/s, 0.2 s command latency, 30 fps): starting at +0.8 and −0.6, it settled within 0.03 of center and stopped by 10 s.
- `horizontal_error` on test images gave sensible values (e.g. +0.71 for a minifig far right).
- Board `main.py` with stubbed `App`/`Bridge`/paho: `car:45` → 45, `car:-30` → −30, `car:250` → 100, `car:stop` → 0; `minifig:3,2`, `Goal` and `car:abc` were ignored; stop sends `set_speed(0)`.
- **Not yet tested on the real car.** Suggested first run: wheels off the ground, then check `DRIVE_DIRECTION`, `INVERT_M2`, MIN_SPEED and MAX_SPEED.
- PyTorch couldn't see the GPU during this session, so expect a lower fps on CPU. MQTT latency to the public broker plus a low fps limits how aggressive the gains can be. Raise KD or lower KP if it oscillates.

## Next steps / open questions
- Confirm the UNO Q PWM pins and the Maker Drive model (MDD3A vs. the original Maker Drive), and update the pin constants if needed.
- Tune the gains on the real car and record the final values here.
