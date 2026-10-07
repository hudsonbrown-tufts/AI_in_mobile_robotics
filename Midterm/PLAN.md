# Midterm: Virtual Ping Pong (Wii Sports–style)

Status: **planned, not started.** See the [Progress log](#progress-log) at the bottom.

## Goal

A virtual ping pong game in the spirit of Wii Sports Table Tennis:

- **IMU** on the LEGO Double Motor detects *when* I swing (and how hard).
- **Camera pose detection** knows *where* my paddle (right wrist) is.
- **AprilTag** starts the game when first seen and tracks my player level, which sets ball speed.
- **Motor haptics**: the Double Motor buzzes in my hand when I hit the ball.
- **GUI** shows my camera view and a virtual table with the ball.

## Decisions (confirmed)

| Question | Answer |
|---|---|
| Handedness | Right-handed: MediaPipe Pose **right wrist (landmark 16)** is the paddle |
| Motor mounting | Hold the Double Motor **by itself** in the right hand, like a Wii Remote |
| AprilTag | **One tag** (family 36h11, any ID); level is stored per tag ID in `players.json` |
| Camera | **Phone camera via Phone Link** (same as HW4: `CAMERA_NAME_HINT = "Windows Virtual Camera"`) |

## Code reused from earlier assignments

- **Double Motor connection**: `connect()` in [Class 6/q_learning_straight.py](../Class%206/q_learning_straight.py) uses `le.DoubleMotor()` with card `LEGO_COLOR_PURPLE`, serial `"5164"`.
- **IMU fields**: `motor.imu_device` has `yaw/pitch/roll` (tenths of a degree), `accelerometerX/Y/Z` and `gyroscopeX/Y/Z`. Values are NaN until the first notification arrives.
- **AprilTag**: the OpenCV aruco `DICT_APRILTAG_36h11` detector in [HW2/apriltag_detector.py](../HW2/apriltag_detector.py) needs no extra install.
- **Camera**: `open_camera()` and `list_cameras()` in [HW4/minifig_tracker.py](../HW4/minifig_tracker.py) use `cv2_enumerate_cameras` with MSMF.
- **MediaPipe tasks API**: same style as [HW1/steering_control.py](../HW1/steering_control.py). It needs `pose_landmarker_lite.task`, a one-time download into `Midterm/models/`.
- **GUI**: pygame (already installed in `.venv`).

## File layout (`Midterm/`)

| File | What it does |
|---|---|
| `pingpong.py` | Entry point: pygame main loop and the game state machine |
| `paddle_imu.py` | Double Motor connection, IMU polling thread, swing detection and the haptics queue |
| `vision.py` | Camera thread: MediaPipe Pose and AprilTag. Hands back the latest frame, wrist position and tag ID |
| `game.py` | Pure game logic (no hardware): ball physics, CPU opponent, scoring and hit test |
| `render.py` | Draws the camera panel and the table in 3D perspective, plus the HUD |
| `players.json` | Saved level and stats per AprilTag ID (created automatically) |
| `models/pose_landmarker_lite.task` | MediaPipe pose model |
| `README.md` | Setup, controls and tuning notes |

## 1. Swing detection (IMU) in `paddle_imu.py`

- A background thread polls `gyroscopeX/Y/Z` about every 10 ms.
- Angular speed is `ω = √(gx² + gy² + gz²)`, lightly smoothed.
- **Swing event**: ω rises above `SWING_THRESHOLD`. That records `(timestamp, peak ω → strength, direction)`, followed by a 0.4 s cooldown so one swing never counts twice.
- **Forehand vs. backhand**: the sign of the main gyro axis. The `c` key runs a calibration (a few swings) that learns which axis and sign that is.
- A `check` mode prints live ω so I can tune `SWING_THRESHOLD`.
- The game reads the result through `paddle.last_swing`.

## 2. Paddle position (pose) in `vision.py`

- MediaPipe Pose in VIDEO mode on its own thread, so the game stays at about 60 FPS while pose runs at about 20 FPS.
- The paddle is the **right wrist (landmark 16)**, normalized and EMA-smoothed.
- Wrist (x, y) in the mirrored camera image maps to paddle (x, y) over the player's end of the table.
- The skeleton and a paddle marker are drawn over the camera view.

## 3. AprilTag: start trigger and player level

- **`WAITING_FOR_TAG`** shows "Hold up your player card". The first tag seen sets the player ID, which loads that record from `players.json` (or creates level 1), then moves to `COUNTDOWN` (3, 2, 1) and `PLAYING`.
- **Ball speed from level**: `ball_speed = BASE_SPEED × (1 + 0.15 × (level − 1))`. The CPU also aims better and places the ball wider at higher levels.
- **Levelling**: a win moves up one level and a loss moves down one (minimum 1). The result is saved to `players.json`.

## 4. Game logic in `game.py`

- **Table coordinates**: x runs across the table (−1 to 1); z runs along it (0 is my end, 1 is the CPU's end); y is height.
- **Ball**: position and velocity, gravity, table bounce, flight between the ends.
- **Hit window**: the ball is near my end (z < about 0.1). A hit needs **both** of these:
  1. **Position**: the pose paddle is within `HIT_RADIUS` of the ball's x/y.
  2. **Timing**: a swing happened within ±150 ms of the ball arriving.
- **Return shot**: swing strength sets return speed; the paddle's x offset from the ball and the swing direction set the angle.
- **Hit quality**: PERFECT, GOOD or EARLY/LATE, based on timing and centering.
- **Misses**: a swing with the paddle in the wrong place is a whiff, and no swing means the ball goes past. Either way the CPU scores.
- **CPU opponent**: tracks the ball with a maximum speed and an error that both depend on level, so it misses sometimes.
- **Scoring**: first to 11, win by 2, serve switches every 2 points. `GAME_OVER` updates the level, then goes back to `WAITING_FOR_TAG`.

## 5. Haptic feedback in `paddle_imu.py`

- A queue and its own thread send the BLE motor commands, which block, so they never stall the game loop.
- Patterns use short `motor_run_for_time` pulses at alternating speeds:
  - **Hit**: a sharp jolt, about 80 ms at speed 100, scaled by quality.
  - **Perfect hit**: a double pulse.
  - **Missed point**: a long, low rumble.
  - **Game start / win**: a "ba-dum" pattern.
- The motor end state is set to **coast** so the motor doesn't lock up in my hand.

## 6. GUI in `render.py` (pygame, 1280×720)

- **Left, about 40%**: the mirrored camera feed with the pose skeleton, the paddle marker on my right wrist, the AprilTag outline and a live ω swing meter.
- **Right, about 60%**: a Wii-style table seen from behind my end. It has a trapezoid table, a net, a ball drawn in perspective with a shadow, my paddle (from pose) and the CPU paddle.
- **HUD**: score, player ID and level, ball speed, pop-up hit-quality text.
- **Screens**: "Hold up your card", countdown, game over.
- **Keys**:
  - `q`: quit
  - `c`: calibrate swing
  - `r`: reset game
  - `k`: keyboard/mouse test mode (space = swing, mouse = paddle), for testing without the motor or camera

## 7. Threads

| Thread | Job |
|---|---|
| Main | pygame events, fixed-time-step game update, rendering |
| Vision | grab frame → pose and AprilTag → store the latest result (protected by a lock) |
| IMU | poll gyro → swing events |
| Haptics | queued pulse patterns → motor commands |

## 8. Build and test order

1. [ ] `game.py` and `render.py` with keyboard/mouse only: physics, hit test, look.
2. [ ] `vision.py`: pose paddle tracking and the AprilTag-triggered start.
3. [ ] `paddle_imu.py`: `check` mode to tune `SWING_THRESHOLD`, then haptics.
4. [ ] Tune thresholds and write `README.md`.

## Progress log

Updated as the code is written: what's done, what changed from the plan, and tuned values.

- **2026-10-07**: Plan written and decisions confirmed. No code yet.
