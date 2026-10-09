# Midterm: Virtual Ping Pong (Wii Sports–style)

Status: **first full version coded (2026-10-07); hardware tuning still to do.** See the [Progress log](#progress-log) at the bottom.

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

1. [x] `game.py` and `render.py` with keyboard/mouse only: physics, hit test, look.
2. [x] `vision.py`: pose paddle tracking and the AprilTag-triggered start (code done; needs a live test with the phone camera).
3. [x] `paddle_imu.py`: `check` mode and haptics (code done; `SWING_THRESHOLD` still needs tuning on the real motor).
4. [ ] Tune thresholds on hardware. `README.md` is written.

## Progress log

Updated as the code is written: what's done, what changed from the plan, and tuned values.

- **2026-10-07**: Plan written and decisions confirmed. No code yet.
- **2026-10-07**: Coded the first full version: `game.py`, `paddle_imu.py`, `vision.py`, `render.py`, `pingpong.py` and `README.md`.
  - **Changes from the plan:**
    - Game coordinates are in **meters** (real table: 2.74 m × 1.525 m), not −1..1, so ball speeds are in m/s. Level 1 is 3.0 m/s, +15% per level.
    - Shots fly on **scripted arcs** (hit → one bounce → a known arrival time at the receiver's plane) instead of free physics. This makes the swing-timing check exact.
    - Pose and AprilTag run on the **un-mirrored** frame: mirrored tags don't decode, and landmark 16 stays my real right wrist. Only the display is mirrored. The `M` key handles a camera that mirrors by itself.
    - The paddle position is **body-relative** (shoulder midpoint and shoulder width, wrist height between hip and shoulder), so standing distance doesn't matter.
    - The hit test uses the paddle's **closest point over the last 0.3 s**, which allows for camera lag.
    - Swing detection is **muted while a haptic pattern plays**, so the motor's own buzzing isn't read as a swing. Motor end state is coast.
    - The IMU calibration is optional. Without it, forehand vs. backhand comes from which side of the body the wrist is on.
    - Added synthesized sounds (hit, bounce, point) and the `--keyboard` test mode (mouse = paddle, space = swing, T = fake tag).
    - A new game needs the tag to be **shown again** after game over, so it doesn't restart on its own.
  - **Tested headlessly:**
    - Simulated games: a perfect player wins with rallies up to 14; a sloppy one loses 1–11.
    - Screenshots checked.
    - Full main loop in keyboard mode: tag start → game over → `players.json` saved.
    - Vision thread on a synthetic AprilTag frame: tag ID decoded, about 37 fps.
  - **Still untested on hardware:**
    - Gyro units and rate. `SWING_THRESHOLD = 300` and `SWING_FULL = 1000` are guesses; run `python paddle_imu.py check`.
    - Haptic feel.
    - Wrist-to-paddle mapping with a real person.
    - Phone camera latency, which may need a larger `TIMING_WINDOW`.
  - **Tuning so far:**
    - CPU aim error σ = max(0.03, 0.12 − 0.01·level) m and reach 0.15 m. The first guess, σ ≈ 0.28 m, made the CPU miss most balls.
- **2026-10-07 (fixes after first hardware test)**: Swing detection works on the real motor, but there was no buzz in `paddle_imu.py check`. That mode only printed swings and never sent a haptic; haptics fire only on hits inside the game. `check` now buzzes on every swing (strength picks `hit`/`good`/`perfect`), and the haptics thread prints `[paddle] haptic: <name>` each time it plays one. Running `game.py` directly did nothing because it's the logic module; it now launches the game like `pingpong.py`.
- **2026-10-07 (paddle orientation + forehand/backhand)**: Feedback after playing: the paddle only followed my position, and I could win with one stroke type every time.
  - **Paddle orientation from the IMU:** yaw/pitch/roll (tenths of a degree) relative to a neutral pose become aim/tilt/twist (`ANGLE_SOURCES` maps them; the signs may need flipping). Neutral is captured during the countdown ("hold your paddle facing the screen"); the yaw alone is re-zeroed at every serve to cancel gyro drift. Z re-zeros by hand.
  - **The shot goes where the face points:** aim ±35° = sideline, beyond 55° = out (CPU's point, "Out!"). Tilt: open face = higher arc, slower, deeper; closed = flatter and faster. Timing and off-center no longer steer the ball.
  - **On screen:** the paddle narrows with aim, flattens with tilt and rotates with twist. Red rubber shows on my right (forehand side), black on my left. A dotted aim guide on the table shows the landing spot.
  - **Forehand/backhand is enforced:** a ball arriving at x > 0.10 m (my right) needs a forehand, x < −0.10 m a backhand, and the middle accepts either. The wrong stroke shows "Forehand!"/"Backhand!" and loses the point. A FOREHAND/BACKHAND cue shows for every incoming ball.
  - **Stroke detection:** the gyro vector summed over each swing is compared with averaged recordings of 3 forehands and 3 backhands (C in the game or `paddle_imu.py calibrate`, saved to `calibration.json`). Uncalibrated, it falls back on which way the yaw turned during the swing (`FOREHAND_YAW_SIGN = -1`, a guess). The old pose-side fallback is gone, since it made the stroke check meaningless.
  - **Keyboard mode:** F/B swing a forehand/backhand; space accepts either.
  - **Tested headlessly:** simulated games (perfect strokes win 11–0; mixed strokes with wrong ones lose 4–11), screenshots, and the main loop with F/B/space. Still untested on the real hub: the angle axis mapping and signs, and how well calibration separates the strokes.
- **2026-10-07 (swing-back fix + per-game setup)**:
  - **Swing-back fix:** calibration counted the arm swinging back after each stroke as a second swing. A swing whose direction (summed gyro vector) is opposite to the previous real swing (dot product < −0.3) within `RETURN_WINDOW_S = 0.9` s is now thrown away as the swing-back. This applies in calibration and in play; consecutive real strokes in a rally are more than 0.9 s apart.
  - **Per-game setup:** after the tag is seen, every game runs WAITING → **SETUP_FLAT** → **SETUP_STROKES** → COUNTDOWN.
    - **SETUP_FLAT:** hold the paddle flat, facing the screen, and still (gyro w < 0.25 × `SWING_THRESHOLD` for 1.5 s, with an 8 s timeout). This zeros aim/tilt/twist.
    - **SETUP_STROKES:** 3 forehands + 3 backhands are recorded, so every run is calibrated fresh. The result is still saved to `calibration.json` for `paddle_imu.py check`.
    - Each step has its own on-screen screen with a stillness bar and a stroke counter. Keyboard mode (or no motor) skips straight to the countdown. R cancels the setup. The zeroing at countdown was removed; yaw is still re-zeroed at every serve.
  - **Tested with simulated IMU data:** 3 forehands + 3 backhands, each followed by a swing-back, calibrated with exactly 6 recordings (similarity −1.00); game swings FH, BH, FH (each with a swing-back) were classified FH, BH, FH. The main loop with a fake paddle went WAITING → SETUP_FLAT → SETUP_STROKES → COUNTDOWN; screenshots checked; the keyboard game and the game sim still pass.
- **2026-10-07 (gentler angle controls)**: The angle controls were far too sensitive, and aiming made missing too easy.
  - **Aim:** now a 10° deadzone (straight), full aim only at 50°, and the widest shot lands at 55% of the way to the sideline (~0.42 m from center), so it always stays on the table. Out shots are off (`OUT_AIM_DEG = None`).
  - **Tilt:** its effect is scaled down to 25% (`TILT_EFFECT`) over a ±60° range, so it barely changes the arc, speed and depth.
  - The aim guide and the real shot now share one function (`aim_landing_x`), so they always match.
- **2026-10-07 (harder minimum swing + flipped-hub backhand)**:
  - **Minimum swing:** small movements counted as swings. A swing now starts at w > 550 (was 300) and only counts if it peaks at ≥ `SWING_MIN_PEAK = 800`. Strength runs from 800 to `SWING_FULL = 1600`. `STILL_W` is a fixed 140, and the GUI meter's tick marks the minimum peak.
  - **Backhand detection:** backhands didn't work even after calibrating. Root cause: on a backhand the hub turns over (the other face of the motor leads toward the camera). The gyro is in the hub's own axes, so the flip cancels the reversed world rotation, and the summed spin vector looks the same for both strokes. That is why spin-only calibration failed.
    - **Fix:** each swing also records how the hub is held, as the unit accelerometer (gravity) vector from the last calm sample (w < 300) before the swing. Calibration stores averages of both the spin and the hold for each stroke. Classification scores `dot(spin, stroke_spin) + 1.5 × dot(hold, stroke_hold)`, so the hold dominates.
    - The calibration printout shows both similarities.
  - **Per-stroke "straight":** calibration also stores each stroke's average aim/tilt at the peak, and a swing's aim/tilt is measured from its own stroke's average. A backhand's raw angles are offset (and its tilt flipped if the hub is upside down, `backhand_tilt_sign`). The on-screen paddle and aim guide use the stroke matching the paddle's side of the body.
  - **Tested with simulated IMU data:** with the hub flipped for backhands (spin similarity +1.00, hold −1.00), FH/BH/BH/FH/BH were all classified correctly, and a small flick (peak 660) was ignored. The main-loop smoke test and the setup flow still pass.
- **2026-10-07 (zeroing step skipped)**: The "hold the paddle still" setup step passed instantly or flashed for a split second. Cause: it used the paddle's total still time, so if the paddle had been still while I held up the tag, the 1.5 s was already met when the step appeared. Fix: `Paddle.held_still_since(now, step_start)` counts stillness only from `SETUP_READ_S = 1.0` s after the step starts, so the step always lasts at least 2.5 s. The screen shows "get the paddle into position..." during that first second. The prompt now says hold the paddle **upright**, face toward the screen (it said "flat"). The timeout went from 8 s to 10 s. Tested with a fake paddle that was already still: the step now lasts 2.50 s (it was ~0 s before).
- **2026-10-07 (live score over MQTT)**: The score is published in real time to topic `ME193/Rogers/Hudson` on `test.mosquitto.org`, using the same `mqttlib.py` wrapper as HW3/HW4 (copied into `Midterm/`).
  - **Score:** the record number of continuous hits as a float string (`"7.0"`), computed as `max(saved best_rally, this game's best rally, current rally)`, so it rises mid-rally as soon as a record is beaten.
  - **Sending:** `score_mqtt.ScorePublisher` connects on a background thread (3 retries), publishes on change and every 2 s, and only after a player card has been seen. The left panel has an MQTT status line. Turn it off with `--no-mqtt`.
  - **Tested live:** a subscriber on the topic received the values during a keyboard-mode game.

## Part 2: LEGO Wii Remote for Dolphin (`Midterm/wii/`)

- **2026-10-07 (first version)**: Goal: use the Double Motor as an HID controller in the Dolphin emulator for Wii Sports, reusing the ping pong code (`paddle_imu.connect`/haptics, `vision.Vision`). Dolphin isn't installed yet, so this was built against Dolphin's built-in **DSU (cemuhook) client** protocol, with no extra drivers.
  - **`dsu_server.py`:** a UDP DSU server on 127.0.0.1:26760 that answers version/ports/pad-data requests and streams 100-byte pad packets (buttons, two sticks, accel in g, gyro in deg/s, µs timestamp) at 125 Hz to subscribed clients.
  - **`hub.py`:** the hub IMU in Wii Remote axes (x left, y back, z up), from a **two-pose calibration**: hold flat (gravity = up, its size = 1 g), then point at the ceiling (gravity = forward). The gyro's raw integral over the 90° tip gives its units, snapped to a power of ten. Saved to `wii_calibration.json`. `ACCEL_SIGN` covers an accelerometer that reports gravity instead of its reaction, which two still poses can't detect.
    - **Shaft buttons:** with the motors coasting, turning the **left shaft** ≥15° = **B** (hold), the **right shaft** = **A**, with 7° release hysteresis and auto re-rest after 6 s untouched.
  - **`gestures.py`:** the **right wrist** (body-relative) is the **pointer**, sent as the right stick. **Left-hand gestures:** above the head = A, both hands up for 1 s = HOME, out wide = D-pad Left, across the body = D-pad Right.
  - **`wii_remote.py`:** status window (camera/skeleton, pointer, button sources, motion bars, DSU/hub/pose status), keyboard backup for every button, and calibration prompts. **`--map`** pulses one control at a time so each Dolphin field can be bound by detection (correct regardless of DSU axis conventions). Also `--buzz`, `--no-camera`, `--no-motor`, `--recalibrate`.
  - **`profiles/LEGO Wii Remote.ini`:** a best-guess Dolphin profile. The DSU input/device names are from memory and unverified, so `--map` is the documented fallback.
  - **Tested:** a protocol client checked the version/ports/pad-data replies (CRC, 100-byte pads, ~118 packets/s); the calibration with a randomly mounted hub came out right (30° roll right → accel (left, back, up) = (+0.5, 0, +0.866); yaw left 100 dps → (0, 0, +100); gyro units 0.1 detected); the shaft press/release sequence is correct; and the full app ran headless with a simulated hub (CAL_FLAT → CAL_UP → PLAY) and in map mode, with screenshots checked.
  - **Not yet tested with real Dolphin or the real hub:** Dolphin's DSU input names (profile), the hub's accelerometer sign and units, and shaft position reporting while coasting.
- **2026-10-08 (configured the real Dolphin)**: Dolphin 2609a is installed (`OneDrive\Desktop\dolphin-2609a-x64`, user folder `%APPDATA%\Dolphin Emulator`).
  - **Names read from `Dolphin.exe`:** the DSU input names are `Pad W/S/E/N`, `Square`, `Cross`, `Circle`, `L1`…`R3`, `Share`, `Touch Button`, `Left/Right X/Y ±`, `Touch X/Y ±`, `Accel Up/Down/Left/Right/Forward/Backward` and `Gyro Pitch/Roll/Yaw …`. `Triangle`, `Options` and `PS` weren't in that block, probably because the strings are stored once elsewhere in the exe. Dolphin's default `WiimoteNew.ini` confirmed the profile keys (`IMUAccelerometer/Up`, `IR/Up`, …), and DSU servers are stored as `[Server] Enabled / Entries = description:address:port;`.
  - **Profile fix:** the D-pad was wrong (`Up/Down/Left/Right` → `Pad N/S/E/W`). The device is now `DSUClient/0/LEGO Wii Remote`, inferred: Dolphin names DSU devices after the server description, and the default description is `DS4`.
  - **New `wii/setup_dolphin.py`:** run with Dolphin closed. It backs up `DSUClient.ini` and `WiimoteNew.ini`, enables the DSU client with our server, installs the profile and applies it to Wii Remote 1. Already run here.
  - **Verified live:** launched Dolphin with our DSU server running. Dolphin connected by itself (26 controller-list + 26 pad-data subscriptions in ~25 s) and received ~3,200 pad packets. After a normal close, its config still has our settings.
  - **Still to check by hand:** that the Device dropdown shows `DSUClient/0/LEGO Wii Remote`, the axis directions in the Motion Input preview, and playing with the real hub.
- **2026-10-09 (AprilTag pointer)**: The pointer now comes from an **AprilTag taped to the front of the Double Motor** instead of the right wrist (new `wii/tag_pointer.py`, using the corners `vision.py` already detects).
  - **How it works:** the pointer = the tag's position in the frame, relative to a center set with **P**, smoothed. It holds for 0.4 s when the tag blinks out, and the right wrist takes over after 1 s without the tag. `--pointer wrist` restores the old behavior. The camera panel outlines the tag, and there's a "Pointer tag" status line.
  - **Aim from tag pose tested and turned off by default:** the tag's 3D facing direction (`solvePnPGeneric` with IPPE_SQUARE; both solutions with an ambiguity tie-break) was also implemented. Synthetic renders through the same ArUco path, with ≤1° tilt and 0.3 px noise, gave the right signs on average (turn right/left/up/down, moves, a tag taped rotated 90°). But the worst-case errors were ~40° (mirror-image pose flips), and at 2 m it barely separated 15° up from flat. So `AIM_WEIGHT = 0`; position alone is reliable. (Exactly square-on synthetic tags were degenerate for IPPE, which is why the noise was added.)
  - **Tested:** the hand-off sequence (tag → hold → wrist → tag), recenter, a screenshot, and the headless app calibration flow still passing.
- **2026-10-09 (pointer inverted)**: On the real setup, the tag pointer moved backwards on **both** axes. Added `INVERT_X` / `INVERT_Y` in `tag_pointer.py`, both `True`; each axis can be flipped back on its own. Re-ran the hand-off test: a tag right of center now gives a negative x, and recenter still gives (0, 0). The wrist fallback is unchanged.
- **2026-10-09 (shaft swap + pointer can't reach the bottom)**:
  - **Shafts:** now left shaft = **A** and right shaft = **B** (was the other way). Any turn over 15° in either direction presses (unchanged: it uses |offset|). The status line shows "left = A / right = B". Tested: ±16° on the left → A, ±20° on the right → B, 10° → nothing.
  - **Pointer:** it couldn't reach the bottom of the screen. The vertical position shared the width-based range, so the edge needed the tag ~39% of the frame height from center, near the frame edge, where a lowered hand leaves the view. Added a separate `POSITION_RANGE_Y = 0.25` (fraction of frame height), so the full top/bottom is reached at 25% from center. Tested: 0.65 → 0.60, 0.75 → full.
- **2026-10-09 (held backwards + click shafts)**: The motor is held "backwards" (its front, with the tag, faces the screen).
  - **Shafts:** swapped back to hub LEFT motor = **B**, hub RIGHT = **A**, i.e. *your* left shaft = A and your right = B. The status line now labels them by button.
  - **Click mode:** a new `SHAFT_MODE = "click"` (default). Each turn past 15°, either direction, sends one 0.12 s press, then the rest point follows the shaft for 0.35 s so the rest of that turn doesn't re-click. Leaving it turned no longer holds the button, and another turn makes another click. `"hold"` keeps the old behavior (bowling B can be held with Space).
  - **Swing flip toggle:** the **F** key negates the Wii-frame left/right and forward/back of accel and gyro (180° turn about up). It's saved as `"flipped"` in `wii_calibration.json`, kept across recalibration, and shown on the Axes status line.
  - **Tested:** a 20° turn → one ~0.12 s click and no hold; a second turn → one more click; the other shaft → the other button; the toggle saves and persists; the app's calibration flow and screenshot still pass.
  - **Note:** the headless app test used DSU port 26760, and the user's open Dolphin briefly connected to it. Use a different port in future tests.
- **2026-10-09 (pointer still can't reach the bottom)**: In a Wii Sports screenshot the pointer bottomed out about 2/3 of the way down even after `POSITION_RANGE_Y`. Dolphin's Wii Remote 1 had **default IR settings**: Vertical Offset (10 cm, which shifts the pointing range up) and Total Pitch. So a full-down stick can't reach the screen bottom, and the limit is on Dolphin's side. Fix: **IR/Vertical Offset = 0** and **IR/Total Pitch = 30**, added to the profile (`setup_dolphin.py` applies it). Dolphin was running, so the user sets them in the Configure → Point section (editing `WiimoteNew.ini` while Dolphin runs gets overwritten). Also confirmed: Dolphin kept `Device = DSUClient/0/LEGO Wii Remote`, so the inferred device name was right.
- **2026-10-09 (always backhand → hub data rate)**: Wii Sports read every swing as a backhand.
  - **Diagnosis:** added `--log` (records raw + Wii-frame IMU to `wii/swing_log.csv`) and `wii/analyze_swings.py`. The user recorded 5 forehands, a pause, then 5 backhands. Result: **only 9 samples/s** (446 in 50 s), with 1–9 samples per swing and several swings a single frozen value. That's far too coarse for the game to see the swing shape.
  - **The axes look right; flip not needed:** every swing showed strong +back acceleration (+3–6 g, the arm swing's pull toward the body), forehands turned yaw + and backhands yaw −, and at rest the remote reads up +0.90 g / back −0.36 g (held slightly nose-up).
  - **Cause:** legoeducation's `connect()` defaults `device_notification_delay` to 100 ms. The hub accepts 15–1000 ms.
  - **Fix:** `paddle_imu.connect(notification_ms=...)` (default unchanged, so ping pong is unaffected). `wii/hub.py` now uses `NOTIFICATION_MS = 15` (~66 Hz).
  - **Next:** re-record to confirm the rate and that forehand/backhand register.
- **2026-10-09 (second recording, left/right mirrored to Dolphin)**:
  - **Rate:** with `NOTIFICATION_MS = 15` the hub delivers **38 Hz** (1318 samples in 34.6 s), not 66, so the hub/BLE link seems to top out there. Swings now get 25–31 samples (were 1–9).
  - **Data:** forehands (#3–7) pushed strongly **+left** (up to +5 g) with yaw + (right-to-left sweep), physically correct in our calibrated frame. The backhands (#10, 12, 14, 16: yaw −) were weak (~1 g). Each was followed ~0.7 s later by a strong yaw +, +4 g left return motion (#11, 13, 15, 17).
  - **Conclusion:** if Dolphin read our x correctly, forehands would register as forehands. Every strong swing reading as a backhand means Dolphin sees x mirrored. A pure x-mirror can't come from how the hub is held (it's not a rotation), so it's the DSU convention guess.
  - **Fix:** `DSU_ACCEL_SIGNS = (-1, 1, 1)`. Gyro left as is (Wii Sports uses only the accelerometer, and IMU pointing is off).
  - **Technique note for the user:** a weak backhand followed by a strong return swing can register as backhand + forehand. Make the backhand the strong motion, and bring the arm back slowly.
