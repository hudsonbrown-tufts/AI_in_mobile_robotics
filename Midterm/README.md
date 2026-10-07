# Midterm: Virtual Ping Pong

A Wii Sports–style table tennis game:

- The **LEGO Double Motor**, held by itself in your right hand, is the paddle. Its **IMU gyroscope** detects when you swing and how hard.
- The **phone camera** with **MediaPipe Pose** tracks your **right wrist**, which places the paddle over your end of the table.
- The IMU's **yaw, pitch and roll** turn, tilt and twist the paddle on screen. **Where the face points is where your shot goes.**
- Balls on your right need a **forehand** and balls on your left a **backhand**. The gyro tells them apart, so you can't win with one stroke only.
- An **AprilTag player card** starts the game when the camera first sees it. Its ID is your player profile, and your saved **level** sets the ball speed.
- The Double Motor **buzzes** in your hand when you hit the ball. There's a different buzz for perfect hits, lost points and wins.
- Your **score** is published live over **MQTT** to `ME193/Rogers/Hudson` on `test.mosquitto.org` (the same broker as the other assignments).

## Run it

From the repo root, with the `.venv` active:

```
python Midterm/pingpong.py                 # full game (phone camera + Double Motor)
python Midterm/pingpong.py --points 5      # shorter games
python Midterm/pingpong.py --no-mqtt       # don't publish the score
python Midterm/pingpong.py --keyboard      # no hardware: mouse = paddle, space/F/B = swing, T = fake tag
python Midterm/pingpong.py --list-cameras
python Midterm/paddle_imu.py check         # live swing speed + paddle angles; buzzes on every swing
python Midterm/paddle_imu.py calibrate     # record 3 forehands + 3 backhands (or press C in the game)
python Midterm/paddle_imu.py buzz          # play every haptic pattern once
```

Before you start:

1. Connect the phone camera through Phone Link. It shows up as "... (Windows Virtual Camera)".
2. Turn on the Double Motor. It connects with the purple Connection Card, serial `5164`.
3. Print an AprilTag from the **36h11** family.

## How to play

1. Stand facing the phone so your shoulders, right arm and (ideally) your hips are in view.
2. Hold up your AprilTag card. Every game then runs a short setup:
   1. **Zero the angles:** you get 1 s to get into position, then hold the paddle **upright, face toward the screen, and still** until the bar fills (1.5 s). That pose becomes "straight ahead" for aiming. If you can't hold still enough, it takes whatever pose you're in after 10 s.
   2. **Calibrate strokes:** swing **3 forehands, then 3 backhands**, holding the motor the way you naturally would for each. On a backhand the other face of the motor turns toward the camera, and that's what the game uses to tell them apart. The arm swinging back after each stroke is ignored, so just swing normally. Your natural face angle for each stroke also becomes that stroke's "straight".
   3. A 3-2-1 countdown, then play.

   Keyboard mode skips the setup.
3. (Optional) Press C any time to redo the stroke calibration, or Z to re-zero the angles.
4. **Your serve:** swing any time.
5. **Rally:** move your right hand to where the ball will arrive. The red paddle on screen follows your wrist. Swing as the ball reaches you. A hit needs both:
   - **timing:** within ±0.22 s of the ball arriving, and
   - **position:** the paddle near the ball.
6. **Right stroke:** the cue at the bottom says FOREHAND (ball coming to your right) or BACKHAND (to your left). The wrong stroke loses the point. Balls straight at you accept either.
7. **Aiming with the paddle face:**
   - **Turn the face** left or right (yaw) to steer the ball to that side. The dotted guide on the table shows where it's going. Aiming is deliberately gentle: under 10° goes straight, 50° or more is the widest shot, and the ball always stays on the table.
   - **Tilting the face** (pitch) only slightly lofts or flattens the shot.
   - A harder swing makes a faster, deeper return.
8. **Winning:** first to 11, win by 2.
   - A win moves you **up** a level and a loss moves you **down** one. Each level makes the ball 15% faster and the CPU better.
   - Your level is saved per tag in `players.json`.
   - Show your card again to play another game.

## Live score (MQTT)

| | |
|---|---|
| Broker | `test.mosquitto.org`, port 1883 |
| Topic | `ME193/Rogers/Hudson` |
| Payload | Your **record number of continuous hits** as a floating point number, e.g. `7.0` |

- The record is the player card's all-time best rally (from `players.json`).
- It updates mid-rally the moment the current rally beats it.
- It's sent whenever it changes and repeated every 2 s.
- Nothing is sent until a player card has been shown.
- The MQTT line in the left panel shows the connection and the last value sent. If the broker can't be reached, the game keeps running without it.

## Keys

| Key | Action |
|---|---|
| Q / Esc | Quit |
| C | Redo the stroke calibration (3 forehands, then 3 backhands). This also runs automatically at the start of every game. |
| Z | Re-zero the paddle angle (hold it flat, facing the screen). Done automatically in the game setup; the yaw is also re-zeroed at every serve. |
| R | Back to the start screen |
| K | Toggle keyboard/mouse test mode |
| M | Toggle camera mirroring (use it if the paddle follows your **left** hand) |
| Space / F / B | Keyboard swing: either stroke / forehand / backhand |

## Files

| File | What it does |
|---|---|
| `pingpong.py` | Main loop, AprilTag start trigger, player levels (`players.json`), events → haptics/sounds/pop-ups |
| `game.py` | Pure game logic: shot arcs, hit test, CPU opponent, scoring |
| `paddle_imu.py` | Double Motor: IMU swing detection thread and haptics thread |
| `vision.py` | Camera thread: MediaPipe Pose (right wrist → paddle) and AprilTag detection |
| `render.py` | pygame drawing: camera panel, 3D table, HUD, synthesized sounds |
| `score_mqtt.py` | Publishes the record continuous hits to `ME193/Rogers/Hudson` (background connect, resend every 2 s) |
| `mqttlib.py` | Same small paho-mqtt wrapper as HW3/HW4 |
| `models/pose_landmarker_lite.task` | MediaPipe pose model |
| `PLAN.md` | Design plan and progress log |

## Tuning

| Setting | File | Meaning |
|---|---|---|
| `SWING_THRESHOLD`, `SWING_MIN_PEAK`, `SWING_FULL` | `paddle_imu.py` | Gyro speed that starts a possible swing (550), the peak it must reach to count (800; raise it if small movements still count), and full strength (1600). The meter's yellow tick is the minimum peak. |
| `HOLD_WEIGHT` | `paddle_imu.py` | How much the hub's hold (which face points at the camera) counts against its spin when telling forehand from backhand |
| `RETURN_WINDOW_S` | `paddle_imu.py` | An opposite-direction swing this soon after a real one counts as the arm swinging back and is ignored |
| `SETUP_READ_S`, `HOLD_STILL_S`, `STILL_W` | `paddle_imu.py` | Time to get into position, then how long and how still the paddle must be held to zero the angles |
| `ANGLE_SOURCES` | `paddle_imu.py` | Which hub angle (yaw/pitch/roll) and sign drive aim, tilt and twist. Check them with `paddle_imu.py check`. |
| `FOREHAND_YAW_SIGN` | `paddle_imu.py` | Uncalibrated guess for which way the yaw turns on a forehand |
| `AIM_DEADZONE_DEG`, `AIM_FULL_DEG`, `AIM_MAX_X` | `game.py` | Face angle that still goes straight, the angle for the widest shot, and how wide that shot is (fraction of the way to the sideline) |
| `OUT_AIM_DEG` | `game.py` | `None` = aiming can never put the ball out; set a number (e.g. 80) to allow out shots |
| `TILT_FULL_DEG`, `TILT_EFFECT` | `game.py` | Face tilt for the maximum effect, and how strong that effect is (0 = ignored) |
| `STROKE_DEADZONE` | `game.py` | Balls this close to your center accept either stroke |
| `TIMING_WINDOW` | `game.py` | How forgiving the swing timing is |
| `HIT_RADIUS_X`, `HIT_RADIUS_Y` | `game.py` | How close the paddle must be to the ball |
| `BASE_BALL_SPEED`, `SPEED_PER_LEVEL` | `game.py` | Ball speed at level 1, and how much faster each level gets |
| `SHOULDER_WIDTHS_TO_EDGE`, `PADDLE_Y_AT_SHOULDER` | `vision.py` | How far your hand has to move to reach the table edge, and the paddle height when your wrist is at shoulder height |
| `HAPTICS` | `paddle_imu.py` | Buzz patterns |
