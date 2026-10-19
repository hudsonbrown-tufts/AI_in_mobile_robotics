# HW5 sumo: progress log

## 2026-10-19: first full build

**Built:** everything in PLAN.md's file layout: `make_tags.py`, `config.py`, `robot_io.py`, `vision.py`, `fusion.py`, `behavior.py`, `sounds.py`, `sumo.py`, `sim.py` and the `sumo-shake-controller/` UNO Q app. Also copied `mqttlib.py` (HW4) and `lose.mp3` (HW3).

**Changes from the plan:**
- **EKF state** is `[x, y, θ, ex, ey]`. The plan had `[x, y, θ, v]`. Wheel speed comes straight from the encoders. The extra `ex, ey` estimate "external" velocity: motion the wheels don't explain, as when pushed or slipping.
- **Camera delay:**
  - A frame is moved forward to "now" using wheel odometry only.
  - The EKF's measurement model accounts for the frame's age (`H[0,3] = H[1,4] = -age`).
  - An earlier version also pushed the frame forward by the EKF's external velocity. That made the filter go unstable (external velocity swung by ±5 ring units/s).
- **Push "residual" cue:**
  - Camera displacement minus odometry displacement over a 0.25 s window, not the EKF external-velocity estimate.
  - The external-velocity estimate turned out hard to observe through a delayed camera. Its uncertainty grew, and so did the pose uncertainty.
- **Closing speed** is the opponent's own speed toward us. Our charge into it no longer counts as "being pushed".
- **Wheel encoders added:** the Double Motor reports each wheel's position, so odometry uses wheel distance plus gyro turning. Commanded speed is used only as a fallback.
- **`d` drive calibration** drives forward, spins and drives back while watching the tags. It finds:
  - the encoder signs, and whether the motor's "forward" points at the BACK tag (`reverse`);
  - ring units per wheel degree;
  - wheel track;
  - which gyro axis is yaw, and its sign.

  Results are saved to `drive_calibration.json`.
- **Edge escape** backs away from whichever end is at risk: the front, or the back when we're pushed out tail first.
- **PUSHED near the edge** uses hysteresis (enter above r 0.7, leave below 0.6). It drives in an arc rather than spinning in place.
- **Push-back power** is 80 + 20 × shake. At 50 + 50 × shake, defense mode lost 16/20 sim matches to an equally strong chaser.
- **Shake publish rate** is 10 Hz, not 20. The laptop smooths it anyway.

**Tested headlessly (`python HW5/sim.py test`, all pass):**
- EKF converges from an unknown start: position error 0.006, heading error 0.6°.
- A 1 s camera dropout drifts 0.05 ring units.
- A 0.35 ring/s shove is detected in 0.27 s.
- False push alarms: 0.7% of the time over 20 s of free driving.
- Carried out of the ring: loss detected 1.5 s after the carry starts, about 0.5 s after crossing the tape.
- No false "lost" in a 20 s match.
- Alone in the ring: left it 0/100 times (50 runs with the camera, 50 color-sensor only).

**Sim matches** (`python HW5/sim.py matches`, 20 each, equally strong chasing opponent, 40 s limit):

| Shake | Won | Lost | Draw |
|---|---|---|---|
| 0.0 | 0 | 0 | 20 |
| 0.5 | 0 | 0 | 20 |
| 1.0 | 20 | 0 | 0 |

**Tested with a synthetic camera frame:**
- Ring auto-fit: a point on the tape reads r = 0.998.
- Two-tag pose: (-0.300, 0.100, 30.2°) against a truth of (-0.3, 0.1, 30°).
- Opponent: (0.395, -0.194) against a truth of (0.4, -0.2).
- Ring size estimated from the tags: 78 cm against 77 cm.
- The overlay renders.

**Still untested on hardware:**
- **Deploying the shake controller:** the board wasn't reachable (no adb, no answer at the `HOSTS` addresses). The sketch hasn't been compiled yet. `Bridge.provide` returning a `float` is new in this repo, since HW4 only used void handlers.
- **Shake scale:** `SHAKE_MAX = 1.8` g is a guess. Shake the board and watch the printed bar.
- **Color sensor:** the raw RGB scale and BLUE-vs-GRAY separation. Press `k` to calibrate.
- **Phone camera latency** (`CAMERA_LATENCY_S = 0.12`) and the tape HSV range under the room's lights.
- **Sim accuracy:** real wheel slip and LEGO motor stall under load. The sim lets wheels spin freely when shoved.

## 2026-10-19: ring calibration adapts to the tape color

Testing uses light blue painter's tape, and matches will use normal blue painter's tape. So `c` no longer uses one fixed HSV range.

**How it works now:**
- `Ring.auto` tries 4 hue windows × 5 saturation steps. Hue spans sky blue to deep blue. Saturation is measured above the floor's own median, so a gray floor with a blue tint drops out.
- For each mask it fits an ellipse to the biggest blobs, both together and one at a time.
- It keeps the fit that looks most like a ring of tape. The score is coverage (tape found all the way round the ellipse) × precision (blob pixels lying on the ellipse). It must beat 0.35.
- The tape's measured HSV is saved in `ring_calibration.json` and shown in the status line.
- The fit uses every tape pixel, not contour outlines. With outlines, an unbroken ring was fitted to the tape's outer edge, half a tape width (7 px) too big.

**Tested on synthetic 1280×720 frames.** Every case below came within 1.2 px on center and 0.2 px on radius, in about 0.1 s:
- deep blue, blue and light blue painter's tape;
- very pale tape (saturation 75 over a gray floor);
- a blue-tinted floor;
- uneven lighting (60–125%);
- robots covering parts of the tape;
- blue and pale-cyan distractor objects;
- a ring partly off the frame.

A frame with no tape (only a blue LEGO part) fails cleanly.

**Still to check on hardware:**
- The real light blue tape under the room's lights. The status line shows the fit score and tape HSV after `c`.
- The front color sensor on light blue tape. Run `k` calibration once per tape type, since the built-in fallback was written for normal blue.

## 2026-10-19: camera on a chair (side view) corrected for perspective

The camera sits on a chair, above and to the side of the ring. The old ellipse-only (affine) floor map assumed a camera straight overhead. From the side it fails in three ways:
- the near side of the ring is magnified;
- the ellipse's center isn't the ring's center;
- the tags, 8 cm up on the robot, appear shifted away from the camera.

**Changes:**
- **`Ring` is now a homography** (ring units ↔ pixels). `c` still gives the ellipse-based first guess.
- **`p` (new key):** with the robot in the ring, it watches the FRONT/BACK tags for 2 s, then fits in two stages (`perspective.py`):
  1. A free homography in which every tape pixel lands on the unit circle and every tag view is a square, all the same size.
  2. A real camera model fitted to the same evidence: focal length, rotation and position, with square pixels and the principal point centered.

  Tag views add up across presses. Press it at 2–3 spots toward the edge.
- **Height correction:** with the camera position known, points above the floor are corrected. A point at height h appears at `F + (P − F)·Hc/(Hc − h)`, where F is the camera's foot point and Hc its height. This applies to:
  - the tags, at `tag_height_cm` (a new `make_tags.py` flag, default 8);
  - the opponent blob's center, assumed 4 cm up.
- **Our robot's mask** for opponent detection covers its whole box from floor to tag height, because the camera sees its sides.
- **Opponent size** is measured as area on the floor, not in pixels. It's shrunk by 0.8 because its sides show in the silhouette.
- **Overlay:** draws the opponent's circle in perspective, and the status panel says whether the view is perspective-corrected.

**Tested on true 3D renders** (camera f = 0.8·width, 77 cm ring; boxes 16×12×8 cm for us and 15×13×8 cm for the opponent; 5 cm tags on top). Robot position error over 5 test poses:

| Camera | Ellipse only (old) | After `p` at 3 spots |
|---|---|---|
| Chair 1.3 m to the side, 1.0 m up | 15.3 cm mean (17.0 max) | 1.1 cm (1.3 max) |
| Chair at a corner, 0.7 m up | 21.0 cm (23.0 max) | 1.0 cm (1.3 max) |
| Nearly overhead, 1.6 m up | 2.3 cm | 0.3 cm |

- Heading error is under 2° in every case after `p`.
- Opponent error went from 11–16 cm to about 2 cm.
- The recovered camera position is within about 10% of the truth.

**Effect of the number of `p` spots** (chair views):

| Spots | Error |
|---|---|
| 1, at the ring's center | 4.5–7.8 cm |
| 1, off-center | 2 cm |
| 2 | 1.1–1.5 cm |
| 3 | 0.7–1.5 cm |

**Still to check on hardware:**
- The phone's real lens distortion. It's assumed none; phone main cameras correct it in software.
- Tag detection at the far side of the ring. Use the phone's full resolution if the far tags drop out.

## 2026-10-19: each tag has its own height

The real robot uses the already-printed tags. FRONT (id 1) is 7.5 cm off the ground and BACK (id 2) is 4.5 cm.

**Changes:**
- `robot_tags.json` now has `front_tag_height_cm` and `back_tag_height_cm`, replacing the single `tag_height_cm`.
- `make_tags.py --update` changes only the measurements given on the command line. It doesn't regenerate the images.
- Vision corrects each tag for its own height.
- The perspective fit (`p`) compares tag sizes only between views of the same tag. Tags at different heights look slightly different in size.
- Our robot's mask uses the higher of the two tags as the top of its box.

**Tested (3D chair renders, tags at 7.5 / 4.5 cm, `p` at 3 spots):**

| Heights the code was given | Position error | Heading error |
|---|---|---|
| Per tag, 7.5 / 4.5 cm | 0.8–1.5 cm mean (1.9 max) | ≤ 1.3° |
| Both 8 cm (wrong) | 4–6 cm mean (8.6 max) | up to 43° |

The heading error is large with the wrong heights because the heading comes from the line between the two tags.

**Assumption:** both tags lie flat (parallel to the floor).

## 2026-10-19: real robot measurements

**Measurements:** the robot is 14 × 13 cm. The 5 cm tags sit on the center line, with the FRONT tag's front edge at the robot's front and the BACK tag's back edge at its back.

**Tag offsets:** 4.5 cm from the robot's center (7.0 − 2.5). This assumes the black squares, not the white paper borders, reach the robot's edges.

**Saved:** in `robot_tags.json` with `make_tags.py --update`. The simulator's robot was changed to match.

**Sim tests:** all pass. False push alarms are now 1.2% of the time.

**Matches:**

| Shake | Won | Lost | Draw |
|---|---|---|---|
| 0.0 | 0 | 0 | 20 |
| 0.5 | 0 | 0 | 20 |
| 1.0 | 18 | 1 | 1 |

## 2026-10-19: hardware feedback: drives out of the ring, misses the opponent

### Driving out of the ring

**Diagnosis:** the first sim had no delays, so it never drove out. With realistic numbers it reproduces the problem:
- Bluetooth command delay of 80 ms.
- Color sensor every 15 ms, arriving 40 ms late.
- LEGO motor at about 10°/s per %.
- Motor spin-up of 0.12 s.

When the camera wasn't tracking the robot (no `p`, or tags lost), it ran on the color reflex alone. The nose crossed the tape in **30/30** of those runs, and the robot fully left in 2. The stop arrives about 0.12 s after the tape is seen, and the robot keeps rolling 4+ cm.

**Changes:**
- **Reflex:** reverses at once (−60) instead of stopping. The escape no longer has a brake pause.
- **Stopping-distance speed limit:** when the camera tracks us, forward or backward speed is capped so the leading end (nose, or tail when reversing) can stop before the tape along its actual path. It counts 0.18 s of delay and 2 ring/s² of braking. It applies to every command, including escapes.
  - **Exception while shoving the opponent** (touching, straight ahead): full power is allowed while our measured speed is still slow enough to stop.
- **Camera-blind cap:** 22%. This was tested at 35 (5/60 crossings), 28 (3/60) and 22 (1/60, by 2 cm). Overall speed cap is 80%.
- **Color "not the floor" test:** tape is anything clearly brighter (+12 reflection) or bluer (+0.05 blue share) than the learned or recorded gray floor. This catches light blue tape without `k`.
- **Drive check:** compares commanded direction with what the camera sees, and gyro turning with camera turning. If they disagree (motor backwards, `d` never run), the match stops with "press d". `space` now refuses to start until `d` has been done, and warns until `p` has been done (press `space` again to start anyway).
- **Match log:** every match writes `logs/match_<date>.csv` at 30 Hz: state, pose, color, edge risk, push cues, opponent, commands.

**Sim results (realistic delays):**
- Driving alone, 60 runs at high shake: the nose crossed the tape **1/60** (by 2 cm) and the robot was **out 0/60**, against 30/60 and 2/60 before.
- With the camera tracking: 0/60.
- Matches against an equally strong chaser, 20 each:

| Shake | Won | Lost | Draw |
|---|---|---|---|
| 0 | 18 | 0 | 2 |
| 0.5 | 18 | 0 | 2 |
| 1 | 19 | 0 | 1 |

**New sim test:** a backward-wired motor is caught in 1.5 s, and 20 normal matches had no false stops.

### Opponent detection (rewritten as `opponent.py`)

**Problems with the old detector:**
- It processed the whole frame shrunk to 480 px. From a chair the opponent was only a few pixels tall.
- It used a single-frame background with one fixed threshold.
- The background slowly absorbed a still opponent, which faded out in a few seconds.
- It was thrown off by phone auto-exposure.
- It ignored a dark robot on a dark floor.
- It took the biggest blob, and that blob's middle as the position.

**New detector:**
- Processes only the ring's region, at up to 640 px.
- The background averages 15 frames and records each pixel's noise.
- Per-channel exposure and white-balance correction every frame.
- Texture (edge) changes count as well as color changes.
- Only mild discounting of soft shadows.
- Very slow background drift, only where nothing is happening.
- Skips the tape band, and re-joins a robot split by it.
- Blobs are scored on expected size and closeness to the tracker's prediction. A new opponent must be seen twice to be confirmed.
- Position comes from the blob's near edge (where it meets the floor) as seen from the camera.
- **Opponent tag:** if the opponent has any AprilTag that isn't ours, that tag is used.
- **`o` key:** shows the detection mask (red) and candidate boxes.

**Tested (3D chair renders, two camera positions, 20 frames each):**

| Case | Found | Error |
|---|---|---|
| Red / black / floor-colored dark gray LEGO opponent | 19/20 | 0.8–1.6 cm |
| Dark gray + exposure +20% and color cast | 19/20 | 0.8–1.5 cm |
| Dark gray + shadow beside it | 19/20 | 1.5–4.8 cm |
| Touching our robot | 19/20 | 1.8–3.2 cm |
| Sitting still for 10 s | 299/300 | ≤ 1 cm |
| On the tape, 4 angles | 19/20 each | 0.2–1.1 cm |
| No opponent, exposure +15% | 0/30 false alarms | |

The miss in each row is the first frame, held for confirmation.

Processing time is 34–50 ms per frame (tags + opponent).

**Still unknown:** what actually happened on the real robot. The next run's `logs/*.csv` will show which state it was in, what the color sensor read at the tape, and whether the camera was tracking.

## 2026-10-19: `d` crashed after the Double Motor disconnected

**What happened:** pressing `d` moved the robot a little. Then the Double Motor's Bluetooth dropped, 55 "Future exception was never retrieved / Device disconnected unexpectedly" messages printed, and the program crashed saving `drive_calibration.json` ("Object of type bool is not JSON serializable").

**Causes:**
1. **The crash:** `reverse = along < 0` was a numpy bool, which `json` can't write.
2. **The disconnect (most likely cause):** the calibration's `raw_motor` re-sent the motor command every control tick, 30 times a second. Normal driving also sent a command whenever a speed changed by 3+, which can also be about 30/s while steering.
   - The LEGO library keeps a pending reply for every command. Each of those 55 messages is one command still waiting for its reply when the link dropped.
   - So commands were arriving faster than the hub answered them.

**Fixes:**
- Calibration results are cast to plain Python types before saving.
- **Commands are rate-limited** to at most 10 per second, and only sent when something changes:
  - Repeats are never resent.
  - Stops, reversing a wheel, and the edge reflex (`force`) still go out at once.
  - Measured with a fake motor: steering at 30 Hz now sends 8/s, and holding one calibration speed for 2 s sends 1 command (was 60).
- **Dropped-reply errors** are caught by an exception handler on the LEGO library's event loop. They're counted instead of printed: 50 dropped replies printed 0 lines.
- **Watchdog thread:**
  - Reconnects the motor or sensor every 3 s if it drops.
  - The status line shows "disconnected: reconnecting...".
  - A match in progress is stopped.
  - Drive calibration fails cleanly ("Bluetooth dropped... press d again") instead of solving from stale data.

**If it still disconnects:** check the hub battery and how far the laptop is from the robot.

## 2026-10-19: "spins and gets confused": the drive calibration's spin wrapped around

**Evidence:** `logs/match_20261019_154127.csv` and `drive_calibration.json`.
- The saved calibration had a wheel spacing of 0.75 ring units (29 cm on a 13 cm robot) and a gyro scale of 0.46.
- In the log, during "turn left" commands, the wheels and the camera both turned left. The gyro, as calibrated, said right.
- The camera heading jumped frame to frame, and 325 of 415 ticks were EDGE_ESCAPE.

**Cause:** this robot turns about 300°/s. The `d` spin (40% for 0.8 s) turned more than 180°, and the calibration measured the turn as `wrap(end − start)`, so a 240° left turn read as a 120° right turn.
- That saved a **backwards gyro sign** with the wrong scale and wheel spacing. The old code's `abs(track)` hid the sign.
- Fusion then predicted every turn backwards between camera frames, the camera kept correcting it, the heading jumped, and steering chased the noise.
- The motor itself was fine: forward was forward and left was left.

**Fixes:**
- The calibration adds up the camera's turn frame by frame and cross-checks it against the endpoints.
- The spin is shorter and slower (30% for 0.45 s).
- The gyro is integrated over the spin and the stop after it.
- Results that don't make sense are rejected: wheel spacing must be 0.5–1.6× the robot's width, the turn must be over 15°, and the tags must be tracked through the turn.
- **New `swap_sides`:** a left-turn command that turns the robot right (motor ports swapped, or a mirrored camera) is detected and fixed. `robot_to_motor` / `motor_to_robot_wheels` in `robot_io.py` handle reverse + swap together.
- **DriveCheck:** now a correlation between each command (as it was 0.2 s earlier) and the camera's motion over the last 0.8 s, judged every 0.3 s, with clear results counting double.
  - It adds a "turns the WRONG WAY" check.
  - Sim: a backward motor is stopped in 1.4 s, swapped sides in 2.3 s, with no false stops in 60 normal matches.
- An old `drive_calibration.json` (without `swap_sides`) counts as missing, so `space` and the setup ask for `d` again.

**Tested end to end** on the sim robot, made to turn as fast as the real one:
- All 4 mountings (reverse × swap) are detected and fixed.
- 411° spins are measured correctly.
- After calibration, a left-turn command turns left and the gyro sign is right.

### Setup walkthrough (`wizard.py`)

**On start:** it shows which calibrations are saved (ring, camera angle, color sensor, driving).
- ENTER keeps them and sets up only the missing ones.
- N starts fresh: the files move to `calibration_backup/<time>/`, then all 4 steps run.

**During setup:** step-by-step instructions appear in the side panel, and each step advances when it succeeds.
- A failed step shows its reason, and ENTER retries.
- S skips a step, and Esc leaves the walkthrough.
- `w` brings it back, and `--fresh` starts fresh from the command line.

**Tested** with stand-ins through every step, including a failed perspective spot and a failed drive calibration.

## 2026-10-19 — clicking tape points during setup
- Bug: in the setup walkthrough, ENTER was taken by the walkthrough (it re-ran the automatic tape search), so clicked points could never be submitted. Now while clicking, ENTER submits the points and Esc cancels clicking (not the walkthrough); the panel shows the point count. M works from any walkthrough step, and the ring step moves on once the clicked ring and empty-ring recording finish.

## 2026-10-19: slow by default, still defense, no more side-to-side

**What the match log showed** (`logs/match_20261019_155831.csv`):
- **Side to side.** The robot turns about 8°/s per % (about 600°/s at 70%). With about 0.25 s of camera and Bluetooth delay, a steering gain of 90/rad (max 70%) overshoots every correction, so the commands flipped ±70 every 0.25 s.
- **False pushes.** "Pushed" stayed at 0.9–1.0 for the whole match, even with the opponent about 30 cm away. During fast spins the gyro, wheel and camera streams disagree, and the push cues read that disagreement as a push. That kept the robot in the 80% push-back state.
- **Escape flip-flop.** The edge-escape "turn to center" flipped direction every tick when the robot faced straight away from the center.

**Changes:**
- **Steering:** TURN_KP 28, TURN_MAX 22, escape turn 22% (was 55). The escape keeps one direction while facing away.
- **Speed ramp (`SpeedRamp` in behavior.py):**
  - Everything is capped at CRUISE_SPEED 30% unless resisted. Resisted means either:
    - being pushed;
    - the opponent touching us on the side we're driving toward while we move at under 45% of the commanded speed for 0.2 s.
  - While resisted, the cap rises 70%/s to 80%, and it drops back at 150%/s.
  - Attack approaches at 18–30%. Push-back and shoving ask for 100%, and the ramp decides the real speed.
  - The ramp is shown on the panel (`cap NN%`, `RESISTED`) and logged (`speed_cap`, `resisted`).
- **Defense:**
  - Stays still at the center: it creeps back at 20% when r > 0.15, until r < 0.06.
  - It turns at 15% to face the opponent only when it is more than 30° off, and stops turning within 12°.
  - The sidestep dodge was removed.
  - When pushed from the side in defense, it pivots to square up instead of arcing.
- **Push detector:**
  - An opponent seen more than 0.15 ring units past touching argues strongly against a push (`far` cue, weight 4).
  - The yaw cue compares the smoothed gyro and wheel turn rates, allowing 30% of the turn rate.
  - The resid threshold grows with turn rate.
- **Edge escape:**
  - It backs off whichever way has more room, counting the opponent as a wall. With no room either way (sideways at the tape), it arcs toward the center.
  - When pushed, the back-off uses the ramp, and a slow escape turn is dropped in favor of fighting the push.
  - The speed limiter counts an inward-swinging arc as room.
- **DriveCheck:** turning is judged from a 10% turn command (DRIVE_CHECK_MIN_TURN_CMD), since turns are slow now. Forward/backward stays at 25 so shoves don't false-trigger it.

**Sim made realistic** (and the old sim was optimistic):
- Wheel speed is 14°/s per %, matching the real turn rate, and the robot under test uses that calibrated value.
- The camera is 0.06 s later than assumed, with 0.04 s of jitter.
- Encoders arrive 0.04 s late and the IMU 0.02 s late.
- The chaser now reacts to our position from 0.15 s ago; before, it saw us perfectly with zero delay.

On this sim the old code also loses most matches against a chaser, while on the old sim it won 11/12.

**Results (realistic sim):**
- **Wobble check:** with no shaking and the opponent still, the robot averages 2% power and spins 8°/s. The old code averaged 56% and 300°/s, flipping its turn direction 2.2 times per second.
- **Against a still opponent at shake 0.7:** won 10/10.
- **Against the chaser (65–90% nonstop; half the matches at equal strength, half at 80%), 30 matches each:**

  | Mode | Won | Lost | Draw |
  |---|---|---|---|
  | Defense | 0 | 10 | 20 |
  | Attack (shake 0.7) | 0 | 5 | 25 |

  The old code on the same sim lost about 80–90% of matches.
- `sim.py test`: 5/5.

## 2026-10-19: finish the push, faster aggression

**Problem (real robot):** the robot stopped short of the tape even while shoving the opponent, so the opponent never went fully out. Sim test (still opponent near the edge, attacking): before the fix, our front stopped at r 0.90, the opponent's center reached r 1.14, and it was fully out (r > 1.21) 0/20 times.

**Fix:**
- **Push-out mode** (`pushing_out` in behavior.py, set by the Brain each tick, held 0.3 s): our front is on the opponent and the opponent's center is further out than ours by 0.03.
  - No edge escape, no color reflex, and no color vote for "lost".
  - The speed limiter allows our front out to r 1.10, with our center no further than r 0.92.
- **Result (sim):** opponent fully out 20/20, our front at most r 1.03, and we went out 0/20.

**Aggression:**
- Attack now needs aggression ≥ 0.5 (was 0.15; a hand at rest reads about 0.1).
- **Faster response:**
  - Controller: RMS window 0.4 → 0.2 s, release 1.0 → 0.4 s, publishing at 20 Hz (was 10).
  - Laptop: EMA 0.35 → 0.6, MQTT timeout 2 → 1 s, decay 0.7 → 0.4 s.

**Checks:** `sim.py test` 5/5. Chaser matches are unchanged within noise (defense W0 L8 D22, attack W0 L7 D23 of 30).
