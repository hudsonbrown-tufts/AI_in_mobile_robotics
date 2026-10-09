# LEGO Wii Remote for Dolphin (Wii Sports)

This turns the LEGO Double Motor into an emulated Wii Remote for the
[Dolphin](https://dolphin-emu.org) emulator. It builds on the ping pong game one folder up: the motor connection, haptics, camera and pose code are all reused from there.

| Wii Remote part | Comes from |
|---|---|
| Accelerometer + gyro (swinging, tilting) | The hub's real IMU, rotated into Wii Remote axes |
| **A** | One click per turn of **your left shaft** past 15°, either direction, or **raising your left hand above your head** |
| **B** | One click per turn of **your right shaft** past 15°, either direction (hold B with **Space** when a game needs it held) |
| Pointer (aiming at the screen) | An **AprilTag taped to the front of the motor**, seen by the phone camera. Your right wrist takes over if the tag is out of sight for 1 s. |
| HOME | Both hands above your head for 1 s |
| D-pad left / right | Left hand out wide to your left / across your body to the right |
| Every button | Keyboard keys while this program's window is focused (backup) |
| Swings held backwards | Press **F** to flip the motion front/back (remembered in `wii_calibration.json`) |

It talks to Dolphin as a **DSU ("cemuhook") motion controller** on `127.0.0.1:26760`. That's the same protocol DS4Windows and BetterJoy use, and Dolphin supports it out of the box.

## The pointer tag

Tape an AprilTag (family **36h11**, any ID) flat on the **front** of the Double Motor, the end you aim at the screen. Put the phone camera by the screen, facing you. When you aim the remote at the screen, the camera sees the tag, the way a real Wii Remote sees the sensor bar.

- **The pointer follows where the tag is in the camera frame.** Move the remote and the pointer moves with it.
- **Recenter:** hold the remote where you want the middle of the screen to be, then press **P** in the LEGO window.
- **Size:** make it about 4 cm or larger, and print it on matte paper with its white border. A small tag far from the camera may not be detected; the "Pointer tag" status line shows whether it's seen.
- **If the camera can see two tags** (for example the ping pong player card), set `POINTER_TAG_ID` in `tag_pointer.py` to the remote's tag ID.
- **Aiming by tilting the tag is off by default** (`AIM_WEIGHT = 0`). In synthetic tests with realistic noise, a small tag's 3D angle sometimes flipped to its mirror image, with errors up to about 40°, so it made the pointer jump. Its position was accurate. With a large tag held close to the camera, you can try `AIM_WEIGHT = 0.5`.
- `--pointer wrist` goes back to the old wrist pointer.
- **If the pointer can't reach the bottom of the screen**, Dolphin's pointer range is the limit, not the tag. In Dolphin → Controllers → Wii Remote 1 → **Configure**, find the **Point** settings. Set **Vertical Offset = 0 cm** (the default 10 cm shifts the range up) and **Total Pitch = 30°** (more up/down range). The profile now includes these.

## 1. Build tip: levers on the shafts

The shafts turn freely while the program runs, because the motors coast. Push a short LEGO beam onto each shaft so you can flick them with a finger:
- **Your left shaft = A, your right shaft = B.** The motor is held "backwards" (its front, with the pointer tag, faces the screen), so in `hub.py` that's the hub's *right* motor = A and its *left* motor = B.

Each turn of a shaft past 15°, in **either direction**, is **one click**: a short press of about 0.12 s. Leaving it turned doesn't hold the button, and the rest of the same turn doesn't click again. Turn it again for another click. Set `SHAFT_MODE = "hold"` in `hub.py` for the old hold-while-turned behavior. (In hold mode, a shaft left turned for 6 s becomes its new resting position.)

## 2. Run the controller

```
python Midterm/wii/wii_remote.py
```

The first time, it walks you through a **two-pose axis setup** (5 seconds). Redo it any time with `C` or `--recalibrate`.
1. Hold the motor like a Wii Remote: aimed at the screen, top facing up, still.
2. Tip it to point straight up at the ceiling, and hold it still.

From those two poses it works out the hub's axes, the size of 1 g, and the gyro's units. Everything is saved to `wii_calibration.json`.

`--log` records the motion stream to `swing_log.csv`, and `python Midterm/wii/analyze_swings.py` summarizes each swing in it.

Other options: `--no-camera`, `--no-motor`, `--buzz` (buzzes after each swing; this shakes the motion data a little) and `--map` (see step 4).

## 3. Set up Dolphin (once)

**Automatic** (already run on this PC on 2026-10-08):

```
python Midterm/wii/setup_dolphin.py      # with Dolphin closed
```

This backs up `DSUClient.ini` and `WiimoteNew.ini` (as `*.bak-<timestamp>`), then:
- enables the DSU client and adds the server `LEGO Wii Remote:127.0.0.1:26760`;
- installs `profiles/LEGO Wii Remote.ini` into Dolphin's Wiimote profiles;
- applies it to Wii Remote 1 (emulated, extension None, IMU pointer off).

The profile's input names were read out of Dolphin 2609a's own `Dolphin.exe`. For example, the D-pad is `Pad N/S/E/W`, and the motion inputs are `Accel Up`, `Gyro Pitch Up` and so on.

**One thing to check:** Dolphin names the DSU device after the server's description, so it should appear as **`DSUClient/0/LEGO Wii Remote`**. Run `wii_remote.py`, open Dolphin → Controllers → Wii Remote 1 → **Configure**, and look at the *Device* dropdown. If the device shows under a different name, select it there. The bindings don't include the device name, so they keep working.

**Manual** (if you'd rather click through it):
1. **Controllers** → set **Wii Remote 1 = Emulated Wii Remote**.
2. **Controllers** → **Alternate Input Sources** → **DSU Client** → tick **Enable**, then add `127.0.0.1`, port `26760`.
3. Wii Remote 1 → **Configure** → pick the DSUClient device → **Load** the `LEGO Wii Remote` profile, or map everything with `--map` (step 4).
4. **Motion Input** tab: **Point → Enabled** unticked (the pointer comes from your wrist). Leave **Motion Simulation** unbound. Extension: **None**. Wii Sports doesn't use MotionPlus, and Boxing needs a Nunchuk, which isn't supported.

**Verified with the real Dolphin 2609a:** with the server running, Dolphin connected on its own. It requested the controller list and subscribed to pad data 26 times each in about 25 s, and received about 3,200 pad packets.

## 4. `--map`: map every control by detection

`--map` shows one control at a time and **pulses only that input**: a button press, a pointer direction, a 3 g push on one accelerometer axis, or a 500 °/s spin.
1. In Dolphin's Wii Remote 1 configuration, click the field named in the window (for example *Accelerometer > Up*).
2. Dolphin detects the pulsing input and binds it.
3. Press `N` (in the LEGO window) for the next control, or `P` to go back.

Bindings made this way are right by construction, whatever axis conventions Dolphin uses internally. The gyroscope fields are optional, since Wii Sports only reads the accelerometer.

## 5. Playing Wii Sports

- **Menus:** aim the remote (its tag) at the screen, then press A (left shaft or left hand up).
- **Tennis:** swing like the real thing. Serving is a swing too.
- **Bowling:** hold **B** through your swing and **let go** to release the ball. Shaft clicks can't hold, so hold **Space** (or switch to `SHAFT_MODE = "hold"`). D-pad left/right (left-hand gestures or the arrow keys) moves you.
- **Baseball / golf:** swing.
- **Pause / HOME:** `=` (+) or `H` on the keyboard, or both hands up for HOME.

## Check the axes (once)

After calibrating, look at Dolphin's **Motion Input** preview while you hold the motor:
- **Flat:** the preview is level.
- **Roll it to the right:** the preview rolls right.
- **Point it up:** the preview points up.

If rolling right shows it rolling **left**, set `ACCEL_SIGN = -1` in `hub.py` and recalibrate. Two still poses can't tell which way is truly "up" for this accelerometer; that one setting can. If you mapped with `--map`, other directions will already be right.

## Files

| File | What it does |
|---|---|
| `wii_remote.py` | Main program: calibration steps, `--map`, keyboard, status window |
| `hub.py` | Hub IMU → Wii Remote axes (two-pose calibration, auto units), shaft buttons, haptics |
| `tag_pointer.py` | AprilTag on the remote → pointer (position, plus optional facing direction) |
| `gestures.py` | Pose → button gestures (left hand), and the fallback right-wrist pointer |
| `dsu_server.py` | DSU/cemuhook UDP server that Dolphin connects to |
| `profiles/LEGO Wii Remote.ini` | Dolphin Wii Remote profile (input names taken from Dolphin.exe) |
| `setup_dolphin.py` | Configures Dolphin automatically: DSU server, profile, Wii Remote 1 (backs up first) |
| `_paths.py` | Lets this folder import `paddle_imu.py` and `vision.py` from `Midterm/` |

## Tuning

| Setting | File | Meaning |
|---|---|---|
| `SHAFT_PRESS_DEG`, `SHAFT_RELEASE_DEG` | `hub.py` | How far to turn a shaft to press it, and how close to rest counts as released |
| `SHAFT_BUTTONS`, `SHAFT_MODE`, `SHAFT_CLICK_S` | `hub.py` | Which shaft is A and which is B (hub motor names), click or hold, and click length |
| `NOTIFICATION_MS` | `hub.py` | How often the hub sends motion data: 15 ms ≈ 66 Hz, the hub's fastest. The library default of 100 ms (~9 Hz) made Wii Sports read every swing as a backhand. |
| `ACCEL_SIGN` | `hub.py` | See "Check the axes" |
| `DSU_ACCEL_SIGNS`, `DSU_GYRO_SIGNS` | `hub.py` | Flip a single axis if you used the profile and one direction is backwards (not needed with `--map`) |
| `POINTER_X_RANGE`, `POINTER_Y_CENTER`, `POINTER_Y_RANGE`, `POINTER_X_OFFSET` | `gestures.py` | How far the pointer moves for your wrist, and where its center is |
| `GESTURE_HOLD_S`, `DPAD_OUT_WIDTHS`, `DPAD_ACROSS_WIDTHS` | `gestures.py` | Gesture timing and distances |
| `POSITION_RANGE`, `POSITION_RANGE_Y` | `tag_pointer.py` | How far the tag moves from center to reach the screen's side edge (fraction of frame width, 0.22) or its top/bottom (fraction of frame height, 0.25). Lower = less movement needed. |
| `INVERT_X`, `INVERT_Y` | `tag_pointer.py` | Flip a pointer axis. Both are on, since both were backwards on the real setup. |
| `POINTER_TAG_ID`, `AIM_WEIGHT`, `SMOOTHING` | `tag_pointer.py` | Which tag is the pointer, the optional tilt-aim, and pointer smoothing |
