# LEGO Wii Remote for Dolphin (Wii Sports)

This turns the LEGO Double Motor into an emulated Wii Remote for the
[Dolphin](https://dolphin-emu.org) emulator. It builds on the ping pong game one folder up: the motor connection, haptics, camera and pose code are all reused from there.

| Wii Remote part | Comes from |
|---|---|
| Accelerometer + gyro (swinging, tilting) | The hub's real IMU, rotated into Wii Remote axes |
| **B** (the trigger underneath) | Turning the **left motor shaft** about 15°. Hold it turned to keep holding B. |
| **A** | Turning the **right motor shaft** about 15°, or **raising your left hand above your head** |
| Pointer (aiming at the screen) | Your **right wrist** in the phone camera (MediaPipe Pose) |
| HOME | Both hands above your head for 1 s |
| D-pad left / right | Left hand out wide to your left / across your body to the right |
| Every button | Keyboard keys while this program's window is focused (backup) |

It talks to Dolphin as a **DSU ("cemuhook") motion controller** on `127.0.0.1:26760`. That's the same protocol DS4Windows and BetterJoy use, and Dolphin supports it out of the box. The AprilTag isn't needed here.

## 1. Build tip: levers on the shafts

The shafts turn freely while the program runs, because the motors coast. Push a short LEGO beam onto each shaft so you can flick them with a finger:
- **Left shaft = B:** a trigger under your index finger.
- **Right shaft = A:** a lever under your thumb.

Turning a shaft past 15° presses the button, and turning it back releases it. If you leave a shaft turned and stop touching it for 6 s, that position becomes its new resting position.

## 2. Run the controller

```
python Midterm/wii/wii_remote.py
```

The first time, it walks you through a **two-pose axis setup** (5 seconds). Redo it any time with `C` or `--recalibrate`.
1. Hold the motor like a Wii Remote: aimed at the screen, top facing up, still.
2. Tip it to point straight up at the ceiling, and hold it still.

From those two poses it works out the hub's axes, the size of 1 g, and the gyro's units. Everything is saved to `wii_calibration.json`.

Other options: `--no-camera`, `--no-motor`, `--buzz` (buzzes after each swing; this shakes the motion data a little) and `--map` (see step 4).

## 3. Set up Dolphin (once)

1. Install Dolphin and add your Wii Sports game.
2. **Controllers** → under *Wii Remotes*, set **Wii Remote 1 = Emulated Wii Remote**.
3. **Controllers** → **Alternate Input Sources** → **DSU Client** tab → tick **Enable**. Add a server: address `127.0.0.1`, port `26760`.
4. With `wii_remote.py` running, its window should show **Dolphin (DSU): connected**.
5. Wii Remote 1 → **Configure**. In the *Device* dropdown, pick the **DSUClient** device.
6. Map the controls. Either:
   - **Quick:** copy `profiles/LEGO Wii Remote.ini` into Dolphin's profile folder and **Load** it. The folder is `<Dolphin user folder>\Config\Profiles\Wiimote\`, where the user folder is `%APPDATA%\Dolphin Emulator` or `Documents\Dolphin Emulator`. Then re-pick the DSU device in the dropdown if it isn't selected.

     The input names in this file are my best recollection of Dolphin's DSU names, and I couldn't check them without Dolphin installed. If some fields show up empty or red after loading, use the reliable method below.
   - **Reliable:** run `python Midterm/wii/wii_remote.py --map` (step 4).
7. In the **Motion Input** tab, make sure **Point → Enabled** is **unticked**. The pointer comes from your wrist, not the gyro. Leave the **Motion Simulation** tab unbound.
8. Extension: **None**. Wii Sports doesn't use MotionPlus. Boxing needs a Nunchuk, which isn't supported.

## 4. `--map`: map every control by detection

`--map` shows one control at a time and **pulses only that input**: a button press, a pointer direction, a 3 g push on one accelerometer axis, or a 500 °/s spin.
1. In Dolphin's Wii Remote 1 configuration, click the field named in the window (for example *Accelerometer > Up*).
2. Dolphin detects the pulsing input and binds it.
3. Press `N` (in the LEGO window) for the next control, or `P` to go back.

Bindings made this way are right by construction, whatever axis conventions Dolphin uses internally. The gyroscope fields are optional, since Wii Sports only reads the accelerometer.

## 5. Playing Wii Sports

- **Menus:** point with your right hand, then press A (right shaft or left hand up).
- **Tennis:** swing like the real thing. Serving is a swing too.
- **Bowling:** hold **B** (left shaft turned) through your swing and **let go** to release the ball. D-pad left/right (left-hand gestures or the arrow keys) moves you.
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
| `gestures.py` | Pose → pointer (right wrist) and button gestures (left hand) |
| `dsu_server.py` | DSU/cemuhook UDP server that Dolphin connects to |
| `profiles/LEGO Wii Remote.ini` | Dolphin Wii Remote profile (best guess at input names) |
| `_paths.py` | Lets this folder import `paddle_imu.py` and `vision.py` from `Midterm/` |

## Tuning

| Setting | File | Meaning |
|---|---|---|
| `SHAFT_PRESS_DEG`, `SHAFT_RELEASE_DEG` | `hub.py` | How far to turn a shaft to press it, and how close to rest counts as released |
| `SHAFT_BUTTONS` | `hub.py` | Which shaft is A and which is B |
| `ACCEL_SIGN` | `hub.py` | See "Check the axes" |
| `DSU_ACCEL_SIGNS`, `DSU_GYRO_SIGNS` | `hub.py` | Flip a single axis if you used the profile and one direction is backwards (not needed with `--map`) |
| `POINTER_X_RANGE`, `POINTER_Y_CENTER`, `POINTER_Y_RANGE`, `POINTER_X_OFFSET` | `gestures.py` | How far the pointer moves for your wrist, and where its center is |
| `GESTURE_HOLD_S`, `DPAD_OUT_WIDTHS`, `DPAD_ACROSS_WIDTHS` | `gestures.py` | Gesture timing and distances |
