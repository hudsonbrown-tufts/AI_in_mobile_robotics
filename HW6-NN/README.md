# HW6: Ankle Activity Wearable

> **Archived version (2026-10-26).** This is the HW6 wearable with **Part 2 as a neural network** (1D CNN), kept for reference after the course required Part 2 without neural nets. The current version is in `HW6/`. Here the board app is renamed `ankle-activity-wearable-nn` so it never overwrites the HW6 app on the board (deploy with `python tools/deploy.py ankle-activity-wearable-nn`). In the rest of this file, read `ankle-activity-wearable` as `ankle-activity-wearable-nn` and `HW6/` as `HW6-NN/`.


An Arduino UNO Q with an MPU-6050, strapped to the ankle, tells whether the wearer is **standing, walking, running or biking**. Biking means lying on your back and pedaling in the air. The current activity scrolls across the UNO Q's LED matrix. A laptop GUI shows it too and switches between the two parts of the assignment:

| | Allowed input | Method |
|---|---|---|
| **Part 1** | peak acceleration in **one direction** | 3 thresholds on the 2 s peak; the direction is picked once by calibration |
| **Part 2** | **all IMU outputs** (3-axis accel + 3-axis gyro) | small 1D convolutional neural network |

Both parts are calibrated/trained **once**, through a step-by-step wizard in the GUI: press **Shift+C**, a capital C. The results are saved as files in the app, so the wearable works on anyone without recalibration. The calibration stays in use until a new one is installed. Starting a recalibration and then cancelling it changes nothing, and every installed calibration is kept in `calibration_history/`.

[HOW_IT_DECIDES.md](HOW_IT_DECIDES.md) explains in depth how each part makes its decision. This README covers everything else: hardware, files, setup, calibration, testing and troubleshooting.

---

## 1. Hardware

| Part | Notes |
|---|---|
| Arduino UNO Q | runs the sketch (STM32) and the Python app (Linux) |
| MPU-6050 (GY-521 breakout) | the same one as HW5 |
| USB-C power bank | on the ankle strap; the board talks over WiFi |

**Wiring** (same as HW5): VCC → 3.3V, GND → GND, SDA → SDA, SCL → SCL. These are the UNO Q header's I2C pins (`Wire`). AD0 stays unconnected, giving address 0x68.

**Mounting.** This matters, because one calibration has to fit everyone:
- On the **outside of the right ankle**, just above the ankle bone.
- The MPU-6050's **+X axis points up the shin**. Mark the strap with an arrow so every wearer puts it on the same way.
- Strap it tight. A loose sensor bounces and adds fake peaks.
- Part 2 is trained to cope with about ±20° of tilt. Part 1's direction is fixed, so a big orientation change (e.g. the board upside down) will break Part 1.

## 2. How it fits together

```
 [MPU-6050] --I2C 200 Hz--> [sketch.ino]  packs 50 Hz rows, scrolls the LED matrix
                                 |  Bridge RPC: get_samples() / set_display(word, mode)
                            [python/main.py]  windows -> PART 1 + PART 2 (classifiers.py) -> smoothing
                                 |  MQTT (test.mosquitto.org)
          ME193/hudson/hw6/state  (decisions, ~4/s)      ---> [gui.py on the laptop]
          ME193/hudson/hw6/raw    (sample rows, 5 batches/s) ---> (live plot + calibration recordings)
          ME193/hudson/hw6/cmd    <--- "mode:1" / "mode:2" (retained)
```

- **All classifying happens on the board.** The LED matrix keeps working with the laptop off.
- Both parts run on every window. The **mode** only decides which part's answer the LED matrix shows.
- The GUI shows both parts side by side and highlights the active one.

## 3. Files

| Path | What it is |
|---|---|
| `README.md` | this file |
| `HOW_IT_DECIDES.md` | in-depth description of the Part 1 and Part 2 decisions |
| `PLAN.md` / `PROGRESS.md` | the plan and a dated progress log |
| **Board app** `ankle-activity-wearable/` | deployed to `/home/arduino/ArduinoApps/ankle-activity-wearable` |
| `…/app.yaml` | App Lab metadata |
| `…/sketch/sketch.ino` | MPU-6050 at 200 Hz (±16 g, ±2000 °/s), packs 50 Hz rows, 5x7 font, scrolling text |
| `…/python/main.py` | Bridge polling, 2 s windows, both parts, smoothing, MQTT, mode switching |
| `…/python/classifiers.py` | **the decision code**: `SHARED`, `PART 1` and `PART 2` sections. The laptop scripts import it too. |
| `…/python/part1_calibration.json` | the Part 1 calibration in use: direction, activity order, 3 thresholds |
| `…/python/part2_model.json` | the Part 2 calibration in use: network weights + input scales |
| **Laptop** | |
| `gui.py` | the combined GUI: **Live** tab + **Calibration** tab; Shift+C opens the wizard |
| `calibration_wizard.py` | the step-by-step calibration window (record 4 activities → fit → compare → install + deploy) |
| `calibration.py` | installing a calibration, the history, restoring, running `deploy.py` from the GUI |
| `calibration_history/<time>/` | every installed calibration (`installed/`) and the one it replaced (`replaced/`) |
| `calibrate_part1.py` | **PART 1** fit: tries all 6 directions, keeps the best one + thresholds (the wizard calls it) |
| `train_part2.py` | **PART 2** fit: trains the CNN with augmentation, validates, exports JSON (the wizard calls it) |
| `replay.py` | runs recordings through both parts like the board does; confusion matrices |
| `make_model_figures.py` | draws the installed models as `figures/part1_model.jpg` and `figures/part2_model.jpg` (rerun after recalibrating) |
| `dataset.py` | saving/loading recordings, windowing, confusion-matrix printing (shared) |
| `config.py` | topics, paths, GUI settings; also makes `classifiers.py` importable on the laptop |
| `mqttlib.py` | MQTT wrapper (copied from HW5) |
| `data/<person>/<activity>_<time>.csv` | the labeled recordings |
| **Testing** `testing/` | |
| `testing/synth.py` | synthetic ankle IMU + a Python copy of the sketch's packing |
| `testing/make_synthetic_data.py` | writes fake recordings for 3 fake people to `testing/data_synthetic/` |
| `testing/fake_board.py` | runs the real `main.py` on the laptop with a simulated sketch (GUI works against it). `--control-file` lets a script choose the activity. |

Where the Part 1 / Part 2 code lives (combined, but clearly marked):
- `classifiers.py`: `# ===== PART 1 =====` / `# ===== PART 2 =====` sections.
- `main.py`: `# ---- PART 1` / `# ---- PART 2` lines in `Wearable`.
- `sketch.ino`: says which row values each part uses.
- Laptop: `calibrate_part1.py` is Part 1 only, `train_part2.py` is Part 2 only.
- GUI: separate Part 1 / Part 2 panels; the wizard fits and compares each part separately.

## 4. Running it

All commands run from the repo root. Use `.venv/Scripts/python` for the laptop scripts. It has numpy, matplotlib, torch and paho-mqtt.

**Deploy the board app** after any code change. The calibration wizard deploys by itself after installing a calibration.

    python tools/deploy.py ankle-activity-wearable

**Start the GUI:**

    .venv/Scripts/python HW6/gui.py

**Live tab:**
- The big colored banner is the activity the board decided, from the active part.
- The **Part 1 / Part 2** buttons switch modes. The switch is sent as a retained MQTT message, so the board keeps the mode after a restart. "board is running Part N" confirms the board received it.
- Part 1 panel: the chosen direction, the current 2 s peak, the window's result, the smoothed result, and the threshold table.
- Part 2 panel: the network's class probabilities, the window's result, and the smoothed result.
- Plot, top: the Part 1 input signal with its threshold lines. Bottom: accel and gyro magnitudes.

**Calibration tab:**
- The calibration in use: when it was made, the Part 1 direction and thresholds, both parts' accuracy, and whose data it came from.
- Whether **the board runs that same calibration** (it reports which one it loaded). If not, press **Deploy to board**.
- How many seconds of each activity are recorded, per person.
- The **calibration history**. Select an entry and press **Restore selected** to go back to an older calibration. The one in use is kept in the history too.

**LED matrix:**
- It scrolls `STAND`, `WALK`, `RUN` or `BIKE`.
- The bottom-left corner shows the mode: 1 dot = Part 1, 2 dots = Part 2.
- Other messages:

| Matrix shows | Meaning |
|---|---|
| `HW6` | just started; fewer than 3 matching windows so far (~2.5 s) |
| `NO CAL` / `NO MODEL` | no calibration on the board yet: press Shift+C in the GUI (section 5) |
| `NO IMU` | MPU-6050 not answering: check the wiring |
| `WAIT` | the Python app isn't talking to the sketch (app stopped or crashed) |

## 5. Calibration (Shift+C in the GUI)

Calibration is done once. The result is saved in the app and deployed, so nobody needs to recalibrate. To redo it later (e.g. to add another person's data), run the same wizard again.

**Start it:**
- Press **Shift+C** (a capital C) anywhere in the main GUI window, or use the **Start calibration** button on the Calibration tab.
- A capital C typed into a text box doesn't count, so typing a name like "Chris" won't open it.
- Pressing it while the wizard is open just brings the wizard to the front.

**The wizard's steps** (7 when all four activities are recorded, fewer when only some are):

| Step | What happens |
|---|---|
| 1. Setup | who is wearing it, seconds per activity (default 60), countdown before each recording (default 8 s, enough time to lie down for biking), **which activities to record this time**, and whether to **also use earlier recordings** (on by default; more people → works better on someone new). Next to each activity it shows how many seconds that person and everyone have already recorded. It also shows the calibration in use now, a mounting reminder, and whether the board is online. |
| Record each ticked activity | each step says what to do (e.g. "change speed partway through"). **Start recording** → countdown → beep → recording with a progress bar → beep at the end. **Stop early and save** and **Record again** are there if a take goes wrong. Next unlocks once the activity is recorded. |
| 6. Fit | runs the Part 1 calibration (all 6 directions, best one + thresholds), then trains the Part 2 network (a few minutes). The full output, including confusion matrices, scrolls by. |
| 7. Results | **new vs in use, side by side**: Part 1 direction, order, thresholds and accuracy; Part 2 accuracy; how each was validated; whose data. Then: **Install and deploy to the board**, or **Discard (keep the current calibration)**. |

**Recording only some activities** (e.g. just biking, after standing, walking and running worked in an earlier session):
- In Setup, untick the activities you don't want to record again. The wizard then only has steps for the ticked ones.
- The unticked activities come from the earlier recordings in `HW6/data`, so "also use earlier recordings" is forced on. If an unticked activity has no earlier recording at all, the wizard says so and won't continue until you tick it.
- "The new takes replace this person's earlier takes of the same activities" (on by default): when you re-record an activity, the fit leaves out that person's older takes of it. The old files stay in `HW6/data`; they're just not used. Untick it to add the new takes to the old ones instead.
- The fit log lists every recording used, marks the ones from this session, and lists any that were left out.
- **Untick all four** to refit from the existing recordings without recording anything: the wizard goes setup → fit → results. Use this after a change to how the calibration is computed.

**What install does:**
1. Copies the calibration in use (if any) to `calibration_history/<time>/replaced/`.
2. Writes the new files into the app and copies them to `calibration_history/<time>/installed/`.
3. Runs `tools/deploy.py` and shows its output.

If the deploy fails (e.g. the board is off), the new calibration is still installed on the laptop. Use **Retry deploy**, or **Deploy to board** on the Calibration tab later. That tab shows whether the board runs the installed calibration.

**What "doesn't forget" means here:**
- The calibration in use changes **only** when you press Install.
- Closing the wizard, Cancel, Discard, a failed fit or an interrupted recording all leave it untouched.
- On cancel, the wizard asks whether to keep the session's recordings in `HW6/data` (they are good data for a future calibration) or delete them.
- Nothing is ever deleted from `calibration_history/`, so the first calibration can always be restored from the Calibration tab.
- `calibrate_part1.py` / `train_part2.py` on the command line only write to an `--out` file you name, so they can't replace the calibration in use either.

**Tips for a calibration that works on other people:**
- Mix speeds within each recording.
- Run the wizard once per extra person, with "also use earlier recordings" on. With 2+ people, both parts are validated by **holding out a whole person**, which is the honest test of "works on someone new". With one person, the accuracies are optimistic.
- Commit `data/`, `calibration_history/` and the two app JSON files.
- For a final check, record a new session (or a new person) and run:

      .venv/Scripts/python HW6/replay.py --data HW6/data/<person>

## 6. Testing without the board

Everything except the real sensor can be checked on the laptop:

    .venv/Scripts/python HW6/testing/make_synthetic_data.py
    .venv/Scripts/python HW6/calibrate_part1.py --data HW6/testing/data_synthetic --out <scratch>/p1.json
    .venv/Scripts/python HW6/train_part2.py     --data HW6/testing/data_synthetic --out <scratch>/p2.json
    .venv/Scripts/python HW6/replay.py          --data HW6/testing/data_synthetic --part1 <scratch>/p1.json --part2 <scratch>/p2.json
    .venv/Scripts/python HW6/testing/fake_board.py --part1 <scratch>/p1.json --part2 <scratch>/p2.json
    .venv/Scripts/python HW6/gui.py          # in a second terminal, against the fake board

- `fake_board.py` runs the **real** `main.py`, swapping only `arduino.app_utils` for a simulated sketch.
- The simulated sketch packs samples with the same encoding as `sketch.ino`. That way the decoding, windowing, both classifiers, MQTT and the GUI are all exercised.
- `--control-file <file>` makes it simulate whatever activity name is written in that file. A scripted test of the wizard used this to switch activity per recording step.
- If the real board app is running at the same time, give the fake board its own topics with `--topic-prefix ME193/hudson/hw6test`, and point the GUI at the same prefix (set `config.TOPIC_*` in a test script). Otherwise both boards' samples mix into one recording.
- Don't press Install in the wizard while testing against the fake board with synthetic data. That would install a calibration fitted to fake signals. (If it happens, restore the previous one from the Calibration tab.)

## 7. Protocols (for reference)

**Bridge, `get_samples()`:**
- Returns `"E"` if the IMU is down.
- Otherwise: 3 characters for the sequence number of the first row, then up to 6 rows of 12 values.
- Each value is `int16 + 32768`, written as 3 base64 characters, so a reply is at most 219 characters. That fits the 256-byte RPC buffer.
- Row values (raw counts): `ax ay az gx gy gz` averaged over 4 readings at 200 Hz, then `ax ay az` max, then `ax ay az` min.
- Scale: 2048 counts/g, 16.4 counts/(°/s).

**Bridge, `set_display(word, mode)`:** scrolls `word` (A–Z, 0–9, space, `-`). `mode` sets the corner dots.

**MQTT messages:**
- `ME193/hudson/hw6/state` (JSON, after every decision):

      {"t", "imu_ok", "mode", "state", "rows_total", "dropped",
       "part1": {"state", "raw", "peak", "direction", "order", "thresholds", "created"},
       "part2": {"state", "raw", "classes", "probs", "created"}}

  `part1`/`part2` is `null` if its file is missing. `created` identifies which calibration the board loaded.
- `ME193/hudson/hw6/raw` (JSON, 5×/s): `{"seq": <first row's number>, "rows": [[12 values in g and °/s], ...]}`
- `ME193/hudson/hw6/cmd`: `mode:1` or `mode:2`, retained.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `deploy.py` can't reach the board | The board's IP changed: update `HOSTS` in `tools/board_secrets.py`. |
| App Lab library/import errors | Check the board clock: `python tools/deploy.py --cmd "date"`. A wrong date has broken builds before. |
| GUI says "board offline" | Is the app running? Is the board on WiFi? Check the deploy output for Python errors. |
| A wizard recording says "too short" | Hardly any samples arrived during it: the board was offline or the app wasn't running. Fix that, then Record again. |
| `dropped` count keeps rising | The Python app falls behind the sketch (busy board, or WiFi stalls inside `Bridge.call`). Occasional drops are harmless; gaps over 10 rows restart the window. |
| One part keeps confusing two activities | Recalibrate (Shift+C) with more varied takes or another person, with "also use earlier recordings" on. For Part 1, the fit log's direction table shows whether another direction separates them better. |
| A new calibration turned out worse | Calibration tab → select the older entry in the history → Restore selected. |
