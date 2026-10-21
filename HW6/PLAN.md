# HW6: Ankle Activity Wearable (UNO Q + MPU-6050)

**Status:** plan v3 (2026-10-21): coded and tested offline. See PROGRESS.md, README.md and HOW_IT_DECIDES.md.

## Goal
- An UNO Q with an MPU-6050 is strapped to the ankle. It classifies **standing / walking / running / biking**. Biking means lying on your back and pedaling in the air.
- **Part 1:** uses only the **peak acceleration along one fixed axis**.
- **Part 2:** uses **all IMU outputs** (3-axis accel and 3-axis gyro) with a **neural network**.
- The **LED matrix types out** the current activity. A laptop GUI shows the state and switches between Part 1 and Part 2.
- **Calibration and training happen once.** The results are saved in the app and committed to git. The device must work on other people without recalibrating.

## Decisions (confirmed)
| Question | Answer |
|---|---|
| Code split | One combined app. Every Part 1 / Part 2 section is marked with a clear `PART 1` / `PART 2` banner comment. |
| Part 1 direction | Calibration tests all 3 axes (both signs) and saves the best one. The runtime only ever uses that one direction. |
| Calibration | One time only, saved to files in the app, no per-user calibration |
| Calibration UI | Inside the GUI: Shift+C (a capital C) opens a step-by-step wizard (record 4 activities → fit both parts → compare → install + deploy). The calibration in use only changes on Install; cancelling keeps it. Every install is kept in `calibration_history/` and can be restored. |
| Part 2 model | small neural network (1D CNN), trained on the laptop with PyTorch |
| LEDs | LED matrix scrolls the word (STAND / WALK / RUN / BIKE) |
| Power / link | power bank on the ankle, WiFi + MQTT |

## Hardware and data flow
```
[MPU-6050]--I2C-->[sketch: 100 Hz sampling, sample buffer, scrolling text]
                         |  Bridge: get_samples() / set_text(word)
                  [board python: main.py + classifiers.py]
                    |  MQTT  ME193/hudson/hw6/state  -> GUI  (state, mode, confidence, features)
                    |        ME193/hudson/hw6/raw    -> GUI  (sample batches, only while recording)
                    |  <-    ME193/hudson/hw6/cmd           (mode:1|2, record:on|off)
[Laptop: HW6/gui.py]  tkinter: state display, Part 1/2 switch, live plot, data-collection tab
```
Classification runs on the board, so the matrix keeps working with the GUI closed.

## File layout
| Path | Purpose |
|---|---|
| `PLAN.md` / `PROGRESS.md` | this plan / dated progress log |
| `config.py` | laptop settings: topics, state names, window size, the Part 1 axis |
| `mqttlib.py` | copy of the HW5 wrapper |
| `gui.py` | combined GUI: Live tab (state, mode switch, plot) + Data collection tab (one-time recording) |
| `calibrate_part1.py` | **PART 1**, one time: recordings -> 3 thresholds -> `part1_calibration.json` |
| `train_part2.py` | **PART 2**, one time: recordings -> augmented windows -> CNN -> `part2_model.json` |
| `replay.py` | runs recordings through both parts offline: accuracy + confusion matrices |
| `data/<person>/<label>_<time>.csv` | labeled recordings (t, ax, ay, az, gx, gy, gz) |
| `ankle-activity-wearable/app.yaml` | UNO Q app (unique folder name) |
| `.../sketch/sketch.ino` | 100 Hz IMU, ±16 g / ±2000 °/s, batch Bridge call, 5x7 font + scrolling text |
| `.../python/main.py` | shared: Bridge polling, windows, mode switching, smoothing, MQTT |
| `.../python/classifiers.py` | `# ===== PART 1 =====` peak classifier and `# ===== PART 2 =====` CNN forward pass (numpy) |
| `.../python/part1_calibration.json` | the saved Part 1 thresholds |
| `.../python/part2_model.json` | the saved Part 2 network weights + input normalization |
| `.../python/requirements.txt` | `paho-mqtt`, `numpy` |

## Shared pipeline (board)
1. The sketch samples at 100 Hz into a buffer. Python calls `get_samples()` about 10×/s and gets all new samples packed as one string.
2. Sliding window of **2.0 s**, updated every **0.25 s**.
3. The active part classifies the window.
4. **Smoothing:** the state changes only after 3 windows in a row agree. For Part 2, a window also needs network confidence ≥ 0.6 to count.
5. `set_text("WALK")` on the sketch, then publish JSON on `.../state`.

## Mounting (the same for everyone)
- The board and IMU sit on the **outside of the ankle**, with the IMU's **+X axis pointing up the shin**. An arrow on the strap shows which way.
- A fixed orientation is what lets one calibration work for everyone.
- Part 2 is also trained to cope with small mounting differences (see augmentation).

## Part 1: peak acceleration, one fixed axis
- **Direction:** chosen once by `calibrate_part1.py` from +x, -x, +y, -y, +z, -z and saved in `part1_calibration.json`. The Part 1 code only ever gets that one direction's accel samples, never gyro or other axes.
- **Feature:** `peak = max(a_x)` over the 2 s window.
- **Expected ordering:**
  - Biking, lying down: little gravity on the axis, peak about 0.3–0.9 g.
  - Standing: about 1.0 g.
  - Walking: about 1.5–3 g.
  - Running: about 3.5–8+ g.
- **Rule:** 3 thresholds give 4 bins.
- **One-time calibration (`calibrate_part1.py`):**
  - Uses every recording in `data/`. The more people, the better.
  - Places each threshold at the **midpoint between neighboring classes** in log scale. Log scale matters because spread grows with intensity: walking is tighter than running.
  - Uses each class's 5th/95th percentile, not its mean. This leaves margin for people who walk harder or run softer than the recorded people.
  - Writes `part1_calibration.json` and prints how many windows each threshold gets wrong.
- Direction scoring: leave-one-person-out balanced accuracy (all data if only 1 person). Ties are broken by the worst normalized gap between neighbouring activities.

## Part 2: neural network on all six channels
### Input
- A 2 s window downsampled to 50 Hz: **100 time steps × 6 channels** (ax, ay, az in g; gx, gy, gz in °/s).
- Each channel is scaled by fixed constants saved in the model file (mean/std from training). Nothing is tuned per user.

### Network (1D CNN, about 6k weights)
```
Conv1d(6 -> 16, k=7) + ReLU + MaxPool(2)
Conv1d(16 -> 32, k=5) + ReLU + MaxPool(2)
Conv1d(32 -> 32, k=5) + ReLU
Global average pool -> Dropout(0.3) -> Linear(32 -> 4) -> softmax
```
- The convolutions learn stride and pedal shapes on their own, with no hand-made features.
- Global pooling makes the result independent of where in the window a step lands.

### Training (`train_part2.py`, laptop, PyTorch, which is already in `.venv`)
- **Windows:** 2 s windows every 0.25 s from every recording.
- **Augmentation**, so the model works on new people:
  - **Small random rotations (±20°):** mounting angle differs between people.
  - **Amplitude scaling (×0.8–1.25):** some people step harder or softer.
  - **Time stretching (×0.8–1.25):** different cadence and pace.
  - **Small noise.**
- **Validation:** with ≥ 2 people recorded, hold one person out and test on them. This is the test that matters for "works on other people". With one person, hold out whole recordings, not random windows. Random windows overlap, which would make the score look better than it is.
- Adam, cross-entropy, early stopping on validation loss. It prints a confusion matrix.
- **Export:** `part2_model.json` holds the weights + normalization + class names. `classifiers.py` runs the same network in about 40 lines of numpy. There's no PyTorch on the board.
- **Check:** `train_part2.py` confirms the numpy forward pass matches PyTorch on test windows (difference < 1e-4) before saving.

### Data to collect (one time)
- About **60 s per activity** from you. Ideally also 1–2 other people for about 30 s each.
- Mix speeds: slow/fast walk, jog/sprint, slow/fast pedaling. Include fidgeting while standing.

## LED matrix (types out the state)
- Built-in 5x7 font for the needed letters. `STAND`, `WALK`, `RUN`, `BIKE` scroll right-to-left across the 13x8 matrix in a loop, about 12 columns/s.
- A new state restarts the scroll at once.
- If the MPU-6050 isn't answering, it scrolls `NO IMU`.
- A custom font in the sketch avoids depending on text support in the UNO Q's matrix library.

## GUI (`gui.py`, laptop, tkinter)
- **Live tab:**
  - Big state name, plus the active mode.
  - **Part 1 / Part 2** buttons send `mode:N`. The board confirms in its next state message.
  - Live plot. Part 1: `a_x`, the window peak, and the 3 threshold lines. Part 2: accel/gyro magnitudes and a bar of the 4 class probabilities.
  - A "board offline" banner if the board goes quiet for 2 s.
- **Data collection tab (one-time use):** person name + activity label + Record/Stop. It saves raw data to `data/<person>/`. Calibration and training run afterwards from the command line. They're not a GUI or runtime step.

## Build order
1. Sketch with sampling, the batch Bridge call and scrolling text. Deploy and check the sample rate and how much fits per Bridge reply. This is the riskiest unknown.
2. `main.py` with MQTT, mode switching and the raw stream. GUI Live + Data collection tabs.
3. Record data. `calibrate_part1.py`, Part 1 classifier, replay test, live test.
4. `train_part2.py`, numpy CNN, replay test, live test, test on another person.
5. Write PROGRESS.md and commit the calibration/model files.
