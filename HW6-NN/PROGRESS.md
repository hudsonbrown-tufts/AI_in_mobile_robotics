# HW6 progress log

## 2026-10-21: everything coded and tested offline (board unreachable)
- **Code written:**
  - Board app `ankle-activity-wearable/`: sketch, `main.py`, `classifiers.py`.
  - Laptop: `gui.py`, `calibrate_part1.py`, `train_part2.py`, `replay.py`, `dataset.py`, `config.py`.
  - Test tools: `testing/` (synthetic data, fake board).
- **Design changes from plan v2** (asked for by Hudson):
  - One combined app with clearly marked PART 1 / PART 2 sections, instead of separate folders.
  - Part 1 calibration tries all 3 axes (both signs: 6 directions), picks the best one and saves it. Runtime uses only that one direction.
  - Part 2 uses a neural network (1D CNN) instead of a decision tree.
  - The LED matrix types out the word (custom 5x7 font, scrolling).
- **Verified offline:**
  - The sketch compiles with App Lab's arduino-cli (`arduino:zephyr:unoq`): 97.5 KB flash (12%), 43.5 KB RAM (16%), no warnings.
  - The sketch font renders correctly (checked as ASCII art from the sketch's tables).
  - Bridge reply size: at most 219 characters, under the 256-byte RPC buffer (checked in RPClite 0.3.1 source). `provide_safe` handlers run from `__loopHook` after every `loop()`.
  - Synthetic data, 3 fake people:
    - `calibrate_part1.py` ranks all 6 directions; leave-one-person-out works.
    - `train_part2.py` trains, validates leave-one-person-out, and the numpy forward pass matches PyTorch to 1.7e-8.
    - `replay.py` works.
  - `fake_board.py` runs the real `main.py`:
    - about 50 rows/s, 0 dropped, about 4 decisions/s;
    - the GUI mode switch round-trips over MQTT;
    - `NO CAL` shows when the files are missing.
  - GUI: screenshots of both tabs look right; a 10 s recording saved and loaded back (7.1 s usable after trimming).
- **Not yet done:**
  - Running on the real board (it was unreachable at both addresses in `HOSTS`).
  - Real recordings, the real `part1_calibration.json` / `part2_model.json`, and live tests.

## 2026-10-21 (later): calibration moved into the GUI
- **Shift+C** (a capital C, ignored when it's typed into a text box) opens `calibration_wizard.py`.
  - Steps: setup → record standing / walking / running / biking (countdown, beeps, progress bar, redo) → fit Part 1 + train Part 2 in the background → compare new vs in use → **Install and deploy** or **Discard**.
- `calibration.py`:
  - install saves the replaced calibration and the new one to `calibration_history/<time>/`;
  - restore from history;
  - runs `tools/deploy.py` from the GUI.
- The old "Data collection" tab is replaced by a **Calibration** tab. It shows: the calibration in use; whether the board runs it (the board now reports each part's `created` time); recordings per person; history with Restore.
- `calibrate_part1.py` / `train_part2.py` are now functions the wizard calls. On the command line they require `--out`, so only the wizard can replace the calibration in use.
- **Tested** with a scripted run against `fake_board.py --control-file`, using scratch folders and a faked deploy:
  - Shift+C opens the wizard, and a capital C in a text box doesn't;
  - all 4 recordings saved;
  - the fit ran;
  - the results table was filled in;
  - Install wrote both files plus a history entry;
  - the Calibration tab shows the history and the board mismatch.
- **Problem along the way:** a script that rewrote README.md used Windows' default cp1252 encoding and emptied the file. The README was rewritten in full. All other HW6 files were checked: valid UTF-8, no garbled characters.

## 2026-10-23: first run on the real board
- The board was reachable at 192.168.1.137 (a second WiFi network), so that address was added to `HOSTS` in `tools/board_secrets.py`. The old address is kept.
- Hudson made an App Lab app `activity-tracker` ("activity Tracker") with copies of the repo's sketch/main.py/classifiers.py. The copies are identical to the repo's, but it had no `requirements.txt`.
- **Problem:** the build failed at the link step (undefined `arduino::String`, `operator new`, `main`).
  - **Cause:** 21 of 33 files in its `.cache/sketch` build cache were 0 bytes (`core/main.cpp.o`, `core/api/String.cpp.o`, `core/new.cpp.o`, ...), all written at 14:42. An interrupted build left them, and the next build reused them. The board clock was correct.
  - **Fix:** stopped the app, deleted `.cache/sketch`, added `python/requirements.txt` (paho-mqtt, numpy), and started it again.
- **Result:**
  - The sketch compiled and flashed; numpy 2.5.4 and paho-mqtt 2.1.0 installed.
  - MQTT connected, rows arrive at about 50/s with 0 dropped.
  - Both parts report no calibration yet, as expected.

- **Same problem in `ankle-activity-wearable`:** deploying the repo app from VS Code failed to link (`Bridge`, `Wire` undefined). Its cache had 3 empty `.o` files from a build cut off at 14:53.
  - **Durable fix:** `tools/deploy.py` now deletes 0-byte `.o` files in the app's `.cache/sketch` before every `app start`, and prints them.
  - After redeploying, the build succeeded and the app runs: about 50 rows/s, 0 dropped. Deploying stopped `activity-tracker`, since the board runs one app at a time.

## 2026-10-24: calibrate only some activities
- Hudson recorded standing, walking and running (`data/hudson/`, about 57 s usable each, clean sequence numbers), but not biking. Nothing is installed yet.
- **Wizard setup** now has "Activities to record this time" tick boxes, each showing seconds already recorded (this person / everyone).
  - Only the ticked activities get recording steps, and the step count adjusts.
  - Unticked activities come from earlier recordings, so "use earlier recordings" is forced on.
  - Setup won't continue if an unticked activity has no earlier recording.
- **New option, on by default:** "the new takes replace this person's earlier takes of the same activities". The fit leaves those older files out without deleting them. The fit log lists every recording used and every one left out.
- **Tested** with a scripted run on copies of Hudson's three real recordings plus a 20 s fake-board biking take:
  - with no earlier data, biking-only is refused with a clear note;
  - with the earlier data, the steps are setup → biking → fit → results;
  - the fit used all 4 activities;
  - the results showed Part 1 direction +x, order stand < bike < walk < run;
  - nothing was installed.
- **Found during testing:** the fake board and the real board both published on `ME193/hudson/hw6/*`, so a test recording got both streams interleaved. `fake_board.py --topic-prefix` now keeps tests on their own topics.

## 2026-10-26: model figures, Part 1 threshold buffer
- **Figures:** `make_model_figures.py` draws the installed models to `figures/part1_model.jpg` (flow diagram plus every window's +x peak against T1–T3) and `figures/part2_model.jpg` (layer diagram, a real input window, the output probabilities).
- **Problem the Part 1 figure exposed:** in the installed calibration (Oct 24), standing reads only 0.21 g along +x and its region is 0.18–0.24 g, only 0.06 g wide.
  - **Cause:** the gap-splitting rule shares a gap by each side's std, and standing's std was 0.01 g, so it got almost no room.
  - **Fix:** `calibrate_part1.MIN_SPREAD_G = 0.15` floors every spread. Refitting the same recordings gives T1 = 0.08, T2 = 0.51, T3 = 1.94 g, still 100%.
  - **Installed** on 2026-10-26: the new Part 1 plus the unchanged Part 2 (made 2026-10-24). History is in `calibration_history/2026-10-26_110235/`. Deployed via `HudsonArduino.local` (192.168.0.7); the board's file matches and the app loaded both parts. Figures redrawn.
- **Wizard:** unticking all four activities now means "refit from the existing recordings" (setup → fit → results).
- **Mounting:** the +y direction reads 0.98 g while standing, so the board's shin axis is y, not x. Either the board is mounted rotated 90° from the plan, or the plan should say +Y.
- **Broker:** `test.mosquitto.org` port 1883 has been timing out since about Oct 26, while its port 8883 and other brokers' 1883 work. It's an outage on their side, not a code problem.

## 2026-10-26 (later): thresholds meet in the middle
- The 0.15 g spread floor put T1 (0.08 g) right on biking's highest windows. Hudson asked for a smaller buffer, with each threshold meeting in the middle.
- **New rule** in `calibrate_part1.fit_thresholds`: `T = (L99 + U1) / 2`, the middle of the empty space between the lower activity's 99th percentile and the upper one's 1st. `MIN_SPREAD_G` is removed. A midpoint of the 95th/5th percentiles would have left T1 at 0.085 g, because biking's top 5% of windows reach 0.08 g.
- Result: T1 = 0.13, T2 = 0.57, T3 = 2.42 g (buffers 0.06 / 0.33 / 0.58 g each side), 100% on Hudson's data. Synthetic leave-one-person-out is still 100%.
- **Installed** (history `calibration_history/2026-10-26_110612/`, Part 2 unchanged) and **deployed**. The board's file checksum matches the laptop's and the app loaded both parts. Figures redrawn.

## Next
1. Power the board / fix `HOSTS`, then deploy: `python tools/deploy.py ankle-activity-wearable`. Check the deploy output: numpy install, rows/s, dropped count.
2. Check that the matrix scrolls `NO CAL` and that the GUI shows "board online" and the live plot.
3. In the GUI press Shift+C and follow the wizard (about 60 s per activity, plus other people if possible). Then commit data/, calibration_history/ and the two app JSON files.
