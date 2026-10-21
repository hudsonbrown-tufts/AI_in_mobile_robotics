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

## Next
1. Power the board / fix `HOSTS`, then deploy: `python tools/deploy.py ankle-activity-wearable`. Check the deploy output: numpy install, rows/s, dropped count.
2. Check that the matrix scrolls `NO CAL` and that the GUI shows "board online" and the live plot.
3. In the GUI press Shift+C and follow the wizard (about 60 s per activity, plus other people if possible). Then commit data/, calibration_history/ and the two app JSON files.
