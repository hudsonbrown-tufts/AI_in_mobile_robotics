# HW7: Ankle Step Counter with Uncertainty

An UNO Q with an MPU-6050 on the right ankle counts Hudson's steps and estimates the distance walked, each with a live uncertainty: a ± range, a 95% interval, a "how sure" number, and a breakdown of where the uncertainty comes from. The counting is done by an ensemble of 5 small neural networks (1D CNNs like HW6-NN), trained with PyTorch by backpropagation and run on the board with numpy.

- **How the network works** (and the animation storyboard): [HOW_THE_NET_WORKS.md](HOW_THE_NET_WORKS.md)
- **Plan / log:** [PLAN.md](PLAN.md), [PROGRESS.md](PROGRESS.md)

## 1. Hardware
Same as HW6: GY-521 (MPU-6050) on the UNO Q's I2C header (VCC → 3.3V, GND → GND, SDA → SDA, SCL → SCL), strapped to the outside of the right ankle in the same position as HW6, power bank, WiFi. The MQTT broker is the Mosquitto container on the board set up in HW6 (see HW6/README.md section 7).

## 2. How it fits together
```
[MPU-6050] --I2C--> [sketch: 200 Hz -> 50 Hz rows, LED matrix text]        (copied from HW6)
                          | Bridge: get_samples() / set_display(text, 0)
                    [board python: main.py + stepnet.py]
                      every 0.24 s: last 2 s -> 5 networks -> rate ± , speed ±
                      StepAccumulator -> steps ± , distance ± , breakdown
                          | MQTT ME193/hudson/hw7/state  (about 4/s)  -> GUI
                          |      ME193/hudson/hw7/raw    (rows, 5/s)  -> GUI plot + recordings
                          | <-   ME193/hudson/hw7/cmd    "reset"
[laptop: gui.py]  Live tab (count ± , plots)  |  Record and train tab (walks -> train -> install + deploy)
```
The counting runs on the board, so the matrix keeps counting with the GUI closed. The count starts at 0 whenever the app starts (it isn't saved).

## 3. Files
| Path | Purpose |
|---|---|
| `step-counter-wearable/` | the UNO Q app (`app.yaml`) |
| `.../sketch/sketch.ino` | HW6's sketch, unchanged except comments and the start-up word |
| `.../python/main.py` | Bridge polling, windows, the ensemble, MQTT, the matrix |
| `.../python/stepnet.py` | the numpy network, the ensemble, `Windower`, `StepAccumulator` (shared with the laptop) |
| `.../python/stepnet_model.json` | the trained ensemble + calibration (made by training; missing until then → the matrix shows `NO MODEL`) |
| `gui.py` | the laptop GUI |
| `dataset.py` | saving/loading recordings (`data/hudson/<kind>_<time>.csv` + `.json`) |
| `labels.py` | stride times from typed step counts (+ check plots in `figures/labels/`) |
| `train_stepnet.py` | PyTorch training, leave-one-walk-out check, uncertainty calibration, export |
| `replay.py` | runs recordings through the board code: count ± vs. truth, plus a plain peak counter |
| `make_figures.py` | figures for the docs + `figures/animation_data.json` |
| `testing/synth.py`, `testing/fake_board.py` | synthetic walks with known strides; the real `main.py` with a fake sketch |
| `model_history/` | every installed model (and the one it replaced) |
| `training_report.json` | everything the last installed training measured |

## 4. Running it
```
python tools/deploy.py step-counter-wearable          # deploy (stops HW6's app: one app at a time)
.venv/Scripts/python HW7/gui.py                       # laptop GUI
```
Press **Reset count to 0** on the Live tab before a walk.

## 5. Recording and training (about 10 minutes of walking)
All on the GUI's **Record and train** tab:
1. Measure the hallway with a tape measure, type its length, press Save.
2. **Hallway walks, ×6** (2 slow, 2 normal, 2 fast). For each one:
   - Press **Record hallway walk**, then wait for the beep after the 3 s countdown.
   - Stand still at the start line.
   - Walk to the end, turn around and walk back to the start line, **counting every step**.
   - Stop and stand still. Recording stops by itself after 3 s of standing still.
   - Type the count.

   "lengths per hallway walk" = 2 means there and back. The turn is on purpose: the network must see turns.
3. **Long walk, ×1** (3–5 min): **Record long walk (test)**. Walk back and forth, counting steps and hallway lengths, then type both. It is never trained on: it's the final test of the count and its ±.
4. Optional: **Record standing still** (HW6's standing recordings are already used).
5. After each walk, the GUI compares your count with an independent automatic count. If they differ by more than 2 steps (or 5%), it offers to delete the walk so you can redo it.
6. Press **Train** (about 2 min on this laptop; it trains 18 + 5 networks in parallel). Read the log:
   - the stride labels (every walk should say `ok`);
   - the held-out walk table;
   - the reliability table;
   - the gradient check;
   - the long-walk test.
7. **Install and deploy**, or **Discard**. The old model is kept in `model_history/`.
8. Then run `.venv/Scripts/python HW7/make_figures.py` to redraw the figures from the real model, and look at the label plots: `.venv/Scripts/python HW7/labels.py`.

From the command line: `.venv/Scripts/python HW7/train_stepnet.py --install` (or `--out file.json` to try without installing).

## 6. Testing without the board
```
.venv/Scripts/python HW7/testing/synth.py                       # synthetic walks -> testing/data_synthetic/
.venv/Scripts/python HW7/train_stepnet.py --data HW7/testing/data_synthetic --no-hw6 --out m.json
.venv/Scripts/python HW7/replay.py --data HW7/testing/data_synthetic --no-hw6 --model m.json --plot
.venv/Scripts/python HW7/make_figures.py --data HW7/testing/data_synthetic --no-hw6 --model m.json --report m_report.json --out figs
.venv/Scripts/python HW7/testing/fake_board.py --model m.json --topic-prefix test/hw7   # needs a reachable broker
```
`fake_board.py` prints the TRUE step count every 10 s, to compare with the GUI.

## 7. MQTT messages (for reference)
`ME193/hudson/hw7/state` (JSON, after every window):

| Field | Meaning |
|---|---|
| `steps`, `steps_shown` | count (float / rounded) |
| `steps_sd`, `steps_lo`, `steps_hi` | 1σ and the 95% range |
| `sure`, `sure_tol` | chance the truth is within ±`sure_tol` steps |
| `steps_sd_parts` | `random`, `systematic`, `rounding` (1σ each, steps) |
| `dist_m`, `dist_sd`, `dist_lo`, `dist_hi`, `dist_sd_parts` | the same for distance (m) |
| `cadence_spm`, `cadence_sd`, `speed`, `speed_sd` | the current window |
| `epistemic`, `aleatoric`, `members_rate`, `unfamiliar` | the 5 networks' disagreement and the light |
| `walk_s`, `rows_total`, `dropped`, `decision_ms`, `model.created`, `imu_ok` | bookkeeping |

`ME193/hudson/hw7/raw`: `{"seq", "rows": [[12 values], ...]}`, the same as HW6.
`ME193/hudson/hw7/cmd`: `reset`.

## 8. Troubleshooting
- **deploy.py can't reach the board:** its IP changed; update `HOSTS` in `tools/board_secrets.py`.
- **Link errors when the app builds:** see HW6/PROGRESS.md (empty `.o` files in `.cache/sketch`; `deploy.py` now removes them) and check the board's clock (`python tools/deploy.py --cmd "date"`).
- **`NO MODEL` on the matrix:** train and Install and deploy.
- **The count goes up while standing still:** the standing gate (18 steps/min) should prevent it. If it doesn't, record a standing recording and retrain.
- **A walk's labels say TOO FAR OFF:** the typed count and the gyro peaks disagree by more than 1 stride. Delete that walk and redo it.
