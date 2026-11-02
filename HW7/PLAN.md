# HW7: Ankle Step Counter with Uncertainty (UNO Q + MPU-6050)

**Status:** plan v2 (2026-10-30), coded and tested offline on synthetic walks the same day. See PROGRESS.md for the changes made while building (there-and-back walks, standing gate, bias correction k, parallel training), README.md and HOW_THE_NET_WORKS.md.

## Goal
- Same hardware and mounting as HW6: UNO Q + MPU-6050 strapped to the outside of the right ankle, power bank, WiFi.
- **Count Hudson's steps while walking**, with a **PyTorch neural network** in the style of HW6-NN (small 1D CNN on 2 s windows of all 6 IMU channels), **trained by backpropagation**.
- **Estimate distance walked.**
- For both, a **live uncertainty analysis**: a ± range, a 95% interval, a "how sure" number, and a breakdown of where the uncertainty comes from.
- `HOW_THE_NET_WORKS.md`, detailed enough to build an animation from later.

## Decisions (confirmed 2026-10-30)
| Question | Answer |
|---|---|
| Activities | **Walking only** (plus standing still, which must count 0). No running, no biking. |
| Who | **Hudson only.** No need to work on other people. |
| Ground truth | A hallway Hudson measures. |
| Training effort | **Short**: walk the hallway a few times, then one counted walk of a few minutes (about 10 min total). Training must be easy: guided in the GUI, one button to train + install + deploy. |
| Count after restart | **Not saved.** Restarting the app starts at 0. |
| Uncertainty about the model | **Ensemble of 5** networks. |
| Training method | **Backpropagation** (PyTorch autograd + Adam), see §4.4. |

---

## 1. The core idea
The ankle sensor only feels its own (right) leg, so it sees **strides**: one per right-foot step. Steps = 2 × strides. The network looks at the last 2 s of motion and predicts two things, each as a **mean and a spread**: the **stride rate** (strides/s) and the **walking speed** (m/s). Every 0.24 s the board adds `rate × 0.24 s` to the stride count and `speed × 0.24 s` to the distance. Five independently trained copies of the network run side by side; where they disagree, the model is unsure. The spreads are added up over time with rules that are checked against walks the networks didn't train on, so "± 9 steps, 95%" really means 95%.

---

## 2. Reused from HW6 / HW6-NN
| Piece | Reused from | Change |
|---|---|---|
| `sketch/sketch.ino` (200 Hz IMU, ±16 g / ±2000 °/s, 50 Hz rows, Bridge, 5×7 scrolling font with digits) | HW6 (identical to HW6-NN) | none |
| Bridge polling, row decoding, gap handling in `main.py` | HW6 | no Part 1/2 modes |
| MQTT broker: Mosquitto in Docker on the board | HW6 (README §7) | topics `ME193/hudson/hw7/*` |
| `mqttlib.py`, `config.find_broker()` | HW6 | copied |
| Window 2 s (100 rows), hop 0.24 s (12 rows), scaling accel ÷ 2 g, gyro ÷ 500 °/s | HW6-NN | same |
| CNN trunk | HW6-NN `ActivityCNN` | new output heads (§4) |
| Augmentation (rotation, amplitude, time-stretch, noise) | HW6-NN `train_part2.py` | smaller ranges (one person, one mounting); labels follow the stretch (§3.3) |
| Train in PyTorch on the laptop, export JSON, numpy forward pass on the board, refuse to save unless numpy matches PyTorch to 1e-4 | HW6-NN | per ensemble member |
| `testing/synth.py`, `fake_board.py` | HW6 | synthetic walks with known stride times |
| `HW6/data/hudson/standing_*.csv` | HW6 | extra zero-step examples |
| `HW6/data/hudson/walking_*.csv` (57 s of walking) | HW6 | extra rate-only examples (strides auto-labeled, no distance; §3.2) |

---

## 3. Training data (about 10 minutes of Hudson's time)
### 3.1 The recording session, guided in the GUI's Record tab
1. **Once:** measure the hallway (tape measure, ideally 20 m or more), type the length into the GUI.
2. **Hallway walks, ×6** (about 1 min each with setup): press Start → beep → stand still 3 s → walk from the start mark to the end mark, **counting every step** → stop at the line, stand 3 s → Stop. A box asks "Steps counted?" (distance is filled in from the hallway length). Suggested: 2 slow, 2 normal, 2 fast, so the net sees different cadences and stride lengths. Each saved as `data/hudson/hall_<time>.csv` + `.json` (`steps`, `distance_m`).
3. **Long walk, ×1** (3–5 min): walk the hallway back and forth, counting steps and **counting lengths**. At the end the box asks "Steps counted?" and "Hallway lengths walked?" (distance = lengths × hallway length; turns are short, so the distance is off by a little, recorded as a ±1 m label uncertainty per turn). This one is the **final test**: it is never trained on, and the README reports how the count ± did on it.
4. Press **Train**: labels → trains 5 networks → checks the uncertainty → shows the results (accuracy, coverage table, label plots) → **Install and deploy** or **Discard**. About 2–3 min of laptop time.

The GUI shows the auto-detected count next to the typed count after each walk (only after typing, so it can't bias the counting). If they differ by more than 2 steps, it offers Redo.

### 3.2 Stride timestamps (automatic, `labels.py`)
Per-walk totals alone are weak labels for a network that works on 2 s windows, so each walk is turned into stride **times**:
1. The shin swings forward once per stride: one big clean peak on the gyro axis across the shin (picked automatically as the gyro axis with the most variance while walking).
2. Find those peaks (threshold + at least 0.5 s apart), and **tune the threshold until the count equals `steps / 2`** (±1 when the count is odd: which foot started).
3. If no threshold gets within ±1 stride, the walk is flagged in the results instead of being used.
4. A check plot per walk (gyro trace + marked strides) is saved to `figures/labels/`.
5. The HW6 walking recording has no count, so its peaks are used as-is to label stride **rate** only. Its speed is unknown, so the speed head skips it (masked loss).

### 3.3 What the net is trained to output, per window
- **Stride phase:** a counter that goes up by exactly 1 from one stride time to the next, linear in between. Strides in a window = `phase(end) − phase(start)`, a fractional number. A window that cuts a stride in half gets 0.5, not a jumpy 0 or 1.
- **Stride rate** label = strides in the window ÷ 2.0 s.
- **Speed** label = stride rate × that walk's stride length (`distance_m ÷ strides`).
- The 3 s of standing at each end gives the start/stop windows their true, smaller values, so the net learns starting and stopping too. Standing windows: rate 0, speed 0.
- **Time-stretch augmentation** (signal played 0.85–1.15× as fast) multiplies the rate and speed labels by the same factor, which fills in cadences between the ones recorded.

---

## 4. The network
### 4.1 One member ("StrideNet", about 9k weights)
| Layer | Output | Notes |
|---|---|---|
| input | 6 × 100 | ax ay az ÷ 2 g, gx gy gz ÷ 500 °/s, 2 s |
| Conv1d 6→16, k7, ReLU, MaxPool 2 | 16 × 47 | heel-strike spikes, swing zero crossings |
| Conv1d 16→32, k5, ReLU, MaxPool 2 | 32 × 21 | strike-then-swing combinations, ~0.4 s |
| Conv1d 32→32, k5, ReLU | 32 × 17 | whole-stride patterns, ~0.9 s |
| average over time | 32 | "how much of each stride pattern is in this window" |
| **rate head**: Linear 32→2 | μ_rate, σ_rate | μ = softplus(·) ≥ 0, σ = softplus(·) + 0.01 |
| **speed head**: Linear 32→2 | μ_speed, σ_speed | same |

Both heads share the trunk: cadence and speed come from the same stride shapes, and sharing halves the data needed.

### 4.2 Loss: the net predicts its own error bars
- **Gaussian negative log-likelihood** per head: `log σ + (y − μ)² / (2σ²)`.
  - Wrong with a small σ → big loss. A huge σ everywhere → the `log σ` term makes that cost too. So σ learns to match the size of the net's real mistakes on that kind of window. This is the **aleatoric** uncertainty: "this window is ambiguous" (starting, stopping, a shuffle).
  - β-NLL (β = 0.5) so hard windows still get learned instead of just getting a big σ.
- Loss = rate NLL + speed NLL (speed masked where unknown); standing and walking windows weighted equally.

### 4.3 Ensemble: the net's uncertainty about itself
- 5 members, each with a different random start and a different resample of the training windows.
- On a window: `μ = mean of the 5 μ's`, `σ² = mean of the 5 σ²'s (aleatoric) + variance of the 5 μ's (epistemic)`.
- **Epistemic** = the members disagree = motion unlike the training data (stairs, a stumble, carrying something). The GUI shows it as a live "unfamiliar motion" light.

### 4.4 Training = backpropagation
Yes, training is backpropagation, done by PyTorch:
1. **Forward pass:** a batch of 64 windows goes through the network and gives μ and σ for each.
2. **Loss:** β-NLL against the labels, averaged over the batch.
3. **Backward pass (`loss.backward()`):** PyTorch's autograd applies the chain rule backwards through every layer (softplus → linear head → average → conv → ReLU → max-pool → conv ...) and gets the gradient ∂loss/∂w for every one of the ~9k weights: "how much would the loss change if this weight moved a little".
4. **Update (Adam, lr 0.002):** each weight moves a small step against its gradient.
5. Repeat for every batch; one pass over the data = one epoch. Stop when the loss on held-out walks stops improving (early stopping, patience 20, max 150 epochs).
6. Done 5 times, once per ensemble member.

- The training script also prints a **gradient check** once (PyTorch's gradient for a few weights vs. nudging the weight and measuring the loss change), and the loss curve per member is plotted, both for the docs/animation.
- The board **never** backpropagates: it only runs the forward pass with the frozen weights (numpy, no PyTorch).

---

## 5. From windows to a step count ± and a distance ±
### 5.1 Adding up
- Every hop (H = 0.24 s): strides += μ_rate × H; distance += μ_speed × H. Steps = 2 × strides, shown rounded.
- Each window credits only its newest 0.24 s, so the count lags about 1 s, then catches up.

### 5.2 Adding up the uncertainty: three parts
| Part | Where it comes from | How it grows | Formula (strides) |
|---|---|---|---|
| **random** | per-window σ (aleatoric + ensemble) | like √time: errors partly cancel | hops inside one 2 s window share their error, separate windows don't: `Var_rand ≈ (W/H) · Σ (σ_i · H)²` |
| **systematic** | a whole walk can be off by a few % (today's pace, strap angle) | in proportion to the count: doesn't cancel | `(s · N)²`, `s` = relative total error on held-out walks beyond the random part |
| **rounding** | which foot moved first; the left foot's first/last step is invisible | constant, ±1 step | `1/3` steps² |

`σ_steps = sqrt(4 · (Var_rand + (s·N)²) + 1/3)`. Distance works the same way with its own `s_dist`, plus the hallway tape-measure uncertainty.

### 5.3 Checking that the ± is honest
- **Leave-one-walk-out** on the 6 hallway walks: train on 5, run the full board pipeline on the 6th, repeat. (Ensembles of smaller size for speed during this check.) Gives 6 count errors + 6 distance errors, and every window's rate error.
- **Window level:** z = (μ − y)/σ for every held-out window. Honest σ → ~68% of |z| < 1, ~95% < 1.96.
- **Walk level:** fit `s_steps`, `s_dist`, and one scale factor `c` (σ × c) so the claimed coverage matches. With only 6 walks this is rough, so `s` has a floor (1% steps, 3% distance) and the README says so.
- Printed as a **reliability table** (claimed vs. actual coverage at 50/68/95%).
- **Final test:** the long walk (never trained on, never used for `c` or `s`): is the true count inside the 95% range?

### 5.4 Live readout
- `482 steps ± 9 (95%: 473–491)`.
- **How sure:** the probability the true count is within ±1% (at least ±3 steps) of the shown number.
- Breakdown in ± steps: random, systematic, rounding; plus the current ensemble disagreement.
- `351 m ± 12 m`, same breakdown.
- Current cadence (steps/min) and speed (m/s), each with its ±.

---

## 6. Board app (`HW7/step-counter-wearable/`)
- `sketch/` copied from HW6 unchanged.
- `python/stepnet.py`: numpy forward pass of one member, the ensemble combine, and `StepAccumulator` (§5). The laptop imports it too, so replay = board.
- `python/main.py`: Bridge polling → a window every hop → ensemble → accumulator → MQTT `.../hw7/state` about 4×/s → matrix. Commands on `.../cmd`: `reset`, `record:on|off`. Raw rows on `.../raw` only while recording.
- **Matrix:** scrolls the count, e.g. `482`, updated every 2 s; `NO MODEL` if the model file is missing.
- Starts at 0 on every app start (count isn't saved).
- `python/stepnet_model.json`: 5 members' weights, scales, `c`, `s_steps`, `s_dist`, date, validation numbers.
- `requirements.txt`: `paho-mqtt`, `numpy`.

## 7. Laptop side (`HW7/`)
| File | Purpose |
|---|---|
| `PLAN.md` / `PROGRESS.md` | this plan / dated log |
| `README.md` | setup, recording, training, deploying, testing |
| `HOW_THE_NET_WORKS.md` | the explainer + animation notes (§8) |
| `config.py`, `mqttlib.py`, `dataset.py` | from HW6, adapted |
| `gui.py` | **Live tab:** big `steps ± range` and `distance ± range`, how-sure bar, uncertainty breakdown bars, running count plot with 68%/95% bands, cadence with error bars, unfamiliar-motion light, Reset. **Record tab:** hallway length, guided walks (§3.1), list of walks with typed vs auto counts, Train → results → Install and deploy / Discard. |
| `labels.py` | §3.2 stride times + check plots |
| `train_stepnet.py` | windows + labels, augmentation, backprop training of 5 members, leave-one-walk-out check, uncertainty calibration, export, numpy match check. Called by the GUI; on the command line it only installs with `--install` and backs up the old model to `model_history/`. |
| `replay.py` | runs any recording through the exact board code: final count ± vs truth, plot of the count + bands; also a plain gyro-peak counter as a baseline for the README |
| `make_figures.py` | doc figures + `figures/animation_data.json` |
| `testing/synth.py`, `testing/fake_board.py` | synthetic walks with known strides, so everything is tested before Hudson records |

## 8. `HOW_THE_NET_WORKS.md` (written with the code, using the real trained numbers)
1. The problem: one ankle, strides vs steps, why a 2 s window.
2. The input: a real 6 × 100 window from Hudson's walk, what each channel does during a stride.
3. Layer by layer, with real feature maps from that window: convolution, ReLU, pooling, averaging.
4. The two heads, mean + spread, the NLL loss with a worked numeric example.
5. **Backpropagation**: forward pass, loss, chain rule backwards through each layer type (with the gradient of one real weight worked through), Adam step, epochs, the real loss curves, early stopping.
6. Labels and augmentation (why stretching time changes the label).
7. The ensemble: aleatoric vs epistemic, a real example of the members disagreeing.
8. From windows to totals: the √time vs ∝N growth, the reliability table, the long-walk result.
9. Limits.
10. **Animation notes**: a storyboard, each scene with what to show, which real numbers to use, colors and timing:
    1. Ankle with the sensor and its axes; the 6 signals scrolling in.
    2. The 2 s window sliding along 0.24 s at a time.
    3. Convolution: a kernel sliding over a channel, the output trace building; ReLU clipping; pooling halving.
    4. The 32 pattern bars after averaging: a walking window vs a standing window.
    5. The heads: a bell curve (μ, σ) for the rate; wider for an ambiguous window.
    6. **Backprop**: the error at the output flowing backwards layer by layer as glowing gradients, weights nudging, the loss curve dropping over epochs, the bell curve tightening onto the label.
    7. Five ensemble bell curves overlaid: agreeing on normal walking, fanning out on unfamiliar motion.
    8. The step counter ticking up with its band widening (√t random part, cone-shaped systematic part).
    9. Reliability: held-out walks landing inside their bands.
    - `make_figures.py` saves everything those scenes need (a real window, every layer's activations, the 5 members' outputs, a recorded backprop step's gradients, loss curves, the long walk's count and bands) to `figures/animation_data.json`, so the animation uses real data.

## 9. Build order (each step tested before the next)
1. Folder, copied pieces, app skeleton that streams rows + shows `NO MODEL`. Deploy, check 50 rows/s.
2. `testing/synth.py`: synthetic walks with known strides and distances.
3. `labels.py` on synthetic walks (must recover the known strides).
4. `train_stepnet.py` + `stepnet.py` on synthetic data: backprop trains, gradient check passes, numpy matches PyTorch, uncertainty coverage ≈ claimed.
5. `replay.py` + baseline counter.
6. `main.py` + `gui.py` (Live and Record tabs) against `fake_board.py`.
7. **Hudson records** (§3.1, about 10 min) → Train → Install and deploy.
8. Live test on a new counted walk.
9. `make_figures.py`, `HOW_THE_NET_WORKS.md`, README, PROGRESS.
