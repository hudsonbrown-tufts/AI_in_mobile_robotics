# How the step-counting network works

This explains, from sensor to "482 steps ± 9", how the HW7 ankle step counter uses a neural network, and how it knows how sure it is. Section 10 is a storyboard for an animation of all of it.

The code is in three places:
- `step-counter-wearable/python/stepnet.py`: what the board runs (numpy). The laptop scripts import the same file, so offline results match the board.
- `train_stepnet.py`: training with PyTorch.
- `labels.py`: where the training labels come from.

> **About the numbers in this file.** Hudson hasn't recorded his walks yet, so the example numbers (marked *synthetic*) come from a full training run on the synthetic test walks in `testing/`. The architecture numbers are exact. After the real training, `make_figures.py` redraws every figure in `figures/` from the real model, and the synthetic numbers here get replaced with the real ones.

---

## 1. The problem

- **One ankle, two feet.** The MPU-6050 is strapped to the outside of the right ankle, so it only feels the right leg directly. One swing of the right leg is one **stride**, which is 2 steps (right + left). The network counts strides, and the board shows steps = 2 × strides.
- **Why not detect each step as an event?** An event detector has to decide "yes/no, a step happened right here". That leaves no natural way to say *how sure* it is, and one missed or doubled event is a whole step wrong. Instead, the network looks at a stretch of motion and estimates a **rate** (strides per second). The rate is a smooth number that also comes with an error bar, and adding up rate × time gives the count.
- **Why 2 s windows:** a walking stride takes about 1.0–1.2 s, so 2 s always holds at least one and a half strides. Shorter windows can miss a whole stride, and longer ones react slowly when you start or stop. 2 s is also what HW6-NN used.
- **Distance:** the same network also estimates **walking speed** (m/s). Distance = speed × time, added up the same way.

## 2. The input

- The sketch reads the IMU at 200 Hz and sends **50 Hz rows**. The network uses 6 numbers per row: acceleration `ax ay az` (g) and rotation rate `gx gy gz` (°/s), each the average of 4 readings.
- A **window** is the last 100 rows (2.0 s), so the input is a 6 × 100 grid of numbers.
- Each channel is divided by a fixed scale (accel ÷ 2 g, gyro ÷ 500 °/s), so all six are about −1…+1 and none drowns out the others. Nothing is normalized per walk, so the network sees gravity's direction too.
- What a stride looks like in the window (see `figures/how_it_works.png`, panel 1):

| Channel | During one stride |
|---|---|
| gyro across the shin (synthetic: `gz`; on Hudson's mounting probably `gx` or `gz`) | the **shin swing**: one big smooth hump, about 250–400 °/s at mid-swing, then a smaller negative dip while the foot is planted. This is the clearest stride signal. |
| accel along the shin (`ax` synthetic, `ay` on the real mounting per HW6) | about 1 g (gravity) plus a sharp **heel-strike spike** of 1–3 g right after the swing |
| the other accel axes | gravity tilting forward and back as the shin swings |
| the other gyro axes | small wobbles; a big burst while **turning** around |

## 3. Layer by layer

The network (one "StrideNet", **8,564 weights**) is a 1D convolutional network, the same trunk as HW6-NN's activity classifier:

| # | Layer | Output shape | What it does |
|---|---|---|---|
| 0 | input | 6 × 100 | 2 s of 6 channels |
| 1 | Conv1d 6→16, kernel 7 | 16 × 94 | 16 **filters**, each a little 6 × 7 pattern (0.14 s long). It slides along the window, and at each position its output is the sum of (pattern × signal): big when the signal looks like the pattern. |
| | ReLU | 16 × 94 | negatives → 0: "pattern present" or "not present" |
| | MaxPool 2 | 16 × 47 | keep the larger of each pair of neighbours: half the length, same information about *what*, less about *exactly where* |
| 2 | Conv1d 16→32, kernel 5, ReLU, MaxPool 2 | 32 × 21 | patterns **of patterns**, about 0.32 s long (16 input rows), e.g. "swing peak, then heel strike" |
| 3 | Conv1d 32→32, kernel 5, ReLU | 32 × 17 | longer stride patterns, about 0.68 s long (34 input rows, about half a stride; the average then covers the whole 2 s) |
| | average over time | 32 | for each of the 32 detectors: **how much of its pattern is in this window**, wherever it occurs |
| 4a | rate head: Linear 32 → 2 | 2 | z₁, z₂ → **mean** μ = softplus(z₁) and **spread** σ = softplus(z₂) + 0.01, in strides/s |
| 4b | speed head: Linear 32 → 2 | 2 | the same, in m/s |

- **Weights per layer:** conv1 688, conv2 2,592, conv3 5,152, heads 66 + 66.
- **softplus(z) = ln(1 + eᶻ)** is a smooth version of ReLU. It keeps the rate and the spread positive (a negative stride rate makes no sense), and the + 0.01 keeps the spread from ever being exactly 0.
- **Why average over time:** the count shouldn't depend on *where* in the window the strides happen. "How much stride pattern is in these 2 s" is exactly what a rate needs.
- *Synthetic example* (`figures/how_it_works.png`): on a normal-pace window, 10 of the 32 averaged detectors are active. The biggest are detector 9 (3.46), detector 19 (3.09) and detector 7 (2.90). The other 22 are 0, because ReLU switched them off for this window.

## 4. Two outputs, each a bell curve, and the loss that teaches it

- Each head outputs **μ (the best guess)** and **σ (how far off it expects to be)**. Together they are a bell curve (Gaussian): the network says "the stride rate is probably μ, give or take σ".
- **The loss** (what training makes small) is the Gaussian **negative log-likelihood (NLL)**:

      loss = ln σ + (y − μ)² / (2σ²)          y = the true value (label)

  - Wrong with a small σ (confidently wrong): the second term is huge.
  - A big σ just to be safe: the ln σ term grows.
  - The cheapest choice is a σ about the size of the network's real mistakes on that kind of window. So σ is **learned**, not set by hand.
  - This σ is the **aleatoric** uncertainty: "this window is genuinely hard to read". Examples: the start or stop of a walk, a turn, a shuffle.
- **β-NLL (β = 0.5):** during training, each window's loss is multiplied by σ^(2β) = σ (held constant during backprop). Without it, a network can "give up" on hard windows by giving them a huge σ, because that makes their loss small whatever μ is. The factor keeps those windows mattering.
- **Total loss** for a window = rate NLL + speed NLL. The speed part is skipped for windows whose distance isn't known (HW6's uncounted walking recording). Standing windows and walking windows are weighted to count equally.

**Worked example** (*synthetic*, network 1, the normal-pace window in the figure):

| | value |
|---|---|
| true stride rate y (from the labels) | 0.943 strides/s (113 steps/min) |
| network says μ | 0.912 strides/s |
| network says σ | 0.047 strides/s |
| loss = ln(0.047) + (0.943 − 0.912)² / (2 · 0.047²) | −3.059 + 0.222 = **−2.837** |

The error (0.031) is smaller than σ (0.047), so the second term is small. The loss is negative because σ < 1. Only *lower* matters, not the sign.

## 5. Training = backpropagation

Training adjusts the 8,564 weights so the loss gets small on the training windows. PyTorch does it by **backpropagation**:

1. **Forward pass:** a batch of 64 windows goes through the network → μ and σ for each.
2. **Loss:** β-NLL against the labels, averaged over the batch.
3. **Backward pass** (`loss.backward()`): PyTorch applies the **chain rule** backwards through every layer. That gives ∂loss/∂w for every weight w: how much the loss would change if that weight moved a tiny bit. Each layer passes "how much did my output matter" back to the layer before it:
   - head: loss ← μ, σ ← softplus ← linear layer
   - ← average over time (the gradient is shared equally by the 17 time steps)
   - ← ReLU (passes the gradient only where the detector was on)
   - ← conv3 ← ...
   - ← max-pool (passes it only to the larger of each pair)
   - ← ... ← conv1
4. **Update** (Adam optimizer, learning rate 0.002): every weight takes a small step *against* its gradient. A plain gradient step would be `w ← w − 0.002 · ∂loss/∂w`. Adam also scales each weight's step by how big and how consistent its recent gradients were.
5. Repeat for every batch. One pass over all the training data is one **epoch**. Each epoch gets freshly augmented windows (section 6).
6. The number of epochs is picked by the leave-one-walk-out check (section 8): it's where the loss on walks the networks never saw is lowest (*synthetic*: 75 epochs). See `figures/training.png`.

**The chain rule on one real weight** (*synthetic*, network 1, the window from section 4). Take the rate head's weight that connects averaged detector 4 (value 0.477) to z₁:

| step | value |
|---|---|
| ∂loss/∂μ = −(y − μ)/σ² = −0.031 / 0.047² | −14.2 |
| ∂μ/∂z₁ = sigmoid(z₁) (the slope of softplus), z₁ = 0.399 | 0.598 |
| ∂z₁/∂w = the detector value feeding that weight | 0.477 |
| **∂loss/∂w = −14.2 × 0.598 × 0.477** | **−4.06**: backprop (PyTorch) got −4.057 |

- The gradient is negative, so the update *increases* w. That raises z₁, then μ, toward the true 0.943.
- For σ: ∂loss/∂σ = 1/σ − (y − μ)²/σ³ = 21.3 − 9.5 = +11.8, positive. So the update *shrinks* σ: the network was closer than its spread said.

**Gradient check:** `train_stepnet.py` checks backprop every training run. For 5 weights spread over all layers, it compares PyTorch's gradient with a "nudge" estimate: move the weight by ±10⁻⁶ and measure the loss change. *Synthetic:* they agree to about 10⁻⁸ (relative). The check uses the plain NLL (β = 0), because β-NLL's held-constant factor is something a nudge can't hold constant.

The board **never** backpropagates. It only runs the forward pass (section 3) in numpy with the frozen weights. Before saving, the training script checks that numpy and PyTorch give the same outputs (*synthetic:* within 3 × 10⁻⁷).

## 6. Labels and augmentation

**Labels** (`labels.py`): Hudson types in how many steps he counted, but the network needs a label for every 2 s window:
1. Find each stride: the shin-swing gyro peaks. The axis and sign are picked automatically, and the threshold is tuned until the number of peaks = typed steps / 2 (±1, since an odd count leaves one stride half-done). Check plots go to `figures/labels/`.
2. **Stride phase:** a counter that is 0.5 at the first stride and goes up by 1 at each next one (straight lines in between). It also ramps up from 0 during the half stride before the first and up to the total during the half stride after the last. It's scaled so the walk adds up to exactly the typed steps / 2.
3. Strides in a window = phase(end) − phase(start), a smooth fractional number. Rate label = that ÷ 2 s. Speed label = rate × this walk's stride length (measured distance ÷ strides).

**Augmentation:** every epoch, each training window is a fresh random variation of the real data, so the networks don't just memorize a few walks:

| Change | Range | Stands in for |
|---|---|---|
| rotate accel and gyro together | ±8° about a random axis | the strap sitting a bit differently |
| scale the motion | ×0.92–1.08 | slightly softer or harder steps |
| **time-stretch** | play 1.7–2.3 s of walk in 2 s (×0.85–1.15) | walking a bit faster or slower. The labels change with it: the window now holds more (or fewer) strides, the rotation is faster, so gyro × stretch, and the speed label scales the same way. |
| noise | 0.01 g, 1.5 °/s | sensor noise |

## 7. The ensemble: two kinds of "not sure"

- **Five networks** are trained the same way, but each starts from different random weights and sees the recordings with different random importance (a *Bayesian bootstrap*: each recording gets a random weight per network). Where the data pins the answer down, they agree. Where it doesn't, they don't.
- On every window, the board runs all five and combines them:

      μ  = average of the 5 μ's
      σ² = average of the 5 σ²'s          aleatoric: "this window is hard to read"
         + variance of the 5 μ's          epistemic: "the networks disagree"

  and then σ is multiplied by the calibration factor c (section 8).
- **Aleatoric vs epistemic:**
  - **aleatoric:** every network sees the same ambiguity. Example: a window that's half standing, half walking. More data of the same kind doesn't remove it.
  - **epistemic:** motion unlike the training data, e.g. stairs, a stumble, or a new pace. The networks were never forced to agree there. The GUI's **unfamiliar-motion light** turns on when the networks' rate estimates spread by more than 1.5 × the 99th percentile on training windows (*synthetic:* 0.064 strides/s = 7.7 steps/min).
- *Synthetic examples* (`figures/ensemble.png`):

| Window | Combined rate | aleatoric | epistemic |
|---|---|---|---|
| normal walking | 0.920 ± 0.048 (true 0.943) | 0.051 | 0.011 |
| most ambiguous (a turn) | 0.775 ± 0.094 (true 0.702) | 0.094 | 0.039 |
| most disagreement (the first steps of a walk) | 0.817 ± 0.088 (true 0.668) | 0.078 | 0.054 |

## 8. From windows to "482 steps ± 9"

**Adding up** (`StepAccumulator` in `stepnet.py`):
- Every 0.24 s a new window arrives and credits its newest 0.24 s: strides += k · μ · 0.24 s (same for distance with speed).
- The first window after a (re)start credits 1 s. Each moment is covered by about 8 overlapping windows, and the first window stands in for the earlier windows that never existed.
- **Standing gate:** a rate below 0.15 strides/s (18 steps/min) counts as standing, so nothing is added. Softplus can't output exactly 0. Without the gate, standing would add about 2 steps a minute.
- **k (bias correction):** the held-out walks showed the network reacts a little late when a walk starts. k = (true total) / (predicted total) over all held-out walks (*synthetic:* k = 1.013 for steps, 1.007 for distance).

**The uncertainty has three parts** (in strides; steps = 2 × strides):

| Part | Cause | Grows like | How |
|---|---|---|---|
| **random** | each window's σ (aleatoric + epistemic) | √time: errors in separate windows partly cancel | Overlapping windows share their errors, so credits inside one 2 s window are treated as fully correlated and separate windows as independent: Var += 2 s × 0.24 s × σ². |
| **systematic** | a whole walk can be a few % off (that day's pace, strap angle, the bias correction itself) | in proportion to the count: doesn't cancel | (s × steps)². s is fit on held-out walks (what's left of their errors after the random part), with a floor of 1% for steps and 3% for distance. |
| **rounding** | which foot started; the left foot's first or last step is invisible | constant | ±1 step, uniform → 1/3 step² |

    σ_steps = √( 4·Var_random + (s·steps)² + 1/3 )
    95% range = steps ± 1.96 σ
    "how sure" = chance the truth is within ±max(3, 1% of the count) steps = erf(tolerance / (σ√2))

- Over a long walk, the random part grows slowly (√t) and the systematic part grows steadily (∝ t). After a few minutes, the systematic part is most of the ±.
- **Distance** works the same way, with the speed σ's and s_dist, and no rounding part.

**Is the ± honest?** (`figures/reliability.png`)
- **Leave-one-walk-out:** for each hallway walk, 3 networks are trained *without* it. Then the board's exact adding-up code runs on it. Every held-out walk is predicted by networks that never saw it.
- **c (spread calibration):** after training, the networks' σ can be a bit too wide or too narrow. c = the root-mean-square of (prediction − truth)/σ over all held-out windows. Dividing by it makes the held-out errors exactly "one σ" on average (*synthetic:* c = 0.92 for rate, 0.97 for speed).
- **Check:** if the ± is honest, 50% / 68% / 95% of held-out cases fall inside the 50% / 68% / 95% ranges.
  - *Synthetic:* whole walks hit 50% / 67% / 100%.
  - Single windows hit 78% / 90% / 95%. Most windows are easy and a few (starts, stops, turns) are much harder, so their errors have heavier tails than a bell curve.
- With 6 walks, the walk-level check is rough: each walk is 17% of it.
- **Final test:** the long walk is never used for training or calibration. *Synthetic:* counted 316 steps, the network said 314 ± 11 (95%: 303–326), and walked 200 m vs. the network's 205 ± 14 m. Both are inside their ranges.

## 9. Limits

- **Trained for Hudson, walking, this mounting.** Running, stairs, another person or another strap position are outside the training data. The unfamiliar-motion light should catch some of it; the ± doesn't cover it.
- **The ± depends on a handful of walks.** s and c are fit on about 6 held-out walks, which is why they have floors.
- **The labels inherit Hudson's counting.** A miscounted walk teaches the network the wrong total. The GUI's independent automatic check flags walks that disagree with the typed count by more than 2 steps (or 5%).
- **Turns:** the hallway walks are there and back, so turns are in the training data. Sharp, unusual turns are still the most uncertain windows.
- **The count lags about 1 s** behind real time, because each window is 2 s long.
- **Gaps:** if WiFi or the Bridge drops more than 10 rows (0.2 s), the window restarts and the steps during the gap aren't counted.

## 10. Animation notes (storyboard)

Real data for every scene is in **`figures/animation_data.json`**, written by `make_figures.py`:

| Key | Contents |
|---|---|
| `model` | layer shapes, number of weights, calibration constants, window/hop length, input scales |
| `windows.typical` | a normal walking window: input in real units and scaled, label, the 5 networks' outputs, the combined output, and network 1's activations at every layer (`conv1`, `relu1`, `pool1`, `conv2`, ..., `relu3`, `features`, `rate_head`, `speed_head`) |
| `windows.ambiguous`, `windows.unfamiliar` | the same without activations: the hardest window and the window where the networks disagree most |
| `conv1_kernels_network1` | network 1's 16 first-layer filters (16 × 6 × 7) |
| `backprop` | one real backward pass of network 1 on the typical window: outputs, labels, loss, the gradient's size per layer, the full gradients and weights of conv1 and both heads, the gradient-check results |
| `training` | held-out loss per epoch, the 5 networks' training loss per epoch, the chosen epoch count |
| `reliability`, `heldout_walks` | claimed vs. actual coverage; each held-out walk's true and predicted counts |
| `long_walk` | the long test walk every 0.24 s: count, σ, distance, σ, cadence, the networks' disagreement, and the true running count from the labels |

**Colours** (the same as the GUI): steps blue `#1a56c4`, distance green `#2f6f5f`, truth red `#d63a2f`, warnings orange `#c77800`, networks 1–5 `#4e79a7 #f28e2b #59a14f #e15759 #b07aa1`, activations on a dark "magma" colour map.

| # | Scene (≈ length) | Show | Data |
|---|---|---|---|
| 1 | **The sensor** (8 s) | A leg walking. The board on the right ankle, x/y/z arrows drawn on it. The shin swings and the gyro arrow glows at mid-swing; the heel strikes and the accel arrow spikes. Six traces scroll in from the right. | `windows.typical.input_units` (loop it) |
| 2 | **Strides vs steps** (5 s) | Footprints appearing L, R, L, R. Only the R prints light up, and one stride bracket spans R→R. Caption: "the ankle sees strides; steps = 2 × strides". | — |
| 3 | **The window** (6 s) | A 2 s box sliding along the six traces, jumping 0.24 s at a time; each jump flashes "new window". | `model.window_s`, `model.hop_s` |
| 4 | **Convolution** (12 s) | One 6 × 7 filter (`conv1_kernels_network1[k]`) slides across the window. At each position its output dot appears below, building a trace (`layers_network1.conv1[k]`). Then ReLU cuts off the negative part (trace → `relu1[k]`), and pooling squeezes it to half width (`pool1[k]`). Pick k = the filter with the largest output. | `conv1_kernels_network1`, `layers_network1.conv1/relu1/pool1` |
| 5 | **Deeper layers** (8 s) | The 16 traces stack into a heatmap, then become 32 rows (`relu2`), then 32 × 17 (`relu3`). Bright blobs line up with the swings in the input above. | `relu1`, `relu2`, `relu3` |
| 6 | **Average → 32 bars** (5 s) | Each row of `relu3` collapses sideways into one bar (`features`). Split screen: a standing window gives almost all bars at 0. | `features` |
| 7 | **Two heads, two bell curves** (8 s) | The 32 bars feed two small heads. Out comes a bell curve on a "steps/min" axis (μ × 120, σ × 120) with the true value as a red line; a speed bell curve on an "m/s" axis next to it. | `windows.typical.members[0]`, `label_*` |
| 8 | **Backpropagation** (15 s) | The red label line and the bell curve: the gap (y − μ) lights up as "error". The error flows backwards as a glowing wave: heads → average → conv3 → conv2 → conv1, each layer pulsing in proportion to `grad_norm_per_layer`. Zoom on one head weight: show the chain-rule numbers from section 5 (−14.2 × 0.598 × 0.477 = −4.06), then the weight ticking up (w − 0.002 × grad). Then fast-forward: the loss curve dropping over the epochs (`training.train_loss_curves[0]`) while the bell curve slides onto the red line and narrows. | `backprop.*`, `training.*` |
| 9 | **Five networks** (10 s) | Five bell curves in the five colours on the same axis. On `windows.typical` they nearly overlap, and the black combined curve is narrow. Switch to `windows.unfamiliar`: they fan apart and the combined curve widens. Label the two parts: width of each curve = aleatoric; spread between them = epistemic. | `windows.*.members`, `windows.*.ensemble` |
| 10 | **Adding up** (12 s) | Counter `0 → 314` ticking up, using `long_walk.steps`. Behind it the 68% and 95% bands (`steps_sd`, × 1.96) widen. A side panel splits the ± into "random" (grows like √t) and "systematic" (a cone ∝ count). The true count (`true_steps_from_labels`) as a dashed line staying inside the band. Dips at the turns. | `long_walk.*`, `model.calibration` |
| 11 | **Is it honest?** (8 s) | The held-out walks drop one by one onto a number line. Each shows its predicted ± range and the true count; a tally of "inside 95%" counts up. End on the reliability plot (claimed vs. actual, near the diagonal). | `heldout_walks`, `reliability` |
| 12 | **Final** (4 s) | The LED matrix scrolling the count, and the GUI reading "314 steps ± 11 (95%: 303–326)". | `long_walk.final` |
