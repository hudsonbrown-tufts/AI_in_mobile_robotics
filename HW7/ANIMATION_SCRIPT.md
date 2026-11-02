# Animation script: how the HW7 step counter works

**Length:** about 62 s, in 11 scenes.
**Format:** 1920×1080 MP4 at 30 fps, rendered with matplotlib from the real trained model and Hudson's real recordings (`figures/animation_data.json` and the installed `stepnet_model.json`).
**Sound:** none; the captions on screen tell the story.
**Colours:** the same as the GUI and `five_networks.jpg`.
- networks 1–5: blue, orange, green, red, purple
- truth: dashed red
- activations: the "magma" colour map

Each scene lists what's on screen, then the **caption** (bottom of the screen, 1–2 lines), and, where it applies, a **"why" card**. A why card is a small box at the top right that explains the design choice.

---

### 0. Title (0:00–0:03)
- **Screen:** "How the ankle step counter works" over a faint, looping trace of Hudson's gyro signal.
- **Caption:** *From 6 sensor signals to "110 steps ± 6": a neural network, PyTorch and backpropagation.*

### 1. The sensor (0:03–0:08)
- **Screen:** a simple leg outline walking, with the UNO Q + MPU-6050 on the right ankle and x/y/z arrows on the board. As the shin swings, its gyro arrow glows; at each heel strike, the accel arrow spikes. Six real traces (ax ay az gx gy gz, from Hudson's hallway walk) scroll in on the right.
- **Caption:** *The ankle feels only the right leg: one swing = one stride = 2 steps. The sensor sends 6 signals, 50 times a second.*

### 2. The 2 s window (0:08–0:12)
- **Screen:** a box 2 s wide slides along the six traces, jumping 0.24 s at a time; each jump flashes "new window → network". The box zooms into a 6 × 100 grid of numbers.
- **Caption:** *Every 0.24 s, the network looks at the last 2 seconds: always at least 1½ strides.*
- **Why card:** *Input scaled: accel ÷ 2 g, gyro ÷ 500 °/s, so all 6 signals are about −1…+1 and none drowns out the others. No activation: these are just the measurements.*

### 3. Layer 1: convolution + ReLU + max-pool (0:12–0:21)
- **Screen:**
  - A small 6 × 7 filter (network 1's real filter with the biggest output) slides along the window.
  - At each position, the equation `z = b + Σ W·x` shows briefly, and an output dot drops below, drawing a trace.
  - ReLU then cuts off the trace below 0: the negative part fades out, and a counter says "43% switched off".
  - Max-pool squeezes the trace to half its width.
  - 16 such traces stack into a heatmap.
- **Caption:** *Layer 1: 16 small filters slide along the signals. Each lights up where the motion looks like its shape: a heel spike, the start of a swing.*
- **Why cards (one after another):**
  - **Convolution:** *A stride looks the same wherever it falls in the window, so one sliding filter is reused at every position. 688 weights instead of tens of thousands.*
  - **ReLU, a = max(0, z):** *"Is this pattern here: yes, by how much, or no." Without it, stacking layers would collapse into one plain filter, so the network couldn't learn anything more than layer 1 alone. Its slope is 1 wherever it's on, so backprop passes gradients through without shrinking them.*
  - **Max-pool:** *Keep the stronger of each pair of neighbours. Half as long, so later layers see more time, and the exact timing can shift a little.*

### 4. Layers 2 and 3 (0:21–0:26)
- **Screen:** the 16-row heatmap turns into 32 rows (layer 2), then into 32 × 17 (layer 3). A bracket under each shows how much time one value "sees": 0.14 s → 0.32 s → 0.68 s. In layer 3, bright blobs line up with each swing in the input above.
- **Caption:** *Layers 2 and 3 combine patterns into longer ones: "swing, then heel strike", then most of a stride (~0.7 s).*
- **Why card:** *Same recipe (convolution → ReLU), with kernels of 5 and pooling, so each layer sees about twice as far back in time. 3 layers is just enough to see one full stride.*

### 5. Average over time (0:26–0:30)
- **Screen:** each of the 32 rows collapses sideways into one bar (network 1's real features). A split screen shows a standing window, where almost all bars drop to 0.
- **Caption:** *Average each detector over the window: 32 numbers saying how much stride-pattern is in these 2 seconds.*
- **Why card:** *The count shouldn't depend on where in the window the strides happen. An average throws away "where" and keeps "how much", which is exactly what a rate needs.*

### 6. Two heads + softplus (0:30–0:36)
- **Screen:**
  - The 32 bars feed two small heads.
  - The softplus curve appears with network 1's real numbers: z₁ = 0.17 → μ = 94 steps/min; z₂ = −3.55 → σ = ±4.6.
  - A bell curve forms on a steps/min axis, next to the true value (red, 93).
  - A second, smaller bell curve shows speed (m/s).
- **Caption:** *Each head gives a best guess μ and how unsure it is, σ: a bell curve, for cadence and for walking speed.*
- **Why card:** *Softplus, ln(1 + eᶻ): a cadence and a spread can't be negative. It's a smooth ReLU that never goes flat at 0. A spread of exactly 0 would break the loss, so +0.01.*

### 7. Training in PyTorch: backpropagation (0:36–0:48)
- **Screen:**
  1. A "PyTorch, on the laptop" frame appears around the network.
  2. **Forward:** windows flow in, and bell curves come out.
  3. **Loss:** the gap between the bell curve and the red truth lights up, with `L = ln σ + (y − μ)² / (2σ²)`.
  4. **`loss.backward()`:** a glowing wave flows backwards: heads → average → layer 3 → layer 2 → layer 1. Each layer pulses as brightly as its real gradient size.
  5. **Chain rule:** zoom on one real head weight: ∂L/∂μ × ∂μ/∂z × ∂z/∂w = the gradient. The numbers come from a real backward pass.
  6. **Adam step:** `w ← w − 0.002 · ∂L/∂w`. The weight ticks.
  7. **Fast-forward:** the epoch counter runs 1 → 66, the real loss curve drops, and the bell curve slides onto the red line and narrows.
- **Caption:** *PyTorch records every calculation, then the chain rule runs it backwards: how much did each of the 8,564 weights cause the error? Each weight takes a small step to fix it. 66 times through all the walks.*
- **Why card:** *The loss punishes being confidently wrong (small σ, big error) AND being vague for no reason (big σ). So σ learns to match the network's real mistakes.*

### 8. Why 5 networks (0:48–0:56)
- **Screen:**
  - The trained network splits into 5 copies, in 5 colours.
  - Under each, its "trust in each recording" bars appear (from `five_networks.jpg`), and its random-start dice rolls.
  - **Normal walking window:** 5 bell curves almost on top of each other, and the black combined curve is narrow (96 ± 6 vs true 93).
  - **Most unusual window (a walk start):** the curves fan apart, and the combined curve widens (42 ± 11 vs true 66).
  - The equation appears: `σ̄² = average of σ² (own unsureness) + spread of the μ's (disagreement)`.
- **Caption:** *5 copies, same recipe, each with a different random start and different trust in each walk. Where your data is clear, they agree. Where it isn't, they disagree, and the ± grows.*
- **Why card:** *One network's σ only knows "this window is ambiguous". It can't know when it's out of its depth, because it's confidently wrong on motion it never saw. Their disagreement measures that. 5 is enough to estimate a spread, and still only about 15 ms per window on the board, 4 times a second.*

### 9. From windows to "110 steps ± 6" (0:56–1:00)
- **Screen:** Hudson's real long walk at 10× speed. The counter ticks 0 → 110, and behind it the 68% and 95% bands widen. The dashed true count (from the labels) stays inside the band. A small bar splits the ± into random, systematic and rounding parts.
- **Caption:** *Every 0.24 s: steps += cadence × time. The ± adds up too. You counted 111; the network said 110 ± 6.*

### 10. Laptop → board (1:00–1:03)
- **Screen:** the weights fly out of the PyTorch frame into a `stepnet_model.json` file icon, then onto the UNO Q, whose LED matrix scrolls "110". A tag reads "numpy only, matches PyTorch to 0.0000003".
- **Caption:** *PyTorch trains on the laptop; the board just runs the forward pass with the frozen weights. No backprop on the board.*

---

## Data each scene uses
| Scene | Source |
|---|---|
| 1, 2 | the real window of a hallway walk (`windows.typical.input_units`) and a stretch of that walk |
| 3–6 | network 1's real filters, layer outputs, features and head values on that window |
| 7 | a real backward pass of network 1 (`backprop.*`) and the real loss curves (`training.*`) |
| 8 | the 5 networks' trust weights and their answers on the typical and the unusual window |
| 9 | the long walk run through the board code (`long_walk.*`) |
| 10 | the real numpy-vs-PyTorch check from training |

Every number is recomputed from the installed model when the animation is made. If you retrain before then, the numbers in this script will change a little.
