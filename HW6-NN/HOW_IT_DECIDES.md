# How the wearable decides

This document describes how **Part 1** (peak acceleration, one direction) and **Part 2** (neural network, all IMU outputs) each turn the MPU-6050's data into one of **standing / walking / running / biking**. All the decision code is in `ankle-activity-wearable/python/classifiers.py`. The board and the laptop scripts both import that file, so offline results match what the board does.

---

## 0. Shared by both parts

### 0.1 From sensor to sample rows
- The sketch reads the MPU-6050 at **200 Hz**, with a ±16 g range and ±2000 °/s.
  - Running heel strikes at the ankle can pass 8 g, and the shin can spin past 1000 °/s, so smaller ranges would clip.
  - The chip's low-pass filter is set to about 94 Hz, which keeps the short impact spikes.
- Every 4 readings (5 ms apart) become one **50 Hz row** of 12 numbers:

| Values | Used by | Why |
|---|---|---|
| mean of `ax ay az gx gy gz` | Part 2 | smooth 50 Hz signal for the network (the averaging also reduces noise) |
| **max** of `ax ay az` | Part 1 | a heel strike lasts only about 10–20 ms. Averaging it into a 50 Hz sample would shrink its peak, so the true 200 Hz highest value is kept |
| **min** of `ax ay az` | Part 1 | same idea, for the negative directions (`-x`, `-y`, `-z`) |

### 0.2 Windows
- Each decision looks at the last **2.0 s** (100 rows), so even slow activities fit about two cycles: walking strides take about 1 s, slow pedaling about 1 s per turn.
- A new decision is made every **0.24 s** (12 new rows), so consecutive windows overlap by 88%.
- If the data has a hole of more than 10 rows (Bridge or WiFi stall), the window restarts rather than mixing data from either side of the hole.

### 0.3 Smoothing
- Each part has its own `Smoother`: the displayed state changes only when **3 windows in a row** give the same answer.
- That adds about 0.5 s of delay. In exchange, one odd stride (a stumble, a hard step while walking) can't flip the display.
- A Part 2 window the network is unsure about doesn't count toward a change (section 2.6).
- On startup the state is empty until the first 3 agreeing windows, about 2.5 s. The matrix shows `HW6` until then.

### 0.4 The mode
- Both parts run on **every** window.
- The mode (1 or 2, set from the GUI) only picks which part's smoothed state the LED matrix scrolls and the GUI banner shows.
- So a mode switch takes effect immediately, with no re-warm-up, and the GUI can show both parts at once for comparison.

---

## 1. Part 1: peak acceleration in one direction

### 1.1 The only input
- Part 1 sees exactly one signal: the acceleration along **one direction** — one axis, one sign:

      part1_direction_signal(window, "+y") = the row-wise max of ay
      part1_direction_signal(window, "-y") = −(the row-wise min of ay)

  The sign is applied so that "bigger" always means "harder push in that direction".
- The gyro and the other two axes are never passed to Part 1.
- The direction is fixed in `part1_calibration.json`. The board never changes it.

### 1.2 The feature
- One number per window: **peak = the highest value of that signal in the 2 s window**, in g.
- The peak includes gravity, which helps rather than hurts:

| Activity | What sets the peak along the shin axis |
|---|---|
| biking (lying down) | shin roughly horizontal, so little of gravity falls on the shin axis. Pedaling is smooth, with no impacts. **Low peak.** |
| standing | gravity straight along the shin: **about 1.0 g**, very steady |
| walking | heel strikes: **about 1.5–3 g** |
| running | much harder heel strikes: **about 3.5–8+ g** |

- With the shin axis, the activities fall in the order biking < standing < walking < running. Four bins need three thresholds.

### 1.3 The decision
- `order` lists the four activities from lowest to highest typical peak. `thresholds` are the three boundaries `T1 < T2 < T3`:

      peak < T1  → order[0]
      peak < T2  → order[1]
      peak < T3  → order[2]
      otherwise  → order[3]

- That's the whole runtime rule (`Part1PeakClassifier.classify_peak`).
- `order` is saved with the thresholds because a different direction can sort the activities differently. Along a forward axis, for example, standing reads about 0 g, below biking.

### 1.4 One-time calibration (`calibrate_part1.py`, run by the GUI's Shift+C wizard)
- **What it fixes, and what it leaves alone.** Calibration only picks the direction, the order and the thresholds. Runtime never adapts to the wearer.
- **Where it runs.** Step 6 of the calibration wizard runs it on that session's recordings, plus earlier ones if "also use earlier recordings" is ticked. The result only takes effect once you press Install. Until then, the board keeps the calibration it has.
- **Inputs:** every window of every recording, from every person.

**Steps, for each of the 6 candidate directions (+x, −x, +y, −y, +z, −z):**
1. **Peak per window:** compute the Part 1 peak for every window, using the same function the board uses.
2. **Order:** sort the four activities by their **median** peak.
3. **One threshold per neighbouring pair** (lower activity L, upper activity U):
   - Take the top edge of L's cluster, its 99th-percentile peak (`L99`), and the bottom edge of U's cluster, its 1st-percentile peak (`U1`).
   - **If `L99 < U1`, there's a clean gap**, and the threshold goes exactly in the middle of it:

         T = (L99 + U1) / 2

     Both activities get the same buffer, so on the figure each line sits in the middle of the empty space between two clusters. On the current recordings: T1 = 0.13 g (0.06 g buffer each side), T2 = 0.57 g (0.33 g each side) and T3 = 2.42 g (0.58 g each side), at 100% on the calibration data.
   - **If the two overlap**, `T` is the value that makes the fewest balanced mistakes between L and U. Among equally good values, it takes the middle one.
   - The 99th/1st percentiles sit at the very edge of each cluster, but a single stray window (a stumble) can't drag a threshold.
   - **Earlier versions of this rule**, kept here because they explain why it changed:
     - *Split by spread* (gap shared in proportion to each side's standard deviation, edges at the 5th/95th percentiles). Standing varied by only 0.01 g in the recordings, so it got almost none of either gap: T1 = 0.18 g and T2 = 0.24 g hugged its 0.20–0.22 g readings.
     - *Spread with a 0.15 g floor.* This gave standing room (T1 = 0.08 g, T2 = 0.51 g), but pushed T1 onto biking's highest windows.
     - Meeting in the middle of the cluster edges fixes both problems.
4. **Score the direction:**
   - With ≥ 2 people: **leave-one-person-out** balanced accuracy. Fit the thresholds without person P, test on P, and repeat for everyone. This directly measures how well the direction works on a person whose data wasn't used.
   - With 1 person, the accuracy on all the data is used instead.
   - "Balanced" means each activity counts equally, however long its recordings are.
5. **Pick the best:** the highest accuracy wins. Ties are broken by the **margin**: the worst gap `(U1 − L99)/(sd_L + sd_U)` among the three boundaries, i.e. how far the closest pair is from overlapping.

After picking, the thresholds are fitted on **all** the data and saved. The script prints the whole table, so you can see why that direction won.

### 1.5 Why it should work on new people
- **Thresholds sit in the middle of the gaps**, not at the edge of the recorded data, so both neighbouring activities have the same room to vary on a new wearer.
- **The lowest and highest regions are open-ended.** The first activity in `order` owns everything below T1 and the last owns everything above T3. So in the figures, the lowest region looks as wide as the axis is long; only where T1 and T3 sit matters.
- **Leave-one-person-out scoring** picks the direction that holds up across people, not just the one that fits the calibration data best.
- **Fixed mounting** (+X up the shin, outside of the right ankle) keeps the direction meaning the same thing on everyone.

### 1.6 Known weak spots
- **Biking vs standing** is the closest pair. If someone pedals with their shin pointing up a lot, gravity falls back on the shin axis and the peak nears 1 g. The direction table shows whether another direction separates them better. Calibration already picks it automatically if so.
- **A slow walk** has soft heel strikes. With the window peak, a very gentle shuffle can look like standing.
- **One number can't tell *why* the peak is high.** A stomp while standing looks like a walking stride. The stomp stays inside the 2 s window for about 8 decisions, which is more than the 3 the smoothing needs. So a single hard stomp can briefly show WALK. That's a real limit of a one-direction peak rule, and it's what Part 2's extra inputs fix.

---

## 2. Part 2: neural network on all IMU outputs

### 2.1 Input
- The same 2 s window, using the 6 averaged channels: `ax ay az` (g) and `gx gy gz` (°/s). That's an array of shape **6 × 100**.
- Each channel is divided by a **fixed** scale saved with the model: accel ÷ 2 g, gyro ÷ 500 °/s. This brings both to similar sizes.
- Nothing is normalized per user or per session. The mean isn't subtracted either, because gravity's direction (upright vs lying down) is one of the most useful clues.

### 2.2 Network (1D CNN, 8,564 weights)

| Layer | Output shape | What it learns |
|---|---|---|
| input | 6 × 100 | |
| Conv1d 6→16, kernel 7 (0.14 s), ReLU | 16 × 94 | short shapes: a heel-strike spike, a zero crossing of the shin swing |
| MaxPool 2 | 16 × 47 | |
| Conv1d 16→32, kernel 5, ReLU | 32 × 43 | combinations over about 0.4 s: strike followed by swing, a smooth pedal arc |
| MaxPool 2 | 32 × 21 | |
| Conv1d 32→32, kernel 5, ReLU | 32 × 17 | whole-stride patterns, about 0.9 s |
| average over time | 32 | "how much of each pattern is in this window", wherever in the window it happens |
| Dropout 0.3 (training only) → Linear 32→4 | 4 | |
| softmax | 4 probabilities | |

- **Why a CNN on the raw signal, not hand-made features:** the network learns its own detectors for strides, impacts and pedal strokes. It can use relationships Part 1 can't, such as:
  - the shin's tilt (gravity direction across all 3 axes);
  - rotation speed (gyro);
  - the timing between an impact and the swing that follows it;
  - how smooth the motion is.
- **Why averaging over time:** the answer shouldn't depend on where in the 2 s a stride lands, and averaging makes it independent of that.
- The model is kept small (8.5k weights) on purpose. The dataset is a few minutes of recordings, and a bigger network would memorize it.

### 2.3 What it can tell apart that Part 1 can't

| Activity | Clues across all six channels |
|---|---|
| standing | gravity along the shin, almost no rotation, no impacts |
| walking | upright on average; about 1 Hz rhythm; moderate impacts; shin swing of a few hundred °/s |
| running | upright on average; faster rhythm (about 1.4 Hz); big impacts; much faster shin swing |
| biking | gravity mostly *off* the shin axis (lying down); smooth, continuous one-axis rotation; no impacts |

### 2.4 One-time training (`train_part2.py`, run by the GUI's Shift+C wizard)
- **Where it runs.** Step 6 of the calibration wizard runs it right after Part 1, on the same recordings. Like Part 1, the new network only replaces the one on the board when you press Install.
- **Windows:** taken from every recording exactly as the board would see them: 2 s long, every 0.24 s.
- **Fresh augmentation every epoch**, so it works on people who weren't recorded. Each training window gets a new random version of:

| Augmentation | Range | Stands in for |
|---|---|---|
| rotation of accel *and* gyro by the same random small 3D rotation | ±20° | the strap sitting at a slightly different angle on a different leg |
| movement amplitude: `gravity + k·(accel − gravity)`, gyro × k | k = 0.8–1.25 | lighter or heavier steppers (gravity itself is not scaled, since it's always 1 g) |
| time stretch: resample 1.6–2.5 s of signal into 100 samples | 0.8–1.25× | slower or faster cadence and pedaling |
| Gaussian noise | 0.02 g, 2 °/s | sensor noise |

- **Loss:** cross-entropy, weighted so each activity counts equally even if its recordings are longer.
- **Optimizer:** Adam (lr 0.002, weight decay 1e-4), batches of 64.
- **Early stopping:** on validation loss, with patience 20 epochs and at most 120.

**Validation** (printed, not used on the board), from most to least trustworthy:
1. **≥ 2 people: leave-one-person-out.** Train without person P, test on P, and repeat. This is the number that answers "does it work on someone new?"
2. 1 person, ≥ 2 recordings per activity: hold out the newest recording of each activity.
3. Otherwise: hold out the last 25% of every recording, with a 1 s gap so no training window overlaps a validation window.

Splits are by **person or recording, never by random window**. Neighbouring windows overlap by 88%, so a random split would test the model on data it has essentially already seen.

**Final model:**
- Retrained on **all** data for the median best epoch count from validation.
- Exported to plain JSON.
- The script runs both the PyTorch model and the board's numpy code on the same windows. It refuses to save if their probabilities differ by more than 1e-4. On test data they agree to about 1e-8.

### 2.5 Running on the board
- `Part2NeuralNet` in `classifiers.py` re-implements the forward pass in about 30 lines of numpy: convolution, ReLU, max-pool, mean, linear layer, softmax.
- No PyTorch on the board, only `numpy`. One window takes about 1 ms on the laptop, and an estimated few ms on the board's ARM CPU. Decisions come only 4 times a second, so that's plenty.

### 2.6 Confidence gate
- If the highest probability is **below 60%**, the window counts as "unsure": the smoother keeps the current state.
- This mostly happens during transitions, e.g. lying down to start biking, when the 2 s window holds half of each activity.
- The GUI shows these windows as "unsure (<60%)".

### 2.7 Known weak spots
- The network can only be as general as its data. Recordings from more than one person matter more than more minutes from one person.
- Activities it never saw (stairs, jumping, sitting) still get forced into one of the 4 classes. The confidence gate catches some of these, not all.
- A strap mounted far outside the ±20° augmentation range, or on the other leg (which mirrors the axes), isn't covered.

---

## 3. Part 1 vs Part 2 at a glance

| | Part 1 | Part 2 |
|---|---|---|
| Input | 1 direction, peak only | 6 channels, full 2 s waveform |
| Model | 3 thresholds | 1D CNN, 8,564 weights |
| Fitted from data | direction, order, 3 numbers | all weights |
| Explainable by hand | completely | through the clues in 2.3, not weight by weight |
| Main confusions to expect | biking ↔ standing, slow walk ↔ standing, stomp ↔ walk | transitions; unusual mounting |
| Delay to switch | about 0.5 s smoothing + up to 2 s for the window to fill with the new activity | same |

Use `replay.py` for the actual numbers on your recordings. It prints both parts' confusion matrices before and after smoothing.
