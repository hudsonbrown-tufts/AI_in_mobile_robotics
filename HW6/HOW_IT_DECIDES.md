# How the wearable decides

This document describes how **Part 1** (peak acceleration, one direction) and **Part 2** (nested decision rules, all IMU outputs) each turn the MPU-6050's data into one of **standing / walking / running / biking**. All the decision code is in `ankle-activity-wearable/python/classifiers.py`. The board and the laptop scripts both import that file, so offline results match what the board does.

---

## 0. Shared by both parts

### 0.1 From sensor to sample rows
- The sketch reads the MPU-6050 at **200 Hz**, with a ±16 g range and ±2000 °/s.
  - Running heel strikes at the ankle can pass 8 g, and the shin can spin past 1000 °/s, so smaller ranges would clip.
  - The chip's low-pass filter is set to about 94 Hz, which keeps the short impact spikes.
- Every 4 readings (5 ms apart) become one **50 Hz row** of 12 numbers:

| Values | Used by | Why |
|---|---|---|
| mean of `ax ay az gx gy gz` | Part 2 | smooth 50 Hz signal for Part 2's features (the averaging also reduces noise) |
| **max** of `ax ay az` | Part 1 | a heel strike lasts only about 10–20 ms. Averaging it into a 50 Hz sample would shrink its peak, so the true 200 Hz highest value is kept |
| **min** of `ax ay az` | Part 1 | same idea, for the negative directions (`-x`, `-y`, `-z`) |

### 0.2 Windows
- Each decision looks at the last **2.0 s** (100 rows), so even slow activities fit about two cycles: walking strides take about 1 s, slow pedaling about 1 s per turn.
- A new decision is made every **0.24 s** (12 new rows), so consecutive windows overlap by 88%.
- If the data has a hole of more than 10 rows (Bridge or WiFi stall), the window restarts rather than mixing data from either side of the hole.

### 0.3 Smoothing
- Each part has its own `Smoother`: the displayed state changes only when **3 windows in a row** give the same answer.
- That adds about 0.5 s of delay. In exchange, one odd stride (a stumble, a hard step while walking) can't flip the display.
- A Part 2 window that ends as unsure (lying still, section 2.2) doesn't count toward a change.
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

     Both activities get the same buffer, so on the figure each line sits in the middle of the empty space between two clusters. On the current recordings: T2 = 0.57 g (0.33 g buffer each side) and T3 = 2.42 g (0.58 g each side).
   - **If the two overlap**, `T` is the value that makes the fewest balanced mistakes between L and U. Among equally good values, it takes the middle one.
   - **Exception: the biking | standing boundary gives standing the whole gap.** The threshold sits right at biking's edge, 0.005 g outside it, instead of in the middle (`STRAP_SENSITIVE_PAIR` and `edge_threshold` in `calibrate_part1.py`). On the current recordings that's **T1 = 0.08 g** instead of the midpoint 0.13 g.

     *Why:* in live use, standing was read as biking. Standing has no motion, so its Part 1 peak is only the slice of gravity that falls along the chosen direction. That slice depends on **the strap angle**. On this board the chosen +x direction is nearly horizontal while standing (it reads 0.21 g, about 78° from vertical). At that angle the reading changes about 0.018 g per degree of strap rotation, the fastest it can change. Rotating the recorded standing windows by a few degrees shows the effect:

     | Strap angle differs from calibration by | Standing reads along +x | Gap down to biking's edge (0.08 g) |
     |---|---|---|
     | 0° | 0.20 – 0.23 g | yes |
     | ±5° | 0.11 – 0.31 g | low end below the old midpoint T1 = 0.13 g, so it read as **biking** |
     | ±10° | 0.02 – 0.39 g | low end below even the new T1 |

     Biking's edge doesn't move like that: it comes from the pedaling motion, which the strap angle barely changes. So the room in this gap is far more useful on the standing side. Moving T1 from the midpoint to biking's edge:

     | T1 | Strap rotation standing tolerates | Biking windows still right (recordings) |
     |---|---|---|
     | 0.13 g (midpoint) | about 4.6° | 100% |
     | **0.08 g (biking's edge, used)** | **about 7.5°** | **100%** |
     | 0.04 g (fitted on copies rotated ±8°) | about 9.8° | 96.5%: biking's highest windows start reading as standing |

     Going below biking's edge would cost biking accuracy. One high pedal stroke stays in the 2 s window for about 8 decisions, so 3-in-a-row smoothing doesn't hide it. Biking's edge is the furthest T1 can move for free. Only this one boundary is treated this way:
     - The other side of standing (T2, standing | walking) is already 0.33 g away from standing, which is about 20° of strap rotation.
     - Walking and running peaks are dominated by impacts, not by gravity.
   - **Recording more standing** with the strap at slightly different angles (wizard, only standing ticked, "replace earlier takes" unticked) shows the calibration standing's real spread and is the next step if 7.5° isn't enough.
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
- **Thresholds sit in the middle of the gaps**, not at the edge of the recorded data, so both neighbouring activities have the same room to vary on a new wearer. The exception is biking | standing: there, standing gets all the room, because its reading is the one that moves with the strap angle (section 1.4).
- **The lowest and highest regions are open-ended.** The first activity in `order` owns everything below T1 and the last owns everything above T3. So in the figures, the lowest region looks as wide as the axis is long; only where T1 and T3 sit matters.
- **Leave-one-person-out scoring** picks the direction that holds up across people, not just the one that fits the calibration data best.
- **Fixed mounting** (outside of the right ankle, strap at the same angle every time) keeps the direction meaning the same thing on everyone. On the calibrated board the shin axis turned out to be +y, so +x is the axis across the shin. Marking the strap helps most for standing (see 1.4).

### 1.6 Known weak spots
- **Biking vs standing** is the closest pair. Standing's reading along +x moves about 0.018 g per degree of strap angle. With T1 at biking's edge it tolerates about 7.5° (section 1.4); a strap put on further off than that can still show BIKE while standing. Separately, if someone pedals with their shin pointing up a lot, gravity falls back on that axis and biking's peak rises toward standing's. The direction table shows whether another direction separates them better, and calibration picks it automatically if so.
- **A slow walk** has soft heel strikes. With the window peak, a very gentle shuffle can look like standing.
- **One number can't tell *why* the peak is high.** A stomp while standing looks like a walking stride. The stomp stays inside the 2 s window for about 8 decisions, which is more than the 3 the smoothing needs. So a single hard stomp can briefly show WALK. That's a real limit of a one-direction peak rule, and it's what Part 2's extra inputs fix.

---

## 2. Part 2: nested decision rules on all IMU outputs

Part 2 uses no neural network. (The earlier neural-network version is kept in `HW6-NN/`.) It works out four physical features from all six IMU channels, then answers a fixed sequence of questions. Each question has fitted thresholds. Most questions are a **majority vote of three features**, so a single feature that disagrees is outvoted. That's the "nested" part: a first answer (e.g. "tilted like lying down") is cross-checked by follow-up questions before it's accepted.

### 2.1 Features (one set per 2 s window)

| Feature | Computed from | What it captures | Your recordings, 5th–95th percentile |
|---|---|---|---|
| `tilt_deg` | the window's average acceleration (that's gravity) vs. "up the shin while standing" | lying down vs upright | standing 0–1°, walking 6–13°, running 12–26°, **biking 59–77°** |
| `acc_std` | standard deviation of the acceleration magnitude | moving at all, and how much | standing 0.00 g, walking 0.29–0.51 g, running 0.60–1.80 g, biking 0.33–0.44 g |
| `acc_peak` | highest acceleration magnitude | impacts (heel strikes) | standing 1.02 g, walking 2.0–2.9 g, running 3.2–8.0 g, biking 1.7–2.0 g |
| `rot_var` | √(var gx + var gy + var gz) | how much the shin swings and rotates | standing 0.4–0.7 °/s, walking 145–197 °/s, running 228–370 °/s, biking 112–140 °/s |

- **"Up"** is the average gravity direction during the standing recordings, saved with the rules. On your board it's (0.19, 0.96, 0.21) in sensor x, y, z, which means the shin axis is +y.
- **Why the rotation's variation and not its average:** the gyro reads about 26 °/s even when perfectly still (a constant bias). The variation ignores a constant offset, so it reads about 0 °/s when still and isn't thrown off by the bias.

### 2.2 The questions

```
Q1  Is the shin tilted like lying down?            tilt_deg > T_tilt
├─ yes ── Q1a  Big impacts and swing anyway?       2 of 3: acc_std, acc_peak, rot_var on the running side (biking|running thresholds)
│         ├─ yes → RUNNING     (a hard stride can tilt the shin a lot for a moment)
│         └─ no ── Q1b  Is the leg still?          2 of 3: acc_std, acc_peak, rot_var on the standing side
│                  ├─ yes → unsure (lying still is none of the 4 activities; the last state is kept)
│                  └─ no  → BIKING
└─ no ─── Q2  Is the leg still?                    2 of 3 on the standing side
          ├─ yes → STANDING
          └─ no ── Q3  Does it move like running?  2 of 3: acc_std, acc_peak, rot_var on the running side (walking|running thresholds)
                   ├─ yes → RUNNING
                   └─ no  → WALKING
```

Fitted thresholds on your recordings (`part2_rules.json`, 2026-10-26):

| Question | acc_std | acc_peak | rot_var | tilt |
|---|---|---|---|---|
| Q1 tilted | | | | 41.3° |
| Q2 / Q1b still (standing \| walking) | 0.133 g | 1.47 g | 69 °/s | |
| Q3 walk or run (walking \| running) | 0.566 g | 3.13 g | 210 °/s | |
| Q1a bike or run (biking \| running) | 0.498 g | 2.57 g | 179 °/s | |

**Why these questions, in this order:**
- **Tilt goes first** because it's the cleanest split: biking is 59–77°, everything else is 26° or less, a gap of more than 30°.
- **Q1a guards against a false biking call.** A very hard running stride could swing the shin far for a moment. If the impacts and swing are running-sized, that wins over tilt.
- **Q1b guards against lying still.** Lying down without pedaling is tilted too, but it isn't biking, and it isn't one of the four activities. So the window counts as unsure and the display keeps its last state, the same way smoothing treats any uncertain window.
- **Upright, standing is checked before running vs walking**, because standing is separated by a huge margin (no movement at all).
- **Walking vs running is the hardest call**, and the clusters touch: some fast walks reach 0.6 g of acceleration variation, some light jogs only 3.2 g of peak. That's why it's a 3-feature vote and not a single threshold.

### 2.3 Fitting (`fit_part2.py`, run by the GUI's Shift+C wizard)
- Same data as Part 1: every 2 s window of every recording, all people.
- **"Up":** the mean gravity direction of the standing windows.
- **Thresholds:** for each question and each feature, the threshold uses the same **meet-in-the-middle rule as Part 1** (`calibrate_part1.midpoint_threshold`). It sits halfway between the 99th percentile of the lower group and the 1st percentile of the upper group; if the groups overlap, it's the value with the fewest balanced mistakes. For Q1 the lower group is all upright windows (standing, walking and running) and the upper group is biking.
- **Direction is automatic:** if a feature runs the other way round (the "lower" activity actually has higher values), the test is flipped, and the file records which class each side votes for.
- **Scoring:** leave-one-person-out with 2 or more people, otherwise all data. The fit log prints every feature's range per activity, every threshold with its margin, and the confusion matrix.
- **Result on your recordings:** 99.7% of windows on their own (3 of 920 wrong, all walking ↔ running), 99.1% after smoothing (`replay.py`).
- Like Part 1, nothing changes on the board until **Install** in the wizard. To refit from the existing recordings without recording anything, untick all four activities.

### 2.4 Running on the board
- `classifiers.Part2Rules` computes the 4 features and walks the questions: a few dozen numpy operations per window.
- For the GUI, it reports the four feature values and the path of questions it answered, e.g. `Q1 tilt 69 deg > 41: lying down`, `Q1a 0/3 say running-sized impacts: not running`, `Q1b 0/3 say still: moving, so biking`.

### 2.5 Why it should work on new people
- **The features are physical:** tilt in degrees, accelerations in g, rotation in °/s. A 41° tilt means lying down for anyone with the same mounting.
- **Thresholds sit in the middle of the gaps**, so both neighboring activities have the same room for someone who walks harder or runs softer.
- **Majority votes** mean one feature that's off for a particular person (a light runner's peak, for example) is outvoted by the other two.

### 2.6 Known weak spots
- **Walking ↔ running** is the closest pair (the margins in the fit log are near 0). A very light jog or a very fast walk can land on the wrong side, and a 2 of 3 vote can be one feature away from flipping.
- **Mounting:** "up" is fixed from the calibration recordings. A strap rotated far from that (or worn on the other leg) changes the tilt, so Q1 can misfire.
- **Activities outside the four** (stairs, jumping, sitting) still get forced into one of them, except lying still, which Q1b catches.

---

## 3. Part 1 vs Part 2 at a glance

| | Part 1 | Part 2 |
|---|---|---|
| Input | 1 direction, peak only | all 6 channels, as 4 features |
| Model | 3 thresholds | 5 nested questions, 10 thresholds plus the "up" direction |
| Fitted from data | direction, order, 3 numbers | "up", 10 numbers |
| Neural network | no | no (the earlier NN version is in `HW6-NN/`) |
| Explainable by hand | completely | completely: the GUI shows the questions answered for every window |
| Main confusions to expect | biking ↔ standing, slow walk ↔ standing, stomp ↔ walk | walking ↔ running at the edges; unusual mounting |
| Delay to switch | about 0.5 s smoothing + up to 2 s for the window to fill with the new activity | same |

Use `replay.py` for the actual numbers on your recordings. It prints both parts' confusion matrices before and after smoothing.
