# HW7 progress log

## 2026-10-30: everything coded and tested offline (board unreachable)
- **Written:**
  - Board app `step-counter-wearable/`: HW6's sketch (only the comments and the start-up word changed), `main.py`, `stepnet.py`.
  - Laptop: `gui.py`, `dataset.py`, `labels.py`, `train_stepnet.py`, `replay.py`, `make_figures.py`, `config.py`, `mqttlib.py`.
  - Tests: `testing/synth.py`, `testing/fake_board.py`.
  - Docs: `README.md`, `HOW_THE_NET_WORKS.md`.
- **Changes from plan v2:**
  - **Hallway walks are there and back (one turn each).** The first synthetic run, with one-way walks, undercounted the long walk (which has turns) by 4% and missed its own 95% range. With turns in the training walks, it was inside the range.
  - **Standing gate:** a window under 0.15 strides/s (18 steps/min) adds nothing. Softplus can't output exactly 0, and without the gate standing added about 2 steps per minute.
  - **Bias correction k:** totals × (true / predicted) over the held-out walks (synthetic: 1.013). The network reacts a little late when a walk starts.
  - **Spread calibration c** = the RMS of the held-out z-scores. The first try (68th percentile of |z|) hit its 0.5 clip.
  - **Ensemble diversity:** random starts plus a Bayesian bootstrap (a random weight per recording per network), instead of resampling windows. Windows from one walk are nearly identical, so resampling them gives little diversity.
  - **Training runs in parallel processes** (18 fold networks, then the 5 final ones), with the GPU hidden. One worker once tried to start CUDA and failed with "device busy". About 1.5 min instead of 5.
  - **The GUI auto-stops a recording** 3 s after Hudson stops walking, so he doesn't have to walk back to the laptop.
- **Tested on synthetic data** (6 there-and-back walks, a 179 s long walk with 7 turns, 30 s standing):
  - The labels find exactly the typed number of strides in every walk.
  - Training: about 105 s.
  - **Gradient check:** backprop vs. finite differences agree to 2e-8 (relative). The first version wrongly included β-NLL's held-constant factor in the finite difference; it now uses the plain NLL.
  - numpy (board) vs. PyTorch: 3e-7.
  - Held-out walks: 1.8% mean step error. Coverage 50% / 67% / 100% at 50% / 68% / 95%.
  - **Long test walk:** counted 316 steps, network 314 ± 11 (inside the range); distance 200 m, network 205 ± 14 m (inside).
  - `replay.py`: standing reads 0.0 steps. The plain peak counter is perfect on synthetic data, whose swing peaks are clean; real data will tell.
  - The real `main.py` (stubbed MQTT, fast fake sketch): the reset works; after it, the board said 122 ± 4.9 steps vs. 124 true; 3 ms per window on the laptop.
  - GUI Live tab screenshot with those messages: all panels fill in correctly.
  - GUI Record and train tab, run on a copy of the synthetic data:
    - The Train button trains in the background (109 s) and enables Install / Discard.
    - The after-walk check runs.
    - Its independent count first used 40% of a big swing as the threshold. That missed the small strides in the turn (72 vs. 76 typed) and falsely offered a redo. At 20% it matches every synthetic walk within 1 step.
  - `make_figures.py`: all 5 figures and `animation_data.json` (172 KB). The worked backprop example in HOW_THE_NET_WORKS.md section 5 was checked by hand against PyTorch's gradient (−4.06 vs. −4.057).
- **Not done (needs the board):**
  - The board was unreachable at both `HOSTS` addresses (192.168.1.137, HudsonArduino.local), and there's no MQTT broker on the laptop. So `fake_board.py` against the GUI over real MQTT, and the deploy, are untested.

## Next
1. Power the board / fix `HOSTS`, then `python tools/deploy.py step-counter-wearable`. Check: `NO MODEL` scrolls, about 50 rows/s, 0 dropped, and the GUI shows "board online".
2. Record (README section 5): hallway length, 6 there-and-back walks, 1 long walk. Train → Install and deploy.
3. Check `figures/labels/` (`labels.py`), run `make_figures.py`, and replace the synthetic numbers in HOW_THE_NET_WORKS.md with the real ones.
4. A new counted walk with the live GUI: is the true count inside the 95% range?
