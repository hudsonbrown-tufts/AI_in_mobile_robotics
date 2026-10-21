"""Activity classifiers for the HW6 ankle wearable.

Used by the board (main.py) and by the laptop scripts (calibrate_part1.py,
train_part2.py, replay.py), so the board and the laptop run exactly the same
decision code.

The file has three sections:
    SHARED  - sample layout, window size, states, smoothing
    PART 1  - peak acceleration along ONE direction, 3 thresholds
    PART 2  - 1D convolutional neural network on all 6 IMU channels

HOW_IT_DECIDES.md describes both parts in depth.
"""

import json
from pathlib import Path

import numpy as np


# =============================================================================
# SHARED
# =============================================================================

STATES = ["standing", "walking", "running", "biking"]
# What the LED matrix scrolls for each state.
STATE_WORDS = {"standing": "STAND", "walking": "WALK", "running": "RUN", "biking": "BIKE"}

SAMPLE_HZ = 50          # rate of the samples the sketch sends (it reads the IMU at 200 Hz)
WINDOW = 100            # samples per decision window = 2.0 s
HOP = 12                # a new decision every 12 samples = 0.24 s

# One sample row (physical units: g and deg/s). The sketch reads the IMU at
# 200 Hz and sends one row per 4 readings:
#   ax..gz          - mean of the 4 readings (used by PART 2)
#   a?_max, a?_min  - highest / lowest of the 4 accel readings on that axis
#                     (used by PART 1, so short impact peaks aren't averaged away)
COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz",
           "ax_max", "ay_max", "az_max", "ax_min", "ay_min", "az_min"]
COL = {name: i for i, name in enumerate(COLUMNS)}

APP_DIR = Path(__file__).resolve().parent
PART1_CALIBRATION_FILE = APP_DIR / "part1_calibration.json"
PART2_MODEL_FILE = APP_DIR / "part2_model.json"


class Smoother:
    """Only change state after `needed` windows in a row agree.

    `None` (an uncertain window) keeps the current state and doesn't count
    toward a change.
    """

    def __init__(self, needed=3):
        self.needed = needed
        self.state = None
        self._candidate = None
        self._count = 0

    def update(self, raw_state):
        if raw_state is None:
            return self.state
        if raw_state == self._candidate:
            self._count += 1
        else:
            self._candidate = raw_state
            self._count = 1
        if self._count >= self.needed:
            self.state = self._candidate
        return self.state

    def reset(self):
        self.state = None
        self._candidate = None
        self._count = 0


# =============================================================================
# PART 1: peak acceleration in ONE direction
# =============================================================================
#
# The only input is the acceleration along a single direction (one axis, one
# sign), picked once by calibrate_part1.py and saved in part1_calibration.json.
# The decision is one number: the highest value in the 2 s window, compared
# against 3 thresholds.

DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]


def part1_direction_signal(window, direction):
    """The acceleration along `direction` ("+x", "-y", ...), from the 200 Hz peaks.

    This is the ONLY data Part 1 ever sees. "+y" uses each row's highest ay
    reading; "-y" uses the negated lowest ay reading, so in both cases a big
    value means a big push in that direction.
    """
    sign, axis = direction[0], direction[1]
    if sign == "+":
        return window[:, COL[f"a{axis}_max"]]
    return -window[:, COL[f"a{axis}_min"]]


def part1_peak(signal):
    """The Part 1 feature: peak acceleration (g) in the window."""
    return float(np.max(signal))


class Part1PeakClassifier:
    """Peak of one direction -> state, using 3 thresholds.

    `order` lists the 4 states from the lowest typical peak to the highest
    (found by calibration, e.g. biking < standing < walking < running), and
    `thresholds` are the 3 boundaries between them, ascending.
    """

    def __init__(self, direction, order, thresholds):
        assert direction in DIRECTIONS
        assert sorted(order) == sorted(STATES) and len(thresholds) == 3
        assert all(a < b for a, b in zip(thresholds, thresholds[1:]))
        self.direction = direction
        self.created = None
        self.order = list(order)
        self.thresholds = [float(t) for t in thresholds]

    @classmethod
    def load(cls, path=PART1_CALIBRATION_FILE):
        with open(path) as f:
            cal = json.load(f)
        obj = cls(cal["direction"], cal["order"], cal["thresholds"])
        obj.created = cal.get("created")      # identifies which calibration is loaded
        return obj

    def classify_peak(self, peak):
        for state, threshold in zip(self.order, self.thresholds):
            if peak < threshold:
                return state
        return self.order[-1]

    def classify(self, window):
        """Classify a window. Returns (state, peak)."""
        peak = part1_peak(part1_direction_signal(window, self.direction))
        return self.classify_peak(peak), peak


# =============================================================================
# PART 2: neural network on all IMU outputs
# =============================================================================
#
# Input: the 2 s window of all 6 channels (ax, ay, az in g, gx, gy, gz in
# deg/s), each divided by a fixed scale saved with the model.
# Network (trained in train_part2.py with PyTorch, run here with numpy):
#   Conv1d(6->16, k7) ReLU MaxPool2 -> Conv1d(16->32, k5) ReLU MaxPool2
#   -> Conv1d(32->32, k5) ReLU -> average over time -> Linear(32->4) -> softmax

PART2_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
PART2_MIN_CONFIDENCE = 0.6   # below this a window counts as "unsure" (keeps the state)


def part2_input(window, scales):
    """(WINDOW, 12) sample rows -> (6, WINDOW) scaled network input."""
    x = window[:, [COL[c] for c in PART2_CHANNELS]].T
    return x / np.asarray(scales, dtype=np.float64)[:, None]


def _conv1d_relu(x, weight, bias):
    """Valid 1D convolution + ReLU. x: (C_in, T), weight: (C_out, C_in, K)."""
    k = weight.shape[2]
    patches = np.lib.stride_tricks.sliding_window_view(x, k, axis=1)   # (C_in, T-K+1, K)
    out = np.einsum("oik,itk->ot", weight, patches) + bias[:, None]
    return np.maximum(out, 0.0)


def _maxpool2(x):
    t = x.shape[1] // 2 * 2
    return x[:, :t].reshape(x.shape[0], t // 2, 2).max(axis=2)


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


class Part2NeuralNet:
    """Numpy forward pass of the trained CNN (weights from part2_model.json)."""

    def __init__(self, model):
        self.classes = model["classes"]
        self.scales = model["scales"]
        w = {k: np.asarray(v, dtype=np.float64) for k, v in model["weights"].items()}
        self.layers = [
            (w["conv1.weight"], w["conv1.bias"], True),
            (w["conv2.weight"], w["conv2.bias"], True),
            (w["conv3.weight"], w["conv3.bias"], False),
        ]
        self.fc_w, self.fc_b = w["fc.weight"], w["fc.bias"]
        self.info = {k: v for k, v in model.items() if k != "weights"}

    @classmethod
    def load(cls, path=PART2_MODEL_FILE):
        with open(path) as f:
            return cls(json.load(f))

    def probabilities_from_input(self, x):
        """x: (6, WINDOW) scaled input -> class probabilities."""
        for weight, bias, pool in self.layers:
            x = _conv1d_relu(x, weight, bias)
            if pool:
                x = _maxpool2(x)
        features = x.mean(axis=1)                    # global average over time
        return softmax(self.fc_w @ features + self.fc_b)

    def classify(self, window):
        """Classify a window. Returns (state or None if unsure, probabilities)."""
        probs = self.probabilities_from_input(part2_input(window, self.scales))
        best = int(np.argmax(probs))
        state = self.classes[best] if probs[best] >= PART2_MIN_CONFIDENCE else None
        return state, probs
