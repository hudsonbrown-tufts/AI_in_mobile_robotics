"""Activity classifiers for the HW6 ankle wearable.

Used by the board (main.py) and by the laptop scripts (calibrate_part1.py,
fit_part2.py, replay.py), so the board and the laptop run exactly the same
decision code.

The file has three sections:
    SHARED  - sample layout, window size, states, smoothing
    PART 1  - peak acceleration along ONE direction, 3 thresholds
    PART 2  - nested decision rules on features of all 6 IMU channels (no neural network)

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
PART2_RULES_FILE = APP_DIR / "part2_rules.json"


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
# PART 2: decision rules on all IMU outputs (no neural network)
# =============================================================================
#
# Four features per 2 s window, computed from all 6 channels:
#   tilt_deg  - angle (deg) between the window's average acceleration (gravity)
#               and "up the shin while standing"  -> lying down vs upright
#   acc_std   - how much the acceleration magnitude varies (g)  -> moving or not
#   acc_peak  - the highest acceleration magnitude (g)  -> impacts
#   rot_var   - how much the rotation varies: sqrt(var gx + var gy + var gz)
#               (deg/s). Uses the variation, not the average, because the
#               gyro reads ~26 deg/s even when still (a constant bias).
#
# The questions, in order (thresholds and "up" come from part2_rules.json,
# fitted by fit_part2.py with the same meet-in-the-middle rule as PART 1):
#   Q1 Is the shin tilted like lying down?            (tilt_deg)
#     yes: Q1a Do the impacts/swing say running instead?   (3 votes: acc_std, acc_peak, rot_var)
#              -> running   (tilt can look like biking for a moment in a hard stride)
#          Q1b Is the leg still?                           (3 votes)
#              -> unsure: lying still is none of the 4 activities, so the state is kept
#          otherwise -> biking
#     no:  Q2 Is the leg still?                            (3 votes)
#              -> standing
#          Q3 Does it move like running?                   (3 votes)
#              -> running, otherwise walking
# Each multi-feature question is a majority vote of 3 features (at least 2
# agree), so one feature that disagrees with the other two is outvoted.

PART2_FEATURES = ["tilt_deg", "acc_std", "acc_peak", "rot_var"]
MOVEMENT_FEATURES = ["acc_std", "acc_peak", "rot_var"]


def part2_features(window, up):
    """The 4 PART 2 features of a (WINDOW, 12) window. `up` = unit gravity direction while standing."""
    acc = window[:, [COL["ax"], COL["ay"], COL["az"]]]
    gyr = window[:, [COL["gx"], COL["gy"], COL["gz"]]]
    g = acc.mean(axis=0)
    n = float(np.linalg.norm(g))
    tilt = float(np.degrees(np.arccos(np.clip(g @ np.asarray(up) / n, -1.0, 1.0)))) if n > 1e-6 else 90.0
    mag = np.linalg.norm(acc, axis=1)
    return {
        "tilt_deg": tilt,
        "acc_std": float(mag.std()),
        "acc_peak": float(mag.max()),
        "rot_var": float(np.sqrt(gyr.var(axis=0).sum())),
    }


def side(value, test):
    """Which class one feature votes for: test = {"threshold", "above", "below"}."""
    return test["above"] if value > test["threshold"] else test["below"]


class Part2Rules:
    """The PART 2 question sequence with thresholds from part2_rules.json."""

    def __init__(self, rules):
        self.up = np.asarray(rules["up"], dtype=float)
        self.questions = rules["questions"]
        self.created = rules.get("created")
        self.info = {k: v for k, v in rules.items() if k != "questions"}

    @classmethod
    def load(cls, path=None):
        with open(path or PART2_RULES_FILE) as f:
            return cls(json.load(f))

    def vote(self, question, features, wanted):
        """How many of the question's features vote for `wanted` (and how many there are)."""
        tests = self.questions[question]
        return sum(side(features[f], t) == wanted for f, t in tests.items()), len(tests)

    def decide(self, features):
        """features -> (state or None if unsure, list of the questions answered)."""
        path = []
        tilt = self.questions["tilted"]["tilt_deg"]
        if features["tilt_deg"] > tilt["threshold"]:
            path.append(f"Q1 tilt {features['tilt_deg']:.0f} deg > {tilt['threshold']:.0f}: lying down")
            n, of = self.vote("bike_or_run", features, "running")
            if n * 2 > of:
                path.append(f"Q1a {n}/{of} say running-sized impacts: running")
                return "running", path
            path.append(f"Q1a {n}/{of} say running-sized impacts: not running")
            n, of = self.vote("still", features, "standing")
            if n * 2 > of:
                path.append(f"Q1b {n}/{of} say still: lying still, unsure")
                return None, path
            path.append(f"Q1b {n}/{of} say still: moving, so biking")
            return "biking", path
        path.append(f"Q1 tilt {features['tilt_deg']:.0f} deg <= {tilt['threshold']:.0f}: upright")
        n, of = self.vote("still", features, "standing")
        if n * 2 > of:
            path.append(f"Q2 {n}/{of} say still: standing")
            return "standing", path
        path.append(f"Q2 {of - n}/{of} say moving")
        n, of = self.vote("walk_or_run", features, "running")
        state = "running" if n * 2 > of else "walking"
        path.append(f"Q3 {n}/{of} say running: {state}")
        return state, path

    def classify(self, window):
        """Classify a window. Returns (state or None if unsure, {"features", "path"})."""
        features = part2_features(window, self.up)
        state, path = self.decide(features)
        return state, {"features": features, "path": path}
