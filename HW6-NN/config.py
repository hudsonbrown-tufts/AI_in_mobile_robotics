"""Laptop-side settings for HW6. Importing this also makes the board's
classifiers.py importable (`import classifiers`), so the laptop scripts use the
exact same decision code as the board."""

import sys
from pathlib import Path

HW6_DIR = Path(__file__).resolve().parent
APP_DIR = HW6_DIR / "ankle-activity-wearable-nn"
APP_PYTHON = APP_DIR / "python"
if str(APP_PYTHON) not in sys.path:
    sys.path.insert(0, str(APP_PYTHON))

DATA_DIR = HW6_DIR / "data"                     # data/<person>/<activity>_<timestamp>.csv

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_STATE = "ME193/hudson/hw6/state"
TOPIC_RAW = "ME193/hudson/hw6/raw"
TOPIC_CMD = "ME193/hudson/hw6/cmd"

TRIM_S = 1.5             # seconds dropped at the start/end of each recording (getting into / out of the activity)
MAX_GAP_ROWS = 10        # a bigger hole in a recording splits it into separate pieces
BOARD_TIMEOUT_S = 2.0    # GUI shows "board offline" after this long without a state message

STATE_COLORS = {"standing": "#9e9e9e", "walking": "#2e9d4f", "running": "#d63a2f", "biking": "#2f6fd6"}
