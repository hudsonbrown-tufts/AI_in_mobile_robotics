"""Laptop-side settings for HW7. Importing this also makes the board's
stepnet.py importable (`import stepnet`), so the laptop scripts use the exact
same network and step-adding code as the board."""

import sys
from pathlib import Path

HW7_DIR = Path(__file__).resolve().parent
APP_DIR = HW7_DIR / "step-counter-wearable"
APP_PYTHON = APP_DIR / "python"
if str(APP_PYTHON) not in sys.path:
    sys.path.insert(0, str(APP_PYTHON))

PERSON = "hudson"                               # HW7 only has to work for Hudson
DATA_DIR = HW7_DIR / "data"                     # data/hudson/<kind>_<time>.csv + .json
FIGURES_DIR = HW7_DIR / "figures"
MODEL_HISTORY_DIR = HW7_DIR / "model_history"
TRAINING_REPORT = HW7_DIR / "training_report.json"

# Extra data reused from HW6 (same person, same mounting): standing = 0 steps,
# walking = cadence only (it wasn't counted, so it has no distance).
HW6_DATA_DIR = HW7_DIR.parent / "HW6" / "data" / "hudson"
HW6_EXTRA = {"standing_*.csv": "standing", "walking_*.csv": "walk_uncounted"}

# MQTT broker: the Mosquitto broker that runs in Docker on the UNO Q (set up
# in HW6, see HW6/README.md section 7). The laptop reaches it at the board's
# address, so it tries the same addresses as tools/deploy.py (HOSTS in
# tools/board_secrets.py). The board's own app uses 172.17.0.1 (see main.py).
PORT = 1883
TOOLS_DIR = HW7_DIR.parent / "tools"


def broker_candidates():
    """The board's addresses from tools/board_secrets.py (or the example file)."""
    import importlib.util
    for name in ("board_secrets.py", "board_secrets.example.py"):
        path = TOOLS_DIR / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("_hw7_board_secrets", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return list(module.HOSTS)
    return []


def find_broker(timeout=2.0):
    """The first board address whose broker port answers, or None."""
    import socket
    for host in broker_candidates():
        try:
            socket.create_connection((host, PORT), timeout=timeout).close()
            return host
        except OSError:
            continue
    return None


TOPIC_STATE = "ME193/hudson/hw7/state"
TOPIC_RAW = "ME193/hudson/hw7/raw"
TOPIC_CMD = "ME193/hudson/hw7/cmd"

MAX_GAP_ROWS = 10        # a bigger hole in a recording splits it into separate pieces
BOARD_TIMEOUT_S = 2.0    # GUI shows "board offline" after this long without a state message
