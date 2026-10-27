"""Laptop-side settings for HW6. Importing this also makes the board's
classifiers.py importable (`import classifiers`), so the laptop scripts use the
exact same decision code as the board."""

import sys
from pathlib import Path

HW6_DIR = Path(__file__).resolve().parent
APP_DIR = HW6_DIR / "ankle-activity-wearable"
APP_PYTHON = APP_DIR / "python"
if str(APP_PYTHON) not in sys.path:
    sys.path.insert(0, str(APP_PYTHON))

DATA_DIR = HW6_DIR / "data"                     # data/<person>/<activity>_<timestamp>.csv

# MQTT broker: our own Mosquitto, running in Docker on the UNO Q itself
# (container "mqtt-broker", see README section 7). The laptop reaches it at the
# board's address, so it tries the same addresses as tools/deploy.py (HOSTS in
# tools/board_secrets.py). The board's own app uses 172.17.0.1 (see main.py).
PORT = 1883
TOOLS_DIR = HW6_DIR.parent / "tools"


def broker_candidates():
    """The board's addresses from tools/board_secrets.py (or the example file)."""
    import importlib.util
    for name in ("board_secrets.py", "board_secrets.example.py"):
        path = TOOLS_DIR / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("_hw6_board_secrets", path)
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


TOPIC_STATE = "ME193/hudson/hw6/state"
TOPIC_RAW = "ME193/hudson/hw6/raw"
TOPIC_CMD = "ME193/hudson/hw6/cmd"

TRIM_S = 1.5             # seconds dropped at the start/end of each recording (getting into / out of the activity)
MAX_GAP_ROWS = 10        # a bigger hole in a recording splits it into separate pieces
BOARD_TIMEOUT_S = 2.0    # GUI shows "board offline" after this long without a state message

STATE_COLORS = {"standing": "#9e9e9e", "walking": "#2e9d4f", "running": "#d63a2f", "biking": "#2f6fd6"}
