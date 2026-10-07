"""Put the Midterm folder on sys.path so wii/ can reuse the ping pong modules
(paddle_imu.connect and haptics, vision.Vision for the camera and pose)."""

import sys
from pathlib import Path

MIDTERM = Path(__file__).resolve().parent.parent
if str(MIDTERM) not in sys.path:
    sys.path.insert(0, str(MIDTERM))
