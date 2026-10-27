"""Run the board's real python/main.py on the laptop with a simulated sketch.

The fake Bridge serves synthetic IMU rows in real time, packed exactly like
the sketch's get_samples(), and prints what set_display() would scroll. The
real MQTT topics are used, so gui.py works against it. Don't run it while the
real board app is running (both would publish on the same topics).

    .venv/Scripts/python HW6/testing/fake_board.py                    # cycles activities every 15 s
    .venv/Scripts/python HW6/testing/fake_board.py --activity running
    .venv/Scripts/python HW6/testing/fake_board.py --part1 p1.json --part2 p2.json

By default it uses the app's saved part1_calibration.json / part2_model.json.
"""

import argparse
import runpy
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

import synth
import classifiers as C
import config


class FakeSketch:
    def __init__(self, activity, period_s, person_seed, control_file=None):
        self.activity = activity
        self.period_s = period_s
        self.control_file = Path(control_file) if control_file else None
        self.person = synth.make_person(person_seed)
        self.rng = np.random.default_rng(0)
        self.t0 = time.monotonic()
        self.next_reading = 0          # index of the next 200 Hz reading to generate
        self.pending = np.zeros((0, 12), dtype=np.int64)
        self.seq = 0
        self.shown = None

    def label(self, t):
        if self.control_file:                 # a test script writes the activity to simulate here
            try:
                text = self.control_file.read_text().strip()
                if text in C.STATES:
                    return text
            except OSError:
                pass
        if self.activity:
            return self.activity
        return C.STATES[int(t // self.period_s) % len(C.STATES)]

    def get_samples(self):
        now = time.monotonic() - self.t0
        upto = int(now * synth.FS) // 4 * 4
        if upto > self.next_reading:
            t = np.arange(self.next_reading, upto) / synth.FS
            label = self.label(t[0])
            if label != getattr(self, "_label", None):
                self._label = label
                print(f"[fake sketch] now simulating: {label}")
            counts = synth.to_counts(synth.simulate(label, t, self.person, self.rng))
            self.pending = np.vstack([self.pending, synth.rows_counts(counts)])
            self.next_reading = upto
        n = min(6, len(self.pending))
        reply = synth.encode_reply(self.seq, self.pending[:n])
        self.pending = self.pending[n:]
        self.seq = (self.seq + n) % 65536
        return reply

    def set_display(self, word, mode):
        if (word, mode) != self.shown:
            self.shown = (word, mode)
            print(f"[fake sketch] LED matrix scrolls {word!r} (Part {mode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity", choices=C.STATES)
    ap.add_argument("--period", type=float, default=15.0, help="seconds per activity when cycling")
    ap.add_argument("--person", type=int, default=7, help="synthetic person seed")
    ap.add_argument("--part1", help="use this Part 1 calibration instead of the app's")
    ap.add_argument("--part2", help="use this Part 2 model instead of the app's")
    ap.add_argument("--control-file", help="simulate the activity named in this file (for scripted tests)")
    ap.add_argument("--topic-prefix", default="ME193/hudson/hw6",
                    help="MQTT topic prefix; use a different one so the real board's messages don't mix in")
    args = ap.parse_args()

    sketch = FakeSketch(args.activity, args.period, args.person, args.control_file)

    class Bridge:
        @staticmethod
        def call(name, *a):
            return getattr(sketch, name)(*a)

    class App:
        @staticmethod
        def run(user_loop):
            try:
                while True:
                    user_loop()
            except KeyboardInterrupt:
                raise SystemExit(0)

    pkg = types.ModuleType("arduino")
    utils = types.ModuleType("arduino.app_utils")
    utils.App, utils.Bridge = App, Bridge
    pkg.app_utils = utils
    sys.modules["arduino"], sys.modules["arduino.app_utils"] = pkg, utils

    # Copy the app code to a temp folder so other calibration files can be swapped in.
    tmp = Path(tempfile.mkdtemp(prefix="hw6_fake_board_"))
    for name in ("main.py", "classifiers.py"):
        shutil.copy(config.APP_PYTHON / name, tmp / name)
    main_py = tmp / "main.py"
    main_py.write_text(main_py.read_text(encoding="utf-8").replace('"ME193/hudson/hw6/', f'"{args.topic_prefix}/'),
                       encoding="utf-8")
    for src, dst in ((args.part1, C.PART1_CALIBRATION_FILE), (args.part2, C.PART2_MODEL_FILE)):
        src = Path(src) if src else dst
        if src.exists():
            shutil.copy(src, tmp / dst.name)
    sys.path = [p for p in sys.path if Path(p).resolve() != config.APP_PYTHON.resolve()]
    sys.path.insert(0, str(tmp))
    sys.modules.pop("classifiers", None)
    runpy.run_path(str(tmp / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
