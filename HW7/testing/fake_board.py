"""Run the board's real python/main.py on the laptop with a simulated sketch.

The fake Bridge serves synthetic walking in real time (stand, walk the
hallway there and back, stand, ...), packed exactly like the sketch's
get_samples(), and prints what set_display() would scroll plus the TRUE step
count, so the GUI's count and its ± range can be compared with the truth.
Don't run it while the real board app is running unless you give it its own
topics (--topic-prefix).

    .venv/Scripts/python HW7/testing/fake_board.py
    .venv/Scripts/python HW7/testing/fake_board.py --model m.json --topic-prefix test/hw7
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
import config
import stepnet as S


class FakeSketch:
    def __init__(self, seed, minutes=30):
        rng = np.random.default_rng(seed)
        parts = []
        while sum(p[1] if p[0] == "stand" else p[1] / p[2] for p in parts) < minutes * 60:
            pace = ["slow", "normal", "fast"][int(rng.integers(3))]
            parts += [("stand", 8.0)] + synth.walk_parts(pace, 2)[1:-1]
        parts.append(("stand", 8.0))
        t, phi, f, amp, yaw, strides, _ = synth.timeline(parts, rng)
        readings = synth.imu_from_phase(t, phi, f, amp, yaw, rng)
        self.rows = synth.rows_counts(synth.to_counts(readings))
        self.stride_times = strides
        self.t0 = time.monotonic()
        self.sent = 0
        self.shown = None
        self.last_truth = 0.0
        print(f"[fake sketch] {len(self.rows) / S.SAMPLE_HZ / 60:.0f} min of synthetic walking ready")

    def get_samples(self):
        now = time.monotonic() - self.t0
        due = min(int(now * S.SAMPLE_HZ), len(self.rows))
        n = min(6, due - self.sent)
        if n <= 0:
            return synth.encode_reply(self.sent, self.rows[:0])
        reply = synth.encode_reply(self.sent, self.rows[self.sent:self.sent + n])
        self.sent += n
        if now - self.last_truth > 10:
            self.last_truth = now
            true_steps = 2 * int(np.sum(self.stride_times < self.sent / S.SAMPLE_HZ))
            print(f"[fake sketch] TRUE steps since start: {true_steps} (reset in the GUI resets only the board)")
        return reply

    def set_display(self, text, mode):
        if text != self.shown:
            self.shown = text
            print(f"[fake sketch] LED matrix scrolls {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", help="use this model instead of the app's stepnet_model.json")
    ap.add_argument("--topic-prefix", default="ME193/hudson/hw7",
                    help="MQTT topic prefix; use a different one so the real board's messages don't mix in")
    ap.add_argument("--broker", help="MQTT broker address (default: the broker on the board, via config.find_broker)")
    args = ap.parse_args()
    # main.py uses 172.17.0.1, which only means "this board" inside the board's Docker.
    broker = args.broker or config.find_broker()
    if broker is None:
        raise SystemExit("No MQTT broker found at the board's addresses (tools/board_secrets.py HOSTS); "
                         "pass --broker <address>")

    sketch = FakeSketch(args.seed)

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

    # Copy the app code to a temp folder so another model can be swapped in.
    tmp = Path(tempfile.mkdtemp(prefix="hw7_fake_board_"))
    for name in ("main.py", "stepnet.py"):
        shutil.copy(config.APP_PYTHON / name, tmp / name)
    main_py = tmp / "main.py"
    code = main_py.read_text(encoding="utf-8").replace('"ME193/hudson/hw7/', f'"{args.topic_prefix}/')
    code = code.replace('BROKER = "172.17.0.1"', f'BROKER = "{broker}"')
    main_py.write_text(code, encoding="utf-8")
    model = Path(args.model) if args.model else S.MODEL_FILE
    if model.exists():
        shutil.copy(model, tmp / S.MODEL_FILE.name)
    print(f"[fake board] MQTT broker {broker}, topics {args.topic_prefix}/..., model {model if model.exists() else 'none'}")
    sys.path = [p for p in sys.path if Path(p).resolve() != config.APP_PYTHON.resolve()]
    sys.path.insert(0, str(tmp))
    sys.modules.pop("stepnet", None)
    runpy.run_path(str(tmp / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
