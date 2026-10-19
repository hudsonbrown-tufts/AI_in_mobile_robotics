"""Setup walkthrough: shown when sumo.py starts (and on the `w` key).

It lists what is already calibrated and asks whether to keep it or start fresh.
Starting fresh moves the saved calibration files into calibration_backup/<time>/
(nothing is deleted), then walks through every step with on-screen instructions:

    1. RING          ring empty -> find the tape, record the empty ring
    2. CAMERA ANGLE  robot at 3 spots in the ring -> perspective correction
    3. COLOR SENSOR  sensor on the gray floor, then on the tape
    4. DRIVING       robot near the center -> it drives, turns and drives back to learn
                     which way is forward, which way it turns, wheels and gyro

Each step moves on by itself when it succeeds; if it fails, the reason is shown and
ENTER tries again. ENTER = do the step, S = skip it, Esc = leave the walkthrough.
"""

import shutil
import textwrap
from datetime import datetime

import config
from robot_io import BLUE, GRAY, ColorClassifier

PANEL_CHARS = 42
SPOTS = ["near the LEFT edge of the picture", "near the RIGHT edge of the picture",
         "near the FAR edge (top of the picture)"]
FILES = [config.RING_CALIBRATION_PATH, config.RING_BACKGROUND_PATH,
         config.COLOR_CALIBRATION_PATH, config.DRIVE_CALIBRATION_PATH]


def drive_calibration_ok():
    """Saved, and made by the current `d` (older files could have a backwards gyro)."""
    try:
        import json
        return "swap_sides" in json.loads(config.DRIVE_CALIBRATION_PATH.read_text())
    except (OSError, ValueError):
        return False


def calibration_status(app):
    v = app.vision
    return {
        "ring": v.ring is not None and v.opponent is not None and v.opponent.full is not None,
        "perspective": v.ring is not None and v.ring.camera is not None,
        "color": BLUE in app.robot.classifier.centroids and GRAY in app.robot.classifier.centroids,
        "drive": drive_calibration_ok(),
    }


NAMES = {"ring": "ring", "perspective": "camera angle", "color": "color sensor", "drive": "driving"}


class SetupWizard:
    def __init__(self, app, fresh=False):
        self.app = app
        self.active = True
        self.stage = "ask"
        self.steps = []
        self.i = 0
        self.sub = 0              # spot number (camera angle) or surface (color)
        self.busy = False         # waiting for the current action to finish
        self.note = ""            # result of the last attempt
        self.seen_events = len(app.vision.events)
        if fresh:
            self.start(fresh=True)

    # ---------- flow ----------

    def start(self, fresh):
        if fresh:
            self._forget_everything()
            self.steps = ["ring", "perspective", "color", "drive"]
        else:
            st = calibration_status(self.app)
            self.steps = [k for k in ("ring", "perspective", "color", "drive") if not st[k]]
        self.stage, self.i, self.sub, self.busy, self.note = "steps", 0, 0, False, ""
        if not self.steps:
            self.stage = "done"

    def _forget_everything(self):
        folder = config.HERE / "calibration_backup" / f"{datetime.now():%Y%m%d_%H%M%S}"
        moved = []
        for f in FILES:
            if f.exists():
                folder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(folder / f.name))
                moved.append(f.name)
        if moved:
            print(f"[setup] old calibration moved to {folder}: {', '.join(moved)}")
        app = self.app
        app.vision.reset_calibration()
        app.robot.classifier = ColorClassifier()
        app.robot.drive.clear()                       # same dict the fusion and motors use
        app.robot.drive.update(config.DEFAULT_DRIVE)
        app.brain.fusion.drive_check.reset()
        app.brain.reset()

    def _advance(self, note=""):
        self.i += 1
        self.sub = 0
        self.busy = False
        self.note = note
        if self.i >= len(self.steps):
            self.stage = "done"

    @property
    def step(self):
        return self.steps[self.i] if self.stage == "steps" else None

    # ---------- keys ----------

    def on_key(self, k, now):
        """True if the walkthrough used the key."""
        if self.app.click_mode:                       # clicking tape points: Enter/Esc belong to the click mode
            if k == 13 and self.step == "ring":
                self.busy, self.note = True, ""       # watch for the ring + empty-ring result
            return k not in (13, 27, ord("q"), ord("Q"))
        if k in (ord("m"), ord("M")):
            return False                              # the normal click-the-tape mode, from any step
        if k == 27:                                   # Esc: leave the walkthrough
            self.active = False
            self.app.message = ""
            return True
        if self.stage == "ask":
            if k == 13:
                self.start(fresh=False)
                return True
            if k in (ord("n"), ord("N")):
                self.start(fresh=True)
                return True
            return k not in (ord("q"), ord("Q"))
        if self.stage == "done":
            if k == ord(" ") or k == 13:
                self.active = False
                return k == 13                        # space falls through and starts the match
            return False
        if k in (ord("s"), ord("S")) and not self.busy:
            self._advance(f"skipped {NAMES[self.step]}")
            return True
        if k == 13 and not self.busy:
            self._do(now)
            return True
        return k not in (ord("q"), ord("Q"))          # swallow other keys during setup

    def _do(self, now):
        app, step = self.app, self.step
        self.note = ""
        if step == "ring":
            app.vision.request_calibration("auto")
        elif step == "perspective":
            app.vision.request_calibration("perspective")
        elif step == "color":
            label = GRAY if self.sub == 0 else BLUE
            app.color_result = None
            app.color_stage = None
            app.color_samples = (now + 0.6, label, [])
        elif step == "drive":
            if not app.start_drive_calibration(now):
                self.note = app.message
                return
        self.busy = True

    # ---------- progress ----------

    def update(self, now):
        if not self.active:
            return
        app, step = self.app, self.step
        events = app.vision.events[self.seen_events:]
        self.seen_events = len(app.vision.events)
        if not self.busy:
            return
        if step == "ring":
            for kind, ok, msg in events:
                if kind == "ring" and not ok:
                    self.busy, self.note = False, msg + ". ENTER = try again, M = click the tape yourself"
                elif kind == "background" and ok:
                    self._advance("ring found and empty ring recorded")
        elif step == "perspective":
            for kind, ok, msg in events:
                if kind == "perspective":
                    self.busy = False
                    if ok:
                        self.sub += 1
                        self.note = msg
                        if self.sub >= len(SPOTS):
                            self._advance("camera angle corrected")
                    else:
                        self.note = msg
        elif step == "color" and app.color_result is not None:
            label, ok = app.color_result
            app.color_result = None
            self.busy = False
            if not ok:
                self.note = "no readings from the color sensor: is it connected?"
            elif label == GRAY:
                self.sub = 1
            else:
                self._advance("color sensor calibrated")
        elif step == "drive" and app.drive_result is not None:
            ok, msg = app.drive_result
            app.drive_result = None
            self.busy = False
            if ok:
                self._advance("driving " + msg)
            else:
                self.note = msg + ". ENTER = try again"

    # ---------- text ----------

    @property
    def text(self):
        if self.app.click_mode:
            n = len(self.app.clicks)
            lines = ["CLICK THE TAPE", f"{n} point{'s' if n != 1 else ''} clicked (need 5+, spread all the way "
                     "around the ring, on the middle of the tape).",
                     "ENTER = use these points" if n >= 5 else "Keep clicking...",
                     "Esc = cancel clicking"]
        elif self.stage == "ask":
            st = calibration_status(self.app)
            have = ", ".join(NAMES[k] + (" OK" if v else " MISSING") for k, v in st.items())
            if all(st.values()):
                lines = ["SETUP", f"Saved: {have}.", "ENTER = keep it", "N = start fresh and recalibrate everything"]
            else:
                lines = ["SETUP", f"Saved: {have}.", "ENTER = set up the missing steps",
                         "N = start fresh and redo everything", "Esc = skip setup"]
        elif self.stage == "done":
            lines = ["SETUP DONE", self.note, "Press SPACE to start a match.",
                     "Shake the controller to attack; hold it still for defense."]
        else:
            n, total = self.i + 1, len(self.steps)
            step = self.step
            head = f"STEP {n}/{total}: {NAMES[step].upper()}"
            if step == "ring":
                body = ["Take everything out of the ring (robots, hands).",
                        "ENTER = find the tape and record the empty ring.",
                        "M = click points on the tape yourself instead."]
                if self.busy:
                    body = ["Finding the tape... keep the ring empty."]
            elif step == "perspective":
                body = [f"Spot {self.sub + 1} of {len(SPOTS)}: put the robot in the ring {SPOTS[self.sub]}, "
                        "both tags facing the camera.",
                        "ENTER = measure (keep still 2 s)."]
                if self.busy:
                    body = ["Watching the tags... don't move the robot."]
            elif step == "color":
                where = "the GRAY floor" if self.sub == 0 else "the BLUE tape"
                body = [f"Put the robot so its color sensor is over {where}.", "ENTER = record."]
                if self.busy:
                    body = ["Recording..."]
            else:
                body = ["Put the robot near the center, nothing within 30 cm.",
                        "ENTER = it drives forward, turns, and drives back."]
                if self.busy:
                    body = ["Driving... don't touch the robot."]
            lines = [head] + body + ([self.note] if self.note else []) + ["S = skip this step, Esc = leave setup"]
        out = []
        for line in lines:
            out += textwrap.wrap(line, PANEL_CHARS) or [""]
        return "\n".join(out)
