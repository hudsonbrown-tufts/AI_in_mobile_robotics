"""Calibration bookkeeping for the GUI: what's installed, installing, history, deploying.

The calibration the board uses is the pair of files in the app:
    ankle-activity-wearable/python/part1_calibration.json   (PART 1)
    ankle-activity-wearable/python/part2_model.json         (PART 2)
They only change when a new calibration is *installed*. A calibration that is
started but cancelled or discarded changes nothing. Every install is recorded in
calibration_history/<time>/:
    installed/   the files that were installed
    replaced/    the files that were in use before (if any)
so the first calibration is never lost and any older one can be restored.
"""

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import config
import classifiers as C

HISTORY_DIR = config.HW6_DIR / "calibration_history"
REPO_DIR = config.HW6_DIR.parent
APP_NAME = config.APP_DIR.name
FILES = {"part1": C.PART1_CALIBRATION_FILE, "part2": C.PART2_MODEL_FILE}


def summary(part, data):
    """The small, readable part of a calibration file (Part 2 without its weights)."""
    if data is None:
        return None
    if part == "part2":
        return {k: v for k, v in data.items() if k != "weights"}
    return {k: v for k, v in data.items() if k not in ("all_directions", "peak_stats_g")}


def read_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def current():
    """{"part1": summary or None, "part2": summary or None} of the installed calibration."""
    return {part: summary(part, read_file(path)) for part, path in FILES.items()}


def install(part1, part2):
    """Make a new calibration the one in use. Returns its history folder."""
    folder = HISTORY_DIR / time.strftime("%Y-%m-%d_%H%M%S")
    (folder / "installed").mkdir(parents=True, exist_ok=False)
    for part, path in FILES.items():
        if path.exists():
            (folder / "replaced").mkdir(exist_ok=True)
            shutil.copy2(path, folder / "replaced" / path.name)
    for (part, path), data in zip(FILES.items(), (part1, part2)):
        text = json.dumps(data, indent=2) if part == "part1" else json.dumps(data, separators=(",", ":"))
        (folder / "installed" / path.name).write_text(text)
        path.write_text(text)
    return folder


def history():
    """Installed calibrations, newest first: list of (folder, part1 summary, part2 summary)."""
    out = []
    for folder in sorted(HISTORY_DIR.glob("*/installed"), reverse=True):
        out.append((folder.parent, summary("part1", read_file(folder / FILES["part1"].name)),
                    summary("part2", read_file(folder / FILES["part2"].name))))
    return out


def restore(folder):
    """Re-install an older calibration from its history folder (current one is kept in history)."""
    part1 = read_file(Path(folder) / "installed" / FILES["part1"].name)
    part2 = read_file(Path(folder) / "installed" / FILES["part2"].name)
    if part1 is None or part2 is None:
        raise FileNotFoundError(f"{folder} doesn't have both calibration files")
    return install(part1, part2)


class Job:
    """Run fn(job) on a background thread; the GUI polls job.lines / job.done.

    fn reports progress with job.log(text) and should check job.stop_requested.
    """

    def __init__(self, fn):
        self.lines = queue.Queue()
        self.done = False
        self.result = None
        self.error = None
        self.stop_requested = False
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn):
        try:
            self.result = fn(self)
        except Exception as exc:          # shown in the GUI
            self.error = exc
        finally:
            self.done = True

    def log(self, text=""):
        for line in str(text).splitlines() or [""]:
            self.lines.put(line)

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                return out


def deploy(job):
    """Run tools/deploy.py for the app, streaming its output to the job."""
    cmd = [sys.executable, str(REPO_DIR / "tools" / "deploy.py"), APP_NAME]
    job.log("$ " + " ".join(cmd))
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", creationflags=flags)
    for line in proc.stdout:
        job.log(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"deploy.py exited with code {code}; the board still has its old calibration")
    return code
