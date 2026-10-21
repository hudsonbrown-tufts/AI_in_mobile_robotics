"""Step-by-step calibration window (opened from gui.py with Shift+C, i.e. a capital C).

Steps: setup -> record standing -> walking -> running -> biking
       -> fit PART 1 (direction + thresholds) and PART 2 (neural network)
       -> compare with the current calibration -> install + deploy, or discard.

Nothing about the calibration in use changes until "Install" is pressed:
closing, cancelling or discarding keeps the current calibration. Installing
keeps the previous one in calibration_history/ (see calibration.py).
"""

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import config
import classifiers as C
import calibration
import dataset

INSTRUCTIONS = {
    "standing": ("Stand still, relaxed, weight on both feet.",
                 "Small shifts of weight and fidgeting are fine; that is what real standing looks like."),
    "walking": ("Walk at a normal pace (in a loop or back and forth).",
                "Change speed partway through: some slow walking, some brisk walking."),
    "running": ("Run or jog.",
                "Mix a light jog with a faster run so different running styles are covered."),
    "biking": ("Lie on your back, lift your legs and pedal in the air like riding a bike.",
               "Vary the pace: some slow pedaling, some fast."),
}
FONT = "Segoe UI"


class CalibrationWizard(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Calibrate the wearable")
        s = self.winfo_fpixels("1i") / 96
        self.geometry(f"{int(880 * s)}x{int(700 * s)}+{int(60 * s)}+{int(40 * s)}")
        self.transient(app)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.steps = ["setup"] + C.STATES + ["fit", "results"]
        self.index = 0
        self.session_files = {}            # activity -> saved recording path (this session)
        self.messages = {}                 # activity -> result of its last recording
        self.fit_job = None
        self.deploy_job = None
        self.new_cal = None                # (part1 json, part2 json) once fitted
        self.installed_folder = None

        self.person_var = tk.StringVar(value="hudson")
        self.seconds_var = tk.IntVar(value=60)
        self.delay_var = tk.IntVar(value=8)
        self.include_old_var = tk.BooleanVar(value=True)

        self.header = tk.Label(self, font=(FONT, 16, "bold"), anchor="w", padx=14, pady=10,
                               bg="#1a56c4", fg="white")
        self.header.pack(fill="x")
        self.body = ttk.Frame(self, padding=14)
        self.body.pack(fill="both", expand=True)
        footer = ttk.Frame(self, padding=(14, 6, 14, 12))
        footer.pack(fill="x")
        self.cancel_btn = ttk.Button(footer, text="Cancel (keep current calibration)", command=self.cancel)
        self.cancel_btn.pack(side="left")
        self.next_btn = ttk.Button(footer, text="Next >", command=self.next)
        self.next_btn.pack(side="right")
        self.back_btn = ttk.Button(footer, text="< Back", command=self.back)
        self.back_btn.pack(side="right", padx=6)

        self.show()
        self.after(100, self.poll)

    # ------------------------------------------------------------ navigation -
    def show(self):
        for w in self.body.winfo_children():
            w.destroy()
        step = self.steps[self.index]
        n = len(self.steps)
        titles = {"setup": "Setup", "fit": "Fitting Part 1 and Part 2", "results": "Results: install or discard"}
        title = titles.get(step, f"Record: {step}")
        self.header.config(text=f"Step {self.index + 1} of {n}:  {title}")
        getattr(self, f"_build_{'record' if step in C.STATES else step}")()
        self.update_buttons()

    def update_buttons(self):
        step = self.steps[self.index]
        recording = self.app.recording is not None
        fitting = self.fit_job is not None and not self.fit_job.done
        deploying = self.deploy_job is not None and not self.deploy_job.done
        can_next = {
            "setup": bool(self.person_var.get().strip()),
            "fit": self.new_cal is not None,
            "results": False,
        }.get(step, step in self.session_files) and not recording
        self.next_btn.config(state="normal" if can_next else "disabled")
        can_back = self.index > 0 and not (recording or fitting or deploying or self.installed_folder)
        self.back_btn.config(state="normal" if can_back else "disabled")
        self.next_btn.config(text="Next >")
        self.cancel_btn.config(text="Close" if self.installed_folder else "Cancel (keep current calibration)",
                               state="disabled" if deploying else "normal")

    def next(self):
        if self.index < len(self.steps) - 1:
            self.index += 1
            self.show()

    def back(self):
        if self.index > 0:
            if self.steps[self.index] == "results":
                self.index -= 1           # going back to "fit" re-runs the fit
                self.new_cal = None
            self.index -= 1
            self.show()

    def cancel(self):
        if self.deploy_job is not None and not self.deploy_job.done:
            return
        if self.installed_folder is None:
            if self.app.recording is not None:
                self.app.stop_recording(save=False)
            if self.fit_job is not None and not self.fit_job.done:
                self.fit_job.stop_requested = True
            if self.session_files:
                keep = messagebox.askyesnocancel(
                    "Cancel calibration",
                    "The current calibration stays as it is.\n\n"
                    f"Keep the {len(self.session_files)} recording(s) made in this session in HW6/data? "
                    "They will be used by future calibrations.\n\n"
                    "Yes = keep them,  No = delete them,  Cancel = go back to the wizard", parent=self)
                if keep is None:
                    return
                if not keep:
                    for path in self.session_files.values():
                        Path(path).unlink(missing_ok=True)
        self.app.wizard = None
        self.app.refresh_calibration_tab()
        self.destroy()

    # ------------------------------------------------------------ setup ------
    def _build_setup(self):
        b = self.body
        text = (
            "This walks you through recording the 4 activities, then fits Part 1 (the best single "
            "direction + 3 thresholds) and Part 2 (the neural network) from the recordings.\n\n"
            "Nothing changes on the board until you press Install at the end. If you cancel or "
            "discard, the current calibration stays in use. When you install, the old one is kept "
            "in HW6/calibration_history/ and can be restored."
        )
        tk.Label(b, text=text, font=(FONT, 11), justify="left", anchor="w", wraplength=820).pack(fill="x")

        tk.Label(b, text="Before you start: the board is on the OUTSIDE of the RIGHT ankle, "
                         "the MPU-6050's +X arrow pointing UP the shin, strap tight.",
                 font=(FONT, 11, "bold"), fg="#c77800", justify="left", wraplength=820,
                 anchor="w").pack(fill="x", pady=(10, 6))

        form = ttk.Frame(b)
        form.pack(anchor="w", pady=6)
        rows = [("Who is wearing it", ttk.Entry(form, textvariable=self.person_var, width=20)),
                ("Seconds per activity", ttk.Spinbox(form, from_=20, to=300, increment=10,
                                                     textvariable=self.seconds_var, width=8)),
                ("Countdown before each (s)", ttk.Spinbox(form, from_=0, to=60,
                                                          textvariable=self.delay_var, width=8))]
        for i, (label, widget) in enumerate(rows):
            tk.Label(form, text=label, font=(FONT, 11)).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=3)
            widget.grid(row=i, column=1, sticky="w", pady=3)
        self.person_var.trace_add("write", lambda *a: self.update_buttons())

        recs = dataset.load_recordings(config.DATA_DIR)
        people = sorted({r.person for r in recs})
        old = (f"{len(recs)} earlier recordings from {', '.join(people)}" if recs else "no earlier recordings yet")
        ttk.Checkbutton(b, text=f"Also use the earlier recordings in HW6/data ({old}). "
                                "Recommended: more people = works better on someone new.",
                        variable=self.include_old_var).pack(anchor="w", pady=(8, 4))

        cur = calibration.current()
        tk.Label(b, text="Calibration in use now:", font=(FONT, 11, "bold"), anchor="w").pack(fill="x", pady=(10, 0))
        tk.Label(b, text=describe(cur), font=("Consolas", 10), justify="left", anchor="w").pack(fill="x")

        self.board_label = tk.Label(b, font=(FONT, 11, "bold"), anchor="w")
        self.board_label.pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------ record -----
    def _build_record(self):
        activity = self.steps[self.index]
        b = self.body
        what, tip = INSTRUCTIONS[activity]
        color = config.STATE_COLORS[activity]
        tk.Label(b, text=activity.upper(), font=(FONT, 30, "bold"), fg="white", bg=color,
                 pady=6).pack(fill="x")
        tk.Label(b, text=what, font=(FONT, 14, "bold"), justify="left", anchor="w",
                 wraplength=820).pack(fill="x", pady=(12, 2))
        tk.Label(b, text=tip, font=(FONT, 11), justify="left", anchor="w", wraplength=820).pack(fill="x")
        tk.Label(b, text=f"Press Start, get into position during the {self.delay_var.get()} s countdown, and keep "
                         f"going until the second beep ({self.seconds_var.get()} s). You can stop early.",
                 font=(FONT, 10), fg="#555555", justify="left", anchor="w", wraplength=820).pack(fill="x", pady=(8, 0))

        btns = ttk.Frame(b)
        btns.pack(anchor="w", pady=14)
        self.start_btn = tk.Button(btns, text="Start recording", font=(FONT, 13, "bold"), bg="#d63a2f",
                                   fg="white", padx=16, command=lambda: self.start(activity))
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(btns, text="Stop early and save", font=(FONT, 12), padx=10,
                                  command=lambda: self.app.stop_recording(save=True))
        self.stop_btn.pack(side="left", padx=8)

        self.rec_label = tk.Label(b, font=(FONT, 20, "bold"), anchor="w")
        self.rec_label.pack(fill="x", pady=(6, 4))
        self.progress = ttk.Progressbar(b, maximum=1.0)
        self.progress.pack(fill="x")
        self.board_label = tk.Label(b, font=(FONT, 10), anchor="w")
        self.board_label.pack(fill="x", pady=(10, 0))
        self.refresh_record()

    def start(self, activity):
        old = self.session_files.pop(activity, None)    # "Record again" replaces this session's take
        if old:
            Path(old).unlink(missing_ok=True)
        self.messages.pop(activity, None)
        self.app.start_recording(self.person_var.get().strip(), activity, int(self.seconds_var.get()),
                                 int(self.delay_var.get()),
                                 on_done=lambda path, msg: self.recorded(activity, path, msg))
        self.refresh_record()
        self.update_buttons()

    def recorded(self, activity, path, msg):
        """Called by the main window when a recording ends (path is None if nothing was saved)."""
        if path:
            self.session_files[activity] = path
        self.messages[activity] = msg
        if self.winfo_exists() and self.steps[self.index] == activity:
            self.refresh_record()
            self.update_buttons()

    def refresh_record(self):
        activity = self.steps[self.index]
        rec = self.app.recording
        done = activity in self.session_files
        self.start_btn.config(state="disabled" if rec else "normal",
                              text="Record again" if done else "Start recording")
        self.stop_btn.config(state="normal" if rec and rec["started"] else "disabled")
        now = time.monotonic()
        if rec and not rec["started"]:
            self.rec_label.config(text=f"Get ready... starting in {max(0, rec['start_at'] - now):.0f} s",
                                  fg="#c77800")
            self.progress["value"] = 0
        elif rec:
            elapsed = now - rec["t0"]
            self.rec_label.config(text=f"RECORDING  {elapsed:.0f} / {rec['duration']} s", fg="#d63a2f")
            self.progress["value"] = min(1.0, elapsed / rec["duration"])
        elif done:
            self.rec_label.config(text=f"Done: {self.messages.get(activity, 'saved')}. Press Next.", fg="#2e9d4f")
            self.progress["value"] = 1.0
        else:
            msg = self.messages.get(activity)
            self.rec_label.config(text=msg or "Not recorded yet.", fg="#d63a2f" if msg else "black")
            self.progress["value"] = 0

    # ------------------------------------------------------------ fit --------
    def _build_fit(self):
        b = self.body
        self.fit_label = tk.Label(b, text="", font=(FONT, 13, "bold"), anchor="w")
        self.fit_label.pack(fill="x")
        self.log_box = make_log(b)
        self.new_cal = None
        files = set(map(str, self.session_files.values()))
        include_old = self.include_old_var.get()

        def work(job):
            recs = dataset.load_recordings(config.DATA_DIR)
            if not include_old:
                recs = [r for r in recs if str(r.path) in files]
            job.log(f"Using {len(recs)} recordings ({'all of HW6/data' if include_old else 'this session only'})")
            job.log("\n===== PART 1: choosing the direction and thresholds =====")
            import calibrate_part1
            part1 = calibrate_part1.calibrate(recs, log=job.log)
            job.log("\n===== PART 2: training the neural network (a few minutes) =====")
            import train_part2              # imports PyTorch, so only when needed
            part2 = train_part2.train_model(recs, log=job.log,
                                            should_stop=lambda: job.stop_requested)
            return part1, part2

        self.fit_job = calibration.Job(work)
        self.fit_label.config(text="Fitting... (Part 2 training takes a few minutes)", fg="#c77800")

    # ------------------------------------------------------------ results ----
    def _build_results(self):
        b = self.body
        part1, part2 = self.new_cal
        cur = calibration.current()
        tk.Label(b, text="Compare the new calibration with the one in use. Accuracies are on held-out data "
                         "(another person when there are several people), so they estimate how it does on "
                         "someone new.", font=(FONT, 10), justify="left", anchor="w", wraplength=820).pack(fill="x")

        grid = ttk.Frame(b)
        grid.pack(fill="x", pady=8)
        new = {"part1": calibration.summary("part1", part1), "part2": calibration.summary("part2", part2)}
        rows = comparison_rows(cur, new)
        for c, head in enumerate(["", "In use now", "New"]):
            tk.Label(grid, text=head, font=(FONT, 11, "bold"), anchor="w").grid(row=0, column=c, sticky="w", padx=6)
        for r, (name, old_v, new_v) in enumerate(rows, start=1):
            tk.Label(grid, text=name, font=(FONT, 10, "bold"), anchor="w").grid(row=r, column=0, sticky="w", padx=6)
            tk.Label(grid, text=old_v, font=("Consolas", 10), anchor="w").grid(row=r, column=1, sticky="w", padx=6)
            tk.Label(grid, text=new_v, font=("Consolas", 10), anchor="w").grid(row=r, column=2, sticky="w", padx=6)

        btns = ttk.Frame(b)
        btns.pack(anchor="w", pady=8)
        self.install_btn = tk.Button(btns, text="Install and deploy to the board", font=(FONT, 12, "bold"),
                                     bg="#2e9d4f", fg="white", padx=12, command=self.install)
        self.install_btn.pack(side="left")
        self.discard_btn = tk.Button(btns, text="Discard (keep the current calibration)", font=(FONT, 12),
                                     padx=10, command=self.discard)
        self.discard_btn.pack(side="left", padx=8)
        self.retry_btn = tk.Button(btns, text="Retry deploy", font=(FONT, 12), padx=10, command=self.start_deploy)
        self.result_label = tk.Label(b, text="", font=(FONT, 12, "bold"), anchor="w", justify="left", wraplength=820)
        self.result_label.pack(fill="x")
        self.log_box = make_log(b, height=10)

    def install(self):
        part1, part2 = self.new_cal
        self.installed_folder = calibration.install(part1, part2)
        self.install_btn.config(state="disabled")
        self.discard_btn.config(state="disabled")
        self.result_label.config(text=f"Installed (copy kept in {short_path(self.installed_folder)}). "
                                      "Deploying to the board...", fg="#c77800")
        self.app.refresh_calibration_tab()
        self.start_deploy()

    def start_deploy(self):
        self.retry_btn.pack_forget()
        self.deploy_job = calibration.Job(calibration.deploy)
        self.update_buttons()

    def discard(self):
        self.cancel()

    # ------------------------------------------------------------ polling ----
    def poll(self):
        if not self.winfo_exists():
            return
        step = self.steps[self.index]
        if hasattr(self, "board_label") and self.board_label.winfo_exists():
            online, text = self.app.board_status()
            self.board_label.config(text=text, fg="#2e9d4f" if online else "#d63a2f")
        if step in C.STATES and self.app.recording is not None:
            self.refresh_record()
        for job, done in ((self.fit_job, self.fit_finished), (self.deploy_job, self.deploy_finished)):
            if job is not None:
                lines = job.drain()
                if lines and hasattr(self, "log_box") and self.log_box.winfo_exists():
                    self.log_box.insert("end", "\n".join(lines) + "\n")
                    self.log_box.see("end")
                if job.done and not getattr(job, "handled", False):
                    job.handled = True
                    done(job)
        self.after(150, self.poll)

    def fit_finished(self, job):
        if job.stop_requested or not self.fit_label.winfo_exists():
            return
        if job.error:
            self.fit_label.config(text=f"Fitting failed: {job.error}", fg="#d63a2f")
        else:
            self.new_cal = job.result
            self.fit_label.config(text="Done. Press Next to compare with the current calibration.", fg="#2e9d4f")
        self.update_buttons()

    def deploy_finished(self, job):
        if job.error:
            self.result_label.config(
                text=f"Installed on this laptop, but deploying failed: {job.error}. Check that the board is on "
                     "and reachable (tools/board_secrets.py), then Retry deploy, or deploy later from the "
                     "Calibration tab.", fg="#d63a2f")
            self.retry_btn.pack(side="left")
        else:
            self.result_label.config(text="Installed and deployed. The board restarts the app with the new "
                                          "calibration in a few seconds. You can close this window.", fg="#2e9d4f")
        self.app.refresh_calibration_tab()
        self.update_buttons()


# ---------------------------------------------------------------- helpers ----
def make_log(parent, height=18):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, pady=(8, 0))
    box = tk.Text(frame, height=height, font=("Consolas", 9), wrap="none")
    sb = ttk.Scrollbar(frame, command=box.yview)
    box.config(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    box.pack(side="left", fill="both", expand=True)
    return box


def describe(cal):
    """Readable lines for a calibration summary {"part1": ..., "part2": ...}."""
    p1, p2 = cal.get("part1"), cal.get("part2")
    lines = []
    if p1:
        th = ", ".join(f"{t:.2f}" for t in p1["thresholds"])
        lines.append(f"Part 1: made {p1.get('created', '?')}, direction {p1['direction']}, "
                     f"thresholds {th} g, accuracy {p1.get('accuracy', 0):.0%} ({p1.get('scored_by', '?')})")
        lines.append(f"        people: {', '.join(p1.get('people', []))}")
    else:
        lines.append("Part 1: not calibrated yet")
    if p2:
        lines.append(f"Part 2: made {p2.get('created', '?')}, validation accuracy "
                     f"{p2.get('validation_accuracy', 0):.0%} ({p2.get('validation', '?')})")
        lines.append(f"        people: {', '.join(p2.get('people', []))}")
    else:
        lines.append("Part 2: not trained yet")
    return "\n".join(lines)


def comparison_rows(cur, new):
    def p1(c, key, fmt=str):
        return fmt(c["part1"][key]) if c.get("part1") else "-"

    def p2(c, key, fmt=str):
        return fmt(c["part2"][key]) if c.get("part2") else "-"

    pct = "{:.0%}".format
    th = lambda v: ", ".join(f"{t:.2f}" for t in v)  # noqa: E731
    order = lambda v: " < ".join(s[:4] for s in v)  # noqa: E731
    people = ", ".join
    return [
        ("Part 1 direction", p1(cur, "direction"), p1(new, "direction")),
        ("Part 1 order", p1(cur, "order", order), p1(new, "order", order)),
        ("Part 1 thresholds (g)", p1(cur, "thresholds", th), p1(new, "thresholds", th)),
        ("Part 1 accuracy", p1(cur, "accuracy", pct), p1(new, "accuracy", pct)),
        ("Part 1 scored by", p1(cur, "scored_by"), p1(new, "scored_by")),
        ("Part 2 accuracy", p2(cur, "validation_accuracy", pct), p2(new, "validation_accuracy", pct)),
        ("Part 2 validated by", p2(cur, "validation"), p2(new, "validation")),
        ("People", p2(cur, "people", people), p2(new, "people", people)),
        ("Made", p1(cur, "created"), p1(new, "created")),
    ]


def short_path(path):
    """Path relative to HW6/ when it's inside it, else the full path."""
    try:
        return str(Path(path).relative_to(config.HW6_DIR))
    except ValueError:
        return str(path)
