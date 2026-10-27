"""HW6 laptop GUI: shows what the ankle wearable decides and switches Part 1 / Part 2.

The board does all the classifying; this window only displays its MQTT
messages and sends the mode switch. Two tabs:
  Live         - current activity, Part 1 / Part 2 switch, both parts' details, live plot
  Calibration  - the calibration in use, whether the board has it, recordings, history

Shift+C (a capital C) opens the calibration wizard (calibration_wizard.py),
which records the 4 activities, fits Part 1 and Part 2, and installs + deploys
the result. Until it's installed, the current calibration stays in use.

    .venv/Scripts/python HW6/gui.py
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import config  # noqa: E402
import classifiers as C  # noqa: E402
import dataset  # noqa: E402
import calibration  # noqa: E402
import calibration_wizard  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

PLOT_SECONDS = 10
PLOT_ROWS = PLOT_SECONDS * C.SAMPLE_HZ
TICK_MS = 100
PLOT_EVERY_TICKS = 2           # plot redraw at 5 Hz
MODE_CONFIRM_S = 4.0           # give the board this long to confirm a mode switch
PART_NAMES = {1: "Part 1: peak acceleration (one direction)", 2: "Part 2: decision rules (all IMU outputs)"}
if sys.platform == "win32":
    try:  # sharp text with Windows display scaling
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

FRAME_TITLES = {1: " PART 1: peak acceleration, one direction", 2: " PART 2: decision rules, all IMU outputs"}


class WearableGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HW6 Ankle Activity Wearable")
        # Size for the screen's scaling (e.g. 150% on a laptop), capped to the screen.
        s = self.winfo_fpixels("1i") / 96
        w = min(int(1100 * s), int(self.winfo_screenwidth() * 0.95))
        h = min(int(860 * s), int(self.winfo_screenheight() * 0.9))
        self.geometry(f"{w}x{h}+10+10")

        self.inbox = queue.Queue()             # (kind, payload) from the MQTT thread
        self.state_msg = None
        self.state_time = 0.0
        self.board_mode = None
        self.requested_mode = None
        self.requested_at = 0.0
        self.plot_rows = deque(maxlen=PLOT_ROWS)
        self.last_raw_seq = None
        self.unwrapped_seq = 0
        self.recording = None                  # dict while counting down / recording
        self.wizard = None
        self.deploy_job = None
        self.tick_count = 0

        # MQTT: the broker runs on the board, so look for it at the board's
        # addresses on a background thread, retrying until it answers. Once
        # connected, paho reconnects (and mqttlib re-subscribes) by itself.
        self.mqtt = None
        self.broker = None
        self.closing = False
        threading.Thread(target=self._connect_mqtt, daemon=True).start()

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)
        live, cal = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(live, text="  Live  ")
        tabs.add(cal, text="  Calibration  ")
        self._build_live(live)
        self._build_calibration(cal)
        self.bind("<Key-C>", self.on_capital_c)       # Shift+C anywhere in the main window
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(TICK_MS, self.tick)

    # ------------------------------------------------------------ Live tab ---
    def _build_live(self, root):
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Mode:", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.mode_var = tk.IntVar(value=1)
        for m in (1, 2):
            tk.Radiobutton(top, text=PART_NAMES[m], variable=self.mode_var, value=m, indicatoron=0,
                           font=("Segoe UI", 11), padx=14, pady=6, selectcolor="#cfe3ff",
                           command=lambda m=m: self.request_mode(m)).pack(side="left", padx=4)
        self.mode_status = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.mode_status.pack(side="left", padx=10)

        self.state_label = tk.Label(root, text="waiting for board...", font=("Segoe UI", 54, "bold"),
                                    fg="white", bg="#555555", pady=14)
        self.state_label.pack(fill="x", padx=8)
        self.status_label = ttk.Label(root, text="", font=("Segoe UI", 10), padding=(8, 4))
        self.status_label.pack(fill="x")

        parts = ttk.Frame(root, padding=(8, 0))
        parts.pack(fill="x")
        parts.columnconfigure(0, weight=1)
        parts.columnconfigure(1, weight=1)

        self.p1_frame = tk.LabelFrame(parts, text=FRAME_TITLES[1],
                                      font=("Segoe UI", 10, "bold"), padx=8, pady=6, bd=3)
        self.p1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.p1_text = tk.Label(self.p1_frame, text="", font=("Consolas", 11), justify="left", anchor="nw")
        self.p1_text.pack(fill="both", expand=True)

        self.p2_frame = tk.LabelFrame(parts, text=FRAME_TITLES[2],
                                      font=("Segoe UI", 10, "bold"), padx=8, pady=6, bd=3)
        self.p2_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.p2_text = tk.Label(self.p2_frame, text="", font=("Consolas", 11), justify="left", anchor="nw")
        self.p2_text.pack(fill="x")

        self.fig = Figure(figsize=(10, 4.2), dpi=90)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, sharex=self.ax1)
        self.ax2b = self.ax2.twinx()
        self.fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.11, hspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _connect_mqtt(self):
        while not self.closing:
            host = config.find_broker()
            if host:
                client = MQTTClient(broker=host, port=config.PORT)
                client.subscribe(config.TOPIC_STATE, lambda t, p: self.inbox.put(("state", p)))
                client.subscribe(config.TOPIC_RAW, lambda t, p: self.inbox.put(("raw", p)))
                try:
                    client.connect()
                    if client.connected:
                        self.mqtt, self.broker = client, host
                        print(f"MQTT: connected to the broker at {host}:{config.PORT}")
                        return
                    client.disconnect()
                except Exception as exc:
                    print(f"MQTT: couldn't connect to {host}: {exc}")
            time.sleep(5)

    @property
    def mqtt_ok(self):
        return self.mqtt is not None and self.mqtt.connected

    def request_mode(self, mode):
        self.requested_mode = mode
        self.requested_at = time.monotonic()
        # Retained, so the board picks the mode up again after a restart.
        if self.mqtt is not None:
            self.mqtt.publish(config.TOPIC_CMD, f"mode:{mode}", retain=True)

    # ------------------------------------------------------ Calibration tab --
    def _build_calibration(self, root):
        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)
        hint = tk.Frame(frm, bg="#1a56c4", padx=12, pady=10)
        hint.pack(fill="x")
        tk.Label(hint, text="Press  Shift + C  (a capital C) to calibrate or recalibrate",
                 font=("Segoe UI", 15, "bold"), bg="#1a56c4", fg="white").pack(side="left")
        tk.Button(hint, text="Start calibration", font=("Segoe UI", 11), command=self.open_wizard).pack(side="right")
        tk.Label(frm, text="The calibration below stays in use until a new one is installed at the end of the "
                           "wizard; cancelling a recalibration keeps it. Every installed calibration is kept in "
                           "HW6/calibration_history/ and can be restored.",
                 font=("Segoe UI", 10), justify="left", anchor="w", wraplength=1000).pack(fill="x", pady=(8, 4))

        tk.Label(frm, text="In use (files on this laptop)", font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(fill="x", pady=(8, 0))
        self.cal_text = tk.Label(frm, font=("Consolas", 10), justify="left", anchor="w")
        self.cal_text.pack(fill="x")
        board_row = ttk.Frame(frm)
        board_row.pack(fill="x", pady=(6, 0))
        self.cal_board = tk.Label(board_row, font=("Segoe UI", 11, "bold"), anchor="w")
        self.cal_board.pack(side="left")
        self.deploy_btn = ttk.Button(board_row, text="Deploy to board", command=self.start_deploy)
        self.deploy_btn.pack(side="right")

        tk.Label(frm, text="Recordings in HW6/data (usable seconds)", font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(fill="x", pady=(12, 2))
        self.summary = ttk.Treeview(frm, columns=["person"] + C.STATES, show="headings", height=4)
        for c in ["person"] + C.STATES:
            self.summary.heading(c, text=c)
            self.summary.column(c, width=110, minwidth=80, anchor="center", stretch=True)
        self.summary.pack(fill="x")

        hist_row = ttk.Frame(frm)
        hist_row.pack(fill="x", pady=(12, 2))
        tk.Label(hist_row, text="Calibration history (newest first)", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(hist_row, text="Restore selected", command=self.restore_selected).pack(side="right")
        cols = ["installed", "part 1", "part 1 acc", "part 2 acc", "people"]
        self.history = ttk.Treeview(frm, columns=cols, show="headings", height=5, selectmode="browse")
        for c, w in zip(cols, (170, 260, 90, 90, 250)):
            self.history.heading(c, text=c)
            self.history.column(c, width=w, anchor="w", stretch=True)
        self.history.pack(fill="x")

        self.cal_log = tk.Text(frm, height=8, font=("Consolas", 9), wrap="none")
        self.cal_log.pack(fill="both", expand=True, pady=(10, 0))
        self.refresh_calibration_tab()

    def refresh_calibration_tab(self):
        cur = calibration.current()
        self.cal_text.config(text=calibration_wizard.describe(cur))
        self.summary.delete(*self.summary.get_children())
        recs = dataset.load_recordings(config.DATA_DIR)
        for p in sorted({r.person for r in recs}):
            secs = [sum(r.seconds for r in recs if r.person == p and r.label == s) for s in C.STATES]
            self.summary.insert("", "end", values=[p] + [f"{s:.0f}" for s in secs])
        self.history.delete(*self.history.get_children())
        for folder, p1, p2 in calibration.history():
            part1 = (f"{p1['direction']}  " + ", ".join(f"{t:.2f}" for t in p1["thresholds"])) if p1 else "-"
            self.history.insert("", "end", iid=str(folder), values=[
                folder.name, part1,
                f"{p1.get('accuracy', 0):.0%}" if p1 else "-",
                f"{p2.get('accuracy', 0):.0%}" if p2 and "accuracy" in p2 else "-",
                ", ".join((p2 or p1 or {}).get("people", []))])
        self.update_board_calibration()

    def update_board_calibration(self):
        """Does the board run the calibration that's installed on this laptop?"""
        msg = self.state_msg
        online = msg is not None and time.monotonic() - self.state_time < config.BOARD_TIMEOUT_S
        cur = calibration.current()
        local = [(cur[p] or {}).get("created") for p in ("part1", "part2")]
        if self.deploy_job is not None and not self.deploy_job.done:
            text, color = "Deploying...", "#c77800"
        elif not online:
            text, color = "Board offline: can't check which calibration it runs.", "#777777"
        else:
            board = [(msg.get(p) or {}).get("created") for p in ("part1", "part2")]
            if board == local:
                text, color = ("Board runs this calibration." if local[0] else
                               "Neither the board nor this laptop has a calibration yet."), "#2e9d4f"
            else:
                text, color = (f"Board runs a different calibration (part 1 made {board[0] or 'none'}, "
                               f"part 2 made {board[1] or 'none'}): deploy to update it."), "#d63a2f"
        self.cal_board.config(text=text, fg=color)

    def restore_selected(self):
        sel = self.history.selection()
        if not sel:
            messagebox.showinfo("Restore", "Select a calibration in the history list first.", parent=self)
            return
        folder = Path(sel[0])
        if not messagebox.askyesno("Restore", f"Install the calibration from {folder.name} again and deploy it?\n"
                                   "(The one in use now is kept in the history.)", parent=self):
            return
        calibration.restore(folder)
        self.refresh_calibration_tab()
        self.start_deploy()

    def start_deploy(self):
        if self.deploy_job is not None and not self.deploy_job.done:
            return
        self.cal_log.delete("1.0", "end")
        self.deploy_job = calibration.Job(calibration.deploy)
        self.deploy_btn.config(state="disabled")

    def poll_deploy(self):
        job = self.deploy_job
        if job is None:
            return
        lines = job.drain()
        if lines:
            self.cal_log.insert("end", "\n".join(lines) + "\n")
            self.cal_log.see("end")
        if job.done and not getattr(job, "handled", False):
            job.handled = True
            self.cal_log.insert("end", f"\nDeploy failed: {job.error}\n" if job.error else "\nDeployed.\n")
            self.cal_log.see("end")
            self.deploy_btn.config(state="normal")

    # ------------------------------------------------------ calibration wizard
    def on_capital_c(self, event):
        # Typing a capital C into a text field (e.g. a name) must not open the wizard.
        if event.widget.winfo_class() in ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Text"):
            return
        self.open_wizard()

    def open_wizard(self):
        if self.wizard is not None and self.wizard.winfo_exists():
            self.wizard.lift()
            self.wizard.focus_force()
            return
        self.wizard = calibration_wizard.CalibrationWizard(self)

    # ------------------------------------------------------ recording engine --
    # Driven by the wizard: rows from TOPIC_RAW are collected while self.recording is set.
    def start_recording(self, person, activity, duration, delay, on_done):
        """Count down `delay` s, record `duration` s, save; then on_done(path or None, message)."""
        self.recording = {"person": person, "activity": activity, "duration": max(5, duration),
                          "start_at": time.monotonic() + max(0, delay), "started": False,
                          "seqs": [], "rows": [], "on_done": on_done}

    def stop_recording(self, save):
        rec, self.recording = self.recording, None
        if rec is None:
            return
        self.bell()
        min_rows = C.WINDOW + 2 * int(config.TRIM_S * C.SAMPLE_HZ)
        path = None
        if not save:
            msg = "cancelled, nothing saved"
        elif len(rec["rows"]) < min_rows:
            msg = (f"too short: only {len(rec['rows'])} samples arrived (need {min_rows}). "
                   "Is the board online? Not saved")
        else:
            path = dataset.recording_path(rec["person"], rec["activity"], config.DATA_DIR)
            dataset.save_recording(path, rec["seqs"], np.array(rec["rows"]))
            msg = f"saved {len(rec['rows']) / C.SAMPLE_HZ:.0f} s to {path.parent.name}/{path.name}"
        rec["on_done"](path, msg)

    def update_recording(self):
        rec = self.recording
        if rec is None:
            return
        now = time.monotonic()
        if not rec["started"]:
            if now < rec["start_at"]:
                return
            rec["started"] = True
            rec["t0"] = now
            self.bell()
        if now - rec["t0"] >= rec["duration"]:
            self.stop_recording(save=True)

    def board_status(self):
        """(online, one-line description) for the wizard."""
        msg = self.state_msg
        if msg is None or time.monotonic() - self.state_time > config.BOARD_TIMEOUT_S:
            return False, "Board offline: recordings need the board app running and on WiFi."
        if not msg.get("imu_ok", True):
            return False, "Board online but the MPU-6050 isn't answering: check the wiring."
        return True, f"Board online, {msg.get('rows_total', 0)} samples received."

    # ------------------------------------------------------ MQTT handling ----
    def handle_raw(self, payload):
        msg = json.loads(payload)
        seq, rows = msg["seq"], msg["rows"]
        # The board's sequence number wraps at 65536; keep a never-wrapping copy
        # so recordings can tell real gaps from wrap-arounds.
        # last_raw_seq / unwrapped_seq = the next sequence number expected.
        if self.last_raw_seq is None:
            base = seq
        else:
            base = self.unwrapped_seq + (seq - self.last_raw_seq) % 65536
        self.last_raw_seq = (seq + len(rows)) % 65536
        self.unwrapped_seq = base + len(rows)
        self.plot_rows.extend(rows)
        rec = self.recording
        if rec is not None and rec["started"]:
            rec["seqs"].extend(range(base, base + len(rows)))
            rec["rows"].extend(rows)

    def handle_state(self, payload):
        self.state_msg = json.loads(payload)
        self.state_time = time.monotonic()
        self.board_mode = self.state_msg.get("mode")

    # ------------------------------------------------------ periodic update --
    def tick(self):
        try:
            while True:
                kind, payload = self.inbox.get_nowait()
                try:
                    (self.handle_raw if kind == "raw" else self.handle_state)(payload)
                except (ValueError, KeyError, TypeError) as exc:
                    print(f"Bad {kind} message: {exc}")
        except queue.Empty:
            pass
        self.update_live()
        self.update_recording()
        self.poll_deploy()
        if self.tick_count % 10 == 0:
            self.update_board_calibration()
        self.tick_count += 1
        if self.tick_count % PLOT_EVERY_TICKS == 0:
            self.update_plot()
        self.after(TICK_MS, self.tick)

    def update_live(self):
        now = time.monotonic()
        msg = self.state_msg
        online = msg is not None and now - self.state_time < config.BOARD_TIMEOUT_S

        if self.requested_mode is not None:
            if self.board_mode == self.requested_mode:
                self.requested_mode = None
            elif now - self.requested_at > MODE_CONFIRM_S:
                self.requested_mode = None
                self.mode_status.config(text="board didn't confirm the switch", foreground="#d63a2f")
        if self.requested_mode is not None:
            self.mode_var.set(self.requested_mode)
            self.mode_status.config(text="switching...", foreground="#c77800")
        elif self.board_mode in (1, 2):
            self.mode_var.set(self.board_mode)
            if online:
                self.mode_status.config(text=f"board is running Part {self.board_mode}", foreground="#2e9d4f")

        if not self.mqtt_ok:
            self.state_label.config(text="no MQTT connection", bg="#555555")
            self.status_label.config(
                text=("Lost the broker at " + self.broker + ", reconnecting..." if self.broker else
                      "Looking for the MQTT broker on the board (addresses in tools/board_secrets.py HOSTS), "
                      "retrying every few seconds..."))
            return
        if not online:
            self.state_label.config(text="board offline", bg="#555555")
            self.status_label.config(text=f"No state message on {config.TOPIC_STATE} for "
                                          f"{config.BOARD_TIMEOUT_S:.0f}+ s. Is the app running on the board?")
            return
        if not msg.get("imu_ok", True):
            self.state_label.config(text="NO IMU", bg="#8a2be2")
            self.status_label.config(text="MPU-6050 not answering: check VCC 3.3V, GND, SDA, SCL")
            return

        mode = msg.get("mode", 1)
        part = msg.get(f"part{mode}")
        state = msg.get("state")
        if part is None:
            self.state_label.config(text="NO CAL", bg="#555555")
        elif state is None:
            self.state_label.config(text="deciding...", bg="#555555")
        else:
            self.state_label.config(text=state.upper(), bg=config.STATE_COLORS[state])
        self.status_label.config(text=f"board online (broker {self.broker}) | Part {mode} on the LED matrix | "
                                      f"{msg.get('rows_total', 0)} samples received, "
                                      f"{msg.get('dropped', 0)} dropped")

        for m, frame in ((1, self.p1_frame), (2, self.p2_frame)):
            title = FRAME_TITLES[m] + ("  [ACTIVE] " if m == mode else " ")
            frame.config(fg="#1a56c4" if m == mode else "#777777", text=title)

        p1 = msg.get("part1")
        if p1:
            ranges = []
            lows = [None] + p1["thresholds"]
            highs = p1["thresholds"] + [None]
            for s, lo, hi in zip(p1["order"], lows, highs):
                lo_s = "      " if lo is None else f"{lo:5.2f} <"
                hi_s = "" if hi is None else f"< {hi:.2f}"
                ranges.append(f"  {lo_s} {s:<9}{hi_s}")
            peak = "-" if p1["peak"] is None else f"{p1['peak']:.2f} g"
            self.p1_text.config(text=f"direction  {p1['direction']}   (only input)\n"
                                     f"peak (2 s) {peak}\n"
                                     f"window     {p1['raw']}\n"
                                     f"smoothed   {p1['state']}\n"
                                     f"thresholds (g):\n" + "\n".join(ranges))
        else:
            self.p1_text.config(text="The board has no Part 1 calibration.\n"
                                     "Press Shift+C (capital C) to calibrate.")

        p2 = msg.get("part2")
        if p2:
            f = p2.get("features") or {}
            feats = (f"tilt {f.get('tilt_deg', 0):5.1f} deg   accel var {f.get('acc_std', 0):.2f} g\n"
                     f"accel peak {f.get('acc_peak', 0):.2f} g   rotation var {f.get('rot_var', 0):.0f} deg/s"
                     if f else "features: -")
            self.p2_text.config(text=f"window     {p2['raw'] or 'unsure (lying still)'}\n"
                                     f"smoothed   {p2['state']}\n{feats}\nquestions answered:\n  "
                                     + "\n  ".join(p2.get("path") or ["-"]))
        else:
            self.p2_text.config(text="The board has no Part 2 calibration.\n"
                                     "Press Shift+C (capital C) to calibrate.")

    def update_plot(self):
        if not self.plot_rows:
            return
        rows = np.array(self.plot_rows)
        t = (np.arange(len(rows)) - len(rows) + 1) / C.SAMPLE_HZ
        msg = self.state_msg or {}
        p1 = msg.get("part1")

        self.ax1.clear()
        if p1:
            sig = C.part1_direction_signal(rows, p1["direction"])
            self.ax1.plot(t, sig, color="#1a56c4", lw=1)
            for th in p1["thresholds"]:
                self.ax1.axhline(th, color="#999999", ls="--", lw=0.8)
            lows = [None] + p1["thresholds"]
            for s, lo in zip(p1["order"], lows):
                y = (lo if lo is not None else min(sig.min(), p1["thresholds"][0] - 0.3))
                self.ax1.text(-PLOT_SECONDS + 0.1, y + 0.05, s, fontsize=8, color=config.STATE_COLORS[s])
            self.ax1.set_title(f"Part 1 input: acceleration along {p1['direction']} (g), "
                               "with its thresholds", fontsize=10)
        else:
            self.ax1.plot(t, rows[:, C.COL["ax"]], lw=1, label="ax")
            self.ax1.plot(t, rows[:, C.COL["ay"]], lw=1, label="ay")
            self.ax1.plot(t, rows[:, C.COL["az"]], lw=1, label="az")
            self.ax1.legend(loc="upper left", fontsize=8)
            self.ax1.set_title("Acceleration (g), no Part 1 calibration yet", fontsize=10)
        self.ax1.set_xlim(-PLOT_SECONDS, 0)
        self.ax1.grid(alpha=0.3)

        self.ax2.clear()
        self.ax2b.clear()
        self.ax2b.yaxis.tick_right()
        self.ax2b.yaxis.set_label_position("right")
        acc = np.linalg.norm(rows[:, [C.COL["ax"], C.COL["ay"], C.COL["az"]]].astype(float), axis=1)
        gyr = np.linalg.norm(rows[:, [C.COL["gx"], C.COL["gy"], C.COL["gz"]]].astype(float), axis=1)
        self.ax2.plot(t, acc, color="#d63a2f", lw=1)
        self.ax2b.plot(t, gyr, color="#2e9d4f", lw=1)
        self.ax2.set_ylabel("|accel| (g)", color="#d63a2f", fontsize=9)
        self.ax2b.set_ylabel("|gyro| (deg/s)", color="#2e9d4f", fontsize=9)
        self.ax2.set_title("Part 2 uses all 6 channels (magnitudes shown)", fontsize=10)
        self.ax2.set_xlabel("seconds")
        self.ax2.set_xlim(-PLOT_SECONDS, 0)
        self.ax2.grid(alpha=0.3)
        self.canvas.draw_idle()

    def on_close(self):
        self.closing = True
        if self.mqtt is not None:
            self.mqtt.disconnect()
        self.destroy()


if __name__ == "__main__":
    WearableGui().mainloop()
