"""HW7 laptop GUI: live step count and distance with their uncertainty, plus recording and training.

The board does all the counting; this window displays its MQTT messages.
Two tabs:
  Live    - steps ± range, "how sure", distance ± range, where the uncertainty
            comes from, the count over time with its 68% / 95% bands, the
            5 networks' stride rates, the unfamiliar-motion light, Reset
  Record  - guided recordings (hallway walks, the long test walk, standing),
            then Train -> results -> Install and deploy / Discard

    .venv/Scripts/python HW7/gui.py
"""

import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import config  # noqa: E402
import dataset  # noqa: E402
import labels as L  # noqa: E402
import stepnet as S  # noqa: E402
from mqttlib import MQTTClient  # noqa: E402

TICK_MS = 100
PLOT_EVERY_TICKS = 5            # plot redraw at 2 Hz
COUNTDOWN_S = 3
STILL_DPS = 30.0                # auto-stop: shin rotation stays below this...
STILL_S = 3.0                   # ...for this long, after at least MIN_WALK_S of walking
MIN_WALK_S = 4.0
MAX_RECORD_S = 600
STANDING_S = 30
REPO_DIR = config.HW7_DIR.parent
APP_NAME = config.APP_DIR.name
BLUE, GREEN, RED, ORANGE, GREY = "#1a56c4", "#2e9d4f", "#d63a2f", "#c77800", "#777777"
if sys.platform == "win32":
    try:  # sharp text with Windows display scaling
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class Job:
    """Run fn(job) on a background thread; the GUI polls job.drain() / job.done."""

    def __init__(self, fn):
        self.lines = queue.Queue()
        self.done = False
        self.result = None
        self.error = None
        self.stop_requested = False
        self.handled = False
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
    if proc.wait() != 0:
        raise RuntimeError("deploy.py failed; the board still runs its old model")


def installed_model_created():
    try:
        with open(S.MODEL_FILE, encoding="utf-8") as f:
            return json.load(f).get("created")
    except FileNotFoundError:
        return None


class StepGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HW7 Ankle Step Counter")
        s = self.winfo_fpixels("1i") / 96
        w = min(int(1150 * s), int(self.winfo_screenwidth() * 0.95))
        h = min(int(900 * s), int(self.winfo_screenheight() * 0.9))
        self.geometry(f"{w}x{h}+10+10")

        self.inbox = queue.Queue()             # (kind, payload) from the MQTT thread
        self.state_msg = None
        self.state_time = 0.0
        self.history = []                      # one dict per state message since the last reset
        self.last_raw_seq = None
        self.unwrapped_seq = 0
        self.recent_gyro = []                  # |gyro| of recent rows, for auto-stop
        self.recording = None
        self.job = None                        # training or deploy
        self.trained = None                    # (model_json, report) waiting for Install / Discard
        self.tick_count = 0

        self.mqtt = None
        self.broker = None
        self.closing = False
        threading.Thread(target=self._connect_mqtt, daemon=True).start()

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)
        live, rec = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(live, text="  Live  ")
        tabs.add(rec, text="  Record and train  ")
        self._build_live(live)
        self._build_record(rec)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(TICK_MS, self.tick)

    # ------------------------------------------------------------ MQTT -------
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

    @property
    def online(self):
        return self.state_msg is not None and time.monotonic() - self.state_time < config.BOARD_TIMEOUT_S

    def reset_count(self):
        if self.mqtt is not None:
            self.mqtt.publish(config.TOPIC_CMD, "reset")
        self.history.clear()

    # ------------------------------------------------------------ Live tab ---
    def _build_live(self, root):
        top = ttk.Frame(root, padding=(8, 6))
        top.pack(fill="x")
        self.status_label = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.status_label.pack(side="left")
        tk.Button(top, text="Reset count to 0", font=("Segoe UI", 11), command=self.reset_count).pack(side="right")

        nums = ttk.Frame(root, padding=(8, 0))
        nums.pack(fill="x")
        nums.columnconfigure(0, weight=3)
        nums.columnconfigure(1, weight=2)

        steps = tk.Frame(nums, bg=BLUE, padx=14, pady=8)
        steps.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.steps_big = tk.Label(steps, text="-", font=("Segoe UI", 56, "bold"), fg="white", bg=BLUE)
        self.steps_big.pack(anchor="w")
        self.steps_range = tk.Label(steps, text="", font=("Segoe UI", 14), fg="white", bg=BLUE, justify="left")
        self.steps_range.pack(anchor="w")
        bar_row = tk.Frame(steps, bg=BLUE)
        bar_row.pack(fill="x", pady=(6, 0))
        self.sure_bar = ttk.Progressbar(bar_row, length=260, maximum=100)
        self.sure_bar.pack(side="left")
        self.sure_text = tk.Label(bar_row, text="", font=("Segoe UI", 11), fg="white", bg=BLUE)
        self.sure_text.pack(side="left", padx=8)

        dist = tk.Frame(nums, bg="#2f6f5f", padx=14, pady=8)
        dist.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.dist_big = tk.Label(dist, text="-", font=("Segoe UI", 40, "bold"), fg="white", bg="#2f6f5f")
        self.dist_big.pack(anchor="w")
        self.dist_range = tk.Label(dist, text="", font=("Segoe UI", 13), fg="white", bg="#2f6f5f", justify="left")
        self.dist_range.pack(anchor="w")

        info = ttk.Frame(root, padding=(8, 6))
        info.pack(fill="x")
        self.now_text = tk.Label(info, text="", font=("Consolas", 11), justify="left", anchor="w")
        self.now_text.pack(side="left")
        self.unfamiliar = tk.Label(info, text="", font=("Segoe UI", 11, "bold"), padx=10, pady=4)
        self.unfamiliar.pack(side="right")

        self.fig = Figure(figsize=(10.5, 5.2), dpi=90)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 2])
        self.ax_steps = self.fig.add_subplot(gs[0, 0])
        self.ax_rate = self.fig.add_subplot(gs[1, 0], sharex=self.ax_steps)
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_parts = self.fig.add_subplot(gs[1, 1])
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.09, hspace=0.42, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def update_live(self):
        msg = self.state_msg
        if not self.mqtt_ok:
            self.status_label.config(text=("Lost the broker at " + self.broker + ", reconnecting..." if self.broker else
                                           "Looking for the MQTT broker on the board (tools/board_secrets.py HOSTS)..."),
                                     foreground=GREY)
            return
        if not self.online:
            self.status_label.config(text=f"Board offline: no message on {config.TOPIC_STATE} for "
                                          f"{config.BOARD_TIMEOUT_S:.0f}+ s. Is the app running?", foreground=GREY)
            return
        if not msg.get("imu_ok", True):
            self.status_label.config(text="MPU-6050 not answering: check VCC 3.3V, GND, SDA, SCL", foreground=RED)
            return
        model = msg.get("model")
        local = installed_model_created()
        if model is None:
            self.status_label.config(text="Board online, but it has no model: record and train on the next tab, "
                                          "then Install and deploy.", foreground=ORANGE)
            self.steps_big.config(text="NO MODEL")
            return
        same = model.get("created") == local
        self.status_label.config(
            text=f"Board online ({self.broker}) | model made {model.get('created')}"
                 + ("" if same else f"  (this laptop's model: {local or 'none'}; deploy to update)")
                 + f" | {msg.get('rows_total', 0)} samples, {msg.get('dropped', 0)} dropped",
            foreground=GREEN if same else ORANGE)

        z = S.Z95
        self.steps_big.config(text=f"{msg['steps_shown']} steps")
        self.steps_range.config(text=f"± {z * msg['steps_sd']:.1f} steps (95% range {msg['steps_lo']} – "
                                     f"{msg['steps_hi']})")
        self.sure_bar["value"] = 100 * msg["sure"]
        self.sure_text.config(text=f"{msg['sure']:.0%} sure the true count is within ±{msg['sure_tol']:g} steps")
        self.dist_big.config(text=f"{msg['dist_m']:.1f} m")
        self.dist_range.config(text=f"± {z * msg['dist_sd']:.1f} m\n(95% range {msg['dist_lo']:.1f} – "
                                    f"{msg['dist_hi']:.1f} m)")
        self.now_text.config(text=f"right now: {msg['cadence_spm']:5.0f} ± {msg['cadence_sd']:.0f} steps/min   "
                                  f"{msg['speed']:.2f} ± {msg['speed_sd']:.2f} m/s   "
                                  f"walking time {msg['walk_s']:.0f} s")
        if msg.get("unfamiliar"):
            self.unfamiliar.config(text="UNFAMILIAR MOTION: the 5 networks disagree", bg=ORANGE, fg="white")
        else:
            self.unfamiliar.config(text="networks agree", bg="#d8efdf", fg=GREEN)

    def update_plot(self):
        for ax in (self.ax_steps, self.ax_rate, self.ax_dist, self.ax_parts):
            ax.clear()
        h = self.history
        if len(h) >= 2:
            t = np.array([m["_t"] for m in h]) - h[0]["_t"]
            steps = np.array([m["steps"] for m in h])
            sd = np.array([m["steps_sd"] for m in h])
            self._band(self.ax_steps, t, steps, sd, BLUE)
            self.ax_steps.set_title("steps since reset, with 68% and 95% ranges", fontsize=10)
            dist = np.array([m["dist_m"] for m in h])
            dsd = np.array([m["dist_sd"] for m in h])
            self._band(self.ax_dist, t, dist, dsd, "#2f6f5f")
            self.ax_dist.set_title("distance (m)", fontsize=10)
            cad = np.array([m["cadence_spm"] for m in h])
            csd = np.array([m["cadence_sd"] for m in h])
            self.ax_rate.fill_between(t, cad - csd, cad + csd, color=BLUE, alpha=0.2, lw=0)
            members = np.array([m.get("members_rate") or [np.nan] for m in h], dtype=float)
            if members.ndim == 2 and members.shape[1] > 1:
                for k in range(members.shape[1]):
                    self.ax_rate.plot(t, 120 * members[:, k], lw=0.7, alpha=0.8)
            self.ax_rate.plot(t, cad, color="black", lw=1.5)
            self.ax_rate.set_title("cadence (steps/min): ensemble ± spread, and each of the 5 networks", fontsize=10)
            self.ax_rate.set_xlabel("seconds since reset")
            last = h[-1]
            parts = last["steps_sd_parts"]
            dparts = last["dist_sd_parts"]
            names = ["steps: random", "steps: systematic", "steps: rounding", "dist: random", "dist: systematic"]
            vals = [parts["random"], parts["systematic"], parts["rounding"], dparts["random"], dparts["systematic"]]
            colors = [BLUE, "#7a9be0", "#b9c9ee", "#2f6f5f", "#8fbfb2"]
            y = np.arange(len(names))
            self.ax_parts.barh(y, [S.Z95 * v for v in vals], color=colors)
            self.ax_parts.set_yticks(y, names, fontsize=8)
            self.ax_parts.invert_yaxis()
            for yi, v, n in zip(y, vals, names):
                unit = "m" if n.startswith("dist") else "steps"
                self.ax_parts.text(S.Z95 * v, yi, f" ±{S.Z95 * v:.1f} {unit}", va="center", fontsize=8)
            self.ax_parts.set_title("where the uncertainty comes from (95%)", fontsize=10)
            self.ax_parts.set_xlim(0, max(1.0, 1.4 * S.Z95 * max(vals)))
        for ax in (self.ax_steps, self.ax_rate, self.ax_dist):
            ax.grid(alpha=0.3)
        self.canvas.draw_idle()

    @staticmethod
    def _band(ax, t, y, sd, color):
        ax.fill_between(t, y - S.Z95 * sd, y + S.Z95 * sd, color=color, alpha=0.15, lw=0, label="95%")
        ax.fill_between(t, y - sd, y + sd, color=color, alpha=0.3, lw=0, label="68%")
        ax.plot(t, y, color=color, lw=1.5)
        ax.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------ Record tab -
    def _build_record(self, root):
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        how = ("1. Measure the hallway once and type its length.   2. Record 6 hallway walks: there and back, "
               "counting every step (2 slow, 2 normal, 2 fast).   3. Record one long walk of a few minutes, "
               "back and forth, counting steps and hallway lengths (the final test, never trained on).   "
               "4. Press Train, check the results, then Install and deploy.\n"
               "Each walk: press the button, wait for the beep, stand still, walk, stop and stand still. "
               "Recording stops by itself 3 s after you stop walking.")
        tk.Label(frm, text=how, font=("Segoe UI", 10), justify="left", anchor="w", wraplength=1050).pack(fill="x")

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=(8, 4))
        ttk.Label(row, text="Hallway length (m):", font=("Segoe UI", 11)).pack(side="left")
        self.hall_var = tk.StringVar(value="" if dataset.load_hallway_m() is None else f"{dataset.load_hallway_m():g}")
        ttk.Entry(row, textvariable=self.hall_var, width=8, font=("Segoe UI", 11)).pack(side="left", padx=4)
        ttk.Button(row, text="Save", command=self.save_hallway).pack(side="left")
        ttk.Label(row, text="   lengths per hallway walk:", font=("Segoe UI", 11)).pack(side="left")
        self.lengths_var = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=1, to=10, textvariable=self.lengths_var, width=4).pack(side="left", padx=4)
        ttk.Label(row, text="(2 = there and back)", font=("Segoe UI", 9)).pack(side="left")

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=4)
        self.rec_buttons = []
        for text, kind in (("Record hallway walk", "hall"), ("Record long walk (test)", "long"),
                           (f"Record standing still ({STANDING_S} s)", "standing")):
            b = tk.Button(btns, text=text, font=("Segoe UI", 11), padx=10, command=lambda k=kind: self.start_recording(k))
            b.pack(side="left", padx=(0, 6))
            self.rec_buttons.append(b)
        self.stop_btn = tk.Button(btns, text="Stop now", font=("Segoe UI", 11), state="disabled",
                                  command=lambda: self.stop_recording(save=True))
        self.stop_btn.pack(side="left", padx=(12, 6))
        self.cancel_btn = tk.Button(btns, text="Cancel", font=("Segoe UI", 11), state="disabled",
                                    command=lambda: self.stop_recording(save=False))
        self.cancel_btn.pack(side="left")
        self.rec_status = tk.Label(frm, text="", font=("Segoe UI", 16, "bold"), anchor="w", fg=GREY)
        self.rec_status.pack(fill="x", pady=(4, 4))

        cols = ["recording", "kind", "seconds", "typed steps", "auto check", "distance m", "labels"]
        self.walks = ttk.Treeview(frm, columns=cols, show="headings", height=9, selectmode="browse")
        for c, w in zip(cols, (210, 70, 70, 90, 110, 90, 420)):
            self.walks.heading(c, text=c)
            self.walks.column(c, width=w, anchor="w", stretch=True)
        self.walks.pack(fill="x")
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=4)
        ttk.Button(row2, text="Delete selected recording", command=self.delete_selected).pack(side="left")
        self.train_btn = tk.Button(row2, text="Train (about 2 min)", font=("Segoe UI", 11, "bold"), bg="#cfe3ff",
                                   command=self.start_training)
        self.train_btn.pack(side="right")
        self.install_btn = tk.Button(row2, text="Install and deploy", font=("Segoe UI", 11), state="disabled",
                                     command=self.install_and_deploy)
        self.install_btn.pack(side="right", padx=6)
        self.discard_btn = tk.Button(row2, text="Discard", font=("Segoe UI", 11), state="disabled",
                                     command=self.discard)
        self.discard_btn.pack(side="right")
        self.model_label = tk.Label(frm, text="", font=("Segoe UI", 10), anchor="w")
        self.model_label.pack(fill="x")

        self.log = tk.Text(frm, height=12, font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.refresh_walks()

    def save_hallway(self):
        try:
            meters = float(self.hall_var.get())
            if not 2 <= meters <= 500:
                raise ValueError
        except ValueError:
            messagebox.showerror("Hallway", "Type the hallway length in meters (e.g. 24.5).", parent=self)
            return None
        dataset.save_hallway_m(meters)
        return meters

    def refresh_walks(self):
        self.walks.delete(*self.walks.get_children())
        recs = dataset.load_recordings(config.DATA_DIR, include_hw6=False)
        for r in recs:
            check = r.meta.get("auto_steps")
            self.walks.insert("", "end", iid=str(r.path), values=[
                r.name, r.kind, f"{r.seconds:.0f}", "-" if r.steps is None else r.steps,
                "-" if check is None else f"auto ≈ {check}",
                "-" if r.distance_m is None else f"{r.distance_m:g}", r.meta.get("label_note", "")])
        halls = sum(r.kind == "hall" for r in recs)
        longs = sum(r.kind == "long" for r in recs)
        created = installed_model_created()
        self.model_label.config(text=f"{halls} hallway walks, {longs} long walks recorded (HW6's standing and "
                                     f"walking recordings are added automatically).   Installed model: "
                                     f"{created or 'none yet'}")

    def delete_selected(self):
        sel = self.walks.selection()
        if not sel:
            return
        path = Path(sel[0])
        if messagebox.askyesno("Delete", f"Delete {path.name}? This can't be undone.", parent=self):
            dataset.delete_recording(path)
            self.refresh_walks()

    # recording engine: rows from TOPIC_RAW are collected while self.recording is set
    def start_recording(self, kind):
        if self.recording is not None:
            return
        hall_m = None
        if kind in ("hall", "long"):
            hall_m = self.save_hallway()
            if hall_m is None:
                return
        if not self.online:
            messagebox.showerror("Record", "The board is offline: start the app on the board first.", parent=self)
            return
        self.recording = {"kind": kind, "hall_m": hall_m, "lengths": self.lengths_var.get(),
                          "start_at": time.monotonic() + COUNTDOWN_S, "started": False,
                          "seqs": [], "rows": [], "walked_s": 0.0, "still_rows": 0}
        for b in self.rec_buttons:
            b.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.cancel_btn.config(state="normal")

    def update_recording(self):
        rec = self.recording
        if rec is None:
            return
        now = time.monotonic()
        if not rec["started"]:
            left = rec["start_at"] - now
            if left > 0:
                self.rec_status.config(text=f"Get ready... {left:.0f}", fg=ORANGE)
                return
            rec["started"] = True
            rec["t0"] = now
            self.bell()
        secs = len(rec["rows"]) / S.SAMPLE_HZ
        if rec["kind"] == "standing":
            self.rec_status.config(text=f"RECORDING: stand still... {STANDING_S - secs:.0f} s left", fg=RED)
            if secs >= STANDING_S:
                self.stop_recording(save=True)
            return
        walking = rec["walked_s"] >= MIN_WALK_S
        self.rec_status.config(
            text=f"RECORDING {secs:.0f} s: " + ("walk, counting every step. Stop and stand still at the end."
                                                if not walking else "walking... stand still 3 s to finish"),
            fg=RED)
        if walking and rec["still_rows"] >= STILL_S * S.SAMPLE_HZ:
            self.stop_recording(save=True)
        elif now - rec["t0"] > MAX_RECORD_S:
            self.stop_recording(save=True)

    def stop_recording(self, save):
        rec, self.recording = self.recording, None
        for b in self.rec_buttons:
            b.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")
        if rec is None:
            return
        self.bell()
        if not save or not rec["started"]:
            self.rec_status.config(text="Cancelled, nothing saved.", fg=GREY)
            return
        if len(rec["rows"]) < 2 * S.WINDOW:
            self.rec_status.config(text="Too short (or no data from the board): not saved.", fg=RED)
            return
        meta = None
        if rec["kind"] in ("hall", "long"):
            steps = simpledialog.askinteger("Steps", "How many steps did you count?", parent=self, minvalue=1)
            if steps is None:
                self.rec_status.config(text="No step count typed: not saved.", fg=GREY)
                return
            lengths = rec["lengths"]
            if rec["kind"] == "long":
                lengths = simpledialog.askinteger("Hallway lengths", "How many hallway lengths did you walk?\n"
                                                  "(there and back = 2)", parent=self, minvalue=1)
                if lengths is None:
                    self.rec_status.config(text="No length count typed: not saved.", fg=GREY)
                    return
            meta = {"steps": steps, "lengths": lengths, "hall_m": rec["hall_m"],
                    "distance_m": round(lengths * rec["hall_m"], 2), "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
        path = dataset.recording_path(rec["kind"])
        dataset.save_recording(path, rec["seqs"], np.array(rec["rows"]), meta)
        self.rec_status.config(text=f"Saved {path.name} ({len(rec['rows']) / S.SAMPLE_HZ:.0f} s).", fg=GREEN)
        if meta:
            self.check_walk(path)
        self.refresh_walks()

    def check_walk(self, path):
        """After typing the count: an independent automatic count, and whether the labels fit."""
        r = dataset.load_recording(path)
        axis, sign = L.pick_axis(r.rows)
        sig = L.swing_signal(r.rows, axis, sign)
        moving = sig[sig > 20]
        # 20% of a typical big swing: low enough to catch the short strides of a turn
        threshold = 0.2 * np.percentile(moving, 95) if len(moving) else 60.0
        auto = 2 * len(L.peaks(sig, r.t, threshold))
        lab = L.label_counted(r, axis, sign)
        r.meta.update({"auto_steps": auto, "label_note": lab.note, "labels_ok": lab.ok})
        dataset.save_meta(path, r.meta)
        if not lab.ok or abs(auto - r.steps) > max(2, 0.05 * r.steps):
            if messagebox.askyesno(
                    "Check this walk",
                    f"You typed {r.steps} steps; the automatic check found about {auto}.\n{lab.note}\n\n"
                    "If you might have lost count, delete it and walk it again. Delete this recording?", parent=self):
                dataset.delete_recording(path)
                self.rec_status.config(text="Deleted: record it again.", fg=ORANGE)

    # training / installing / deploying
    def start_training(self):
        if self.job is not None and not self.job.done:
            return
        self.log.delete("1.0", "end")
        self.trained = None
        self.install_btn.config(state="disabled")
        self.discard_btn.config(state="disabled")
        self.train_btn.config(state="disabled")

        def run(job):
            import train_stepnet     # imported here: torch takes a few seconds to load
            return train_stepnet.train_all(config.DATA_DIR, include_hw6=True, log=job.log,
                                           should_stop=lambda: job.stop_requested or self.closing)

        self.job = Job(run)
        self.job.kind = "train"

    def install_and_deploy(self):
        if self.trained is None:
            return
        import train_stepnet
        folder = train_stepnet.install(*self.trained)
        self.trained = None
        self.install_btn.config(state="disabled")
        self.discard_btn.config(state="disabled")
        self.log.insert("end", f"\nInstalled (old model kept in {folder.relative_to(config.HW7_DIR)}). Deploying...\n")
        self.refresh_walks()
        self.job = Job(deploy)
        self.job.kind = "deploy"

    def discard(self):
        self.trained = None
        self.install_btn.config(state="disabled")
        self.discard_btn.config(state="disabled")
        self.log.insert("end", "\nDiscarded. The installed model is unchanged.\n")

    def poll_job(self):
        job = self.job
        if job is None:
            return
        lines = job.drain()
        if lines:
            self.log.insert("end", "\n".join(lines) + "\n")
            self.log.see("end")
        if job.done and not job.handled:
            job.handled = True
            self.train_btn.config(state="normal")
            if job.error:
                self.log.insert("end", f"\n{job.kind.capitalize()} failed: {job.error}\n")
            elif job.kind == "train":
                self.trained = job.result
                self.install_btn.config(state="normal")
                self.discard_btn.config(state="normal")
                self.log.insert("end", "\nCheck the numbers above, then Install and deploy (or Discard).\n")
            else:
                self.log.insert("end", "\nDeployed. Press Reset on the Live tab before a walk.\n")
            self.log.see("end")

    # ------------------------------------------------------------ messages ---
    def handle_raw(self, payload):
        msg = json.loads(payload)
        seq, rows = msg["seq"], msg["rows"]
        # The board's sequence number wraps at 65536; keep a never-wrapping copy
        # so recordings can tell real gaps from wrap-arounds.
        if self.last_raw_seq is None:
            base = seq
        else:
            base = self.unwrapped_seq + (seq - self.last_raw_seq) % 65536
        self.last_raw_seq = (seq + len(rows)) % 65536
        self.unwrapped_seq = base + len(rows)
        rec = self.recording
        if rec is not None and rec["started"]:
            rec["seqs"].extend(range(base, base + len(rows)))
            rec["rows"].extend(rows)
            for row in rows:
                if float(np.linalg.norm(row[3:6])) < STILL_DPS:
                    rec["still_rows"] += 1
                else:
                    rec["still_rows"] = 0
                    rec["walked_s"] += 1 / S.SAMPLE_HZ

    def handle_state(self, payload):
        msg = json.loads(payload)
        self.state_msg = msg
        self.state_time = time.monotonic()
        if "steps" in msg:
            if self.history and msg["steps"] < self.history[-1]["steps"] - 0.5:
                self.history.clear()          # the board was reset or restarted
            msg["_t"] = time.time()
            self.history.append(msg)

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
        self.poll_job()
        self.tick_count += 1
        if self.tick_count % PLOT_EVERY_TICKS == 0:
            self.update_plot()
        self.after(TICK_MS, self.tick)

    def on_close(self):
        self.closing = True
        if self.job is not None:
            self.job.stop_requested = True
        if self.mqtt is not None:
            self.mqtt.disconnect()
        self.destroy()


if __name__ == "__main__":
    StepGui().mainloop()
