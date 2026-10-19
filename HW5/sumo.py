"""HW5 autonomous LEGO sumo robot (runs on the laptop).

    python HW5/sumo.py                  # everything
    python HW5/sumo.py --no-robot       # camera + overlay only (set up the ring / tags first)
    python HW5/sumo.py --list-cameras
    python HW5/sumo.py --fresh          # forget all calibration and walk through setup again

On start, a setup walkthrough shows what is calibrated and asks: ENTER = keep it,
N = start fresh (old files are moved to calibration_backup/, then each step is
explained on screen). `w` brings the walkthrough back at any time.

Setup, in order (each is saved, so only redo it when something moves):
    1. python HW5/make_tags.py, print, stick FRONT/BACK tags on the robot
    2. c   with the ring EMPTY: find the blue tape + save the empty-ring background
           (m: click 5+ points around the tape, then Enter, if c can't find it)
    3. p   robot in the ring, both tags in view: corrects for the camera looking
           from the side (2 s). Press p again with the robot at 2-3 other spots
           (near the edge on different sides) to make it more accurate.
    4. k   color sensor: on the GRAY floor press k, then on the BLUE tape press k
    5. d   robot near the center: drives forward, spins, drives back to measure
           encoders, gyro and which way is forward
Match:
    space  3 s countdown, then fight      r  reset after a loss      q  quit (motors stop)
    b      re-record the empty-ring background only (ring empty!)
    o      show what the opponent detector sees (red = "something is here")

Every match is logged to HW5/logs/match_<date>.csv (state, pose, sensors, commands
at 30 Hz) so a bad run can be looked at afterwards.

The shake controller (UNO Q, sumo-shake-controller app) publishes "shake:<0..1>"
on ME193/hudson. No messages for 2 s = defense.
"""

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import config
from behavior import Brain
from fusion import wrap
from mqttlib import MQTTClient
from robot_io import BLUE, GRAY, Robot, motor_to_robot_wheels
from sounds import LoseSound
from vision import Geometry, Vision, footprint, list_cameras
from wizard import SetupWizard, drive_calibration_ok

WINDOW = "HW5 sumo"
VIEW_WIDTH = 960
PANEL_W = 330

STATE_COLORS = {   # BGR, matching the decision-tree page
    "IDLE": (128, 115, 107), "COUNTDOWN": (128, 115, 107), "HOLD": (42, 50, 196), "LOST": (42, 50, 196),
    "EDGE_ESCAPE": (5, 110, 184), "PUSHED_SLIP": (189, 71, 114), "PUSHED_BACK": (189, 71, 114),
    "BLIND": (128, 115, 107), "DEFENSE_HOME": (107, 124, 15), "DEFENSE_HOLD": (107, 124, 15),
    "DEFENSE_FACE": (107, 124, 15), "ATTACK_SEARCH": (34, 74, 204), "ATTACK_CHARGE": (34, 74, 204),
    "ATTACK_AIM": (34, 74, 204), "CALIBRATING": (200, 200, 200), "DRIVE_CHECK": (42, 50, 196),
}


def connect_mqtt(on_message):
    """Connect with retries (test.mosquitto.org sometimes fails one attempt). None if unreachable."""
    client = MQTTClient(config.MQTT_BROKER, config.MQTT_PORT)
    for attempt in range(1, config.MQTT_CONNECT_RETRIES + 1):
        try:
            client.connect()
            client.subscribe(config.MQTT_TOPIC, on_message)
            print(f"[mqtt] subscribed to {config.MQTT_TOPIC}")
            return client
        except OSError as exc:
            print(f"[mqtt] connect attempt {attempt}/{config.MQTT_CONNECT_RETRIES} failed ({exc})")
            time.sleep(config.MQTT_RETRY_DELAY_SECONDS)
    print("[mqtt] broker unreachable: no shake controller, the robot stays in DEFENSE.")
    return None


# ---------------- drive calibration (key d) ----------------

class DriveCalibration:
    """Drive forward, stop, spin, stop, drive back; compare camera, encoders and gyro.

    Finds: which way is forward (reverse), which way a turn command turns (swap_sides),
    which encoder is which wheel and its sign, distance per wheel degree, wheel spacing,
    and the gyro's yaw axis and sign. The motor is driven directly (raw ports) throughout.
    """

    MOVE_S = 0.8
    SETTLE_S = 0.7
    SPEED = 40
    SPIN_S = 0.45            # short and slow: this robot turns ~300 deg/s at 55%
    SPIN_SPEED = 30

    def __init__(self, robot, now, geom=None):
        self.robot = robot
        self.geom = geom
        self.t0 = now
        self.steps = [("settle", self.SETTLE_S, (0, 0)), ("forward", self.MOVE_S, (self.SPEED, self.SPEED)),
                      ("settle", self.SETTLE_S, (0, 0)), ("spin", self.SPIN_S, (-self.SPIN_SPEED, self.SPIN_SPEED)),
                      ("settle", self.SETTLE_S, (0, 0)), ("back", self.MOVE_S, (-self.SPEED, -self.SPEED)),
                      ("settle", 0.3, (0, 0))]
        # The turn is added up frame by frame (each frame turns only a few degrees). Comparing
        # only the start and end headings can't tell a 240 deg left turn from a 120 deg right
        # turn -- that mistake once saved a backwards gyro and made the robot spin "confused".
        self.cam_turn = 0.0
        self.cam_prev = None
        self.cam_frames = 0
        self.i = 0
        self.step_t = now
        self.poses = []          # camera pose samples during the current settle
        self.marks = []          # (camera pose, raw encoders) at the end of each settle
        self.gyro_int = np.zeros(3)
        self.last_t = now
        self.result = None
        self.note = "starting"

    def update(self, now, snap, meas):
        if not self.robot.connected:
            self.robot.raw_motor(0, 0)
            self.result = ("failed: the robot's Bluetooth dropped (battery? too far?). "
                           "Wait for 'connected', then press d again")
            return True
        name, dur, motor = self.steps[self.i]
        dt = now - self.last_t
        self.last_t = now
        turning = name == "spin" or (name == "settle" and self.i == 4)    # the spin and the stop after it
        if turning and all(math.isfinite(g) for g in snap.gyro_raw):
            self.gyro_int += np.array(snap.gyro_raw) * dt
        if self.i >= 2 and self.i <= 4 and meas is not None and meas.our is not None:
            if self.cam_prev is not None and self.i >= 3:
                self.cam_turn += wrap(meas.our.theta - self.cam_prev)
                self.cam_frames += 1
            self.cam_prev = meas.our.theta
        if name == "settle" and now - self.step_t > dur - 0.35 and meas is not None and meas.our is not None:
            self.poses.append(meas.our)
        self.robot.raw_motor(*motor)
        self.note = f"{name} ({self.i + 1}/{len(self.steps)})"
        if now - self.step_t < dur:
            return False
        if name == "settle" and self.i < 6:
            if not self.poses:
                self.robot.raw_motor(0, 0)
                self.result = "failed: the camera can't see the robot's tags"
                return True
            p = self.poses
            two = [q for q in p if q.n_tags == 2] or p
            pose = (np.mean([q.x for q in two]), np.mean([q.y for q in two]),
                    math.atan2(np.mean([math.sin(q.theta) for q in two]), np.mean([math.cos(q.theta) for q in two])))
            self.marks.append((pose, self.robot.raw_encoders()))
            self.poses = []
        self.i += 1
        self.step_t = now
        if self.i >= len(self.steps):
            self.robot.raw_motor(0, 0)
            self.result = self._solve()
            return True
        return False

    def _solve(self):
        (p0, e0), (p1, e1), (p2, e2) = self.marks[:3]
        if e0 is None or e1 is None or e2 is None:
            return "failed: no encoder data from the Double Motor"
        d = self.robot.drive
        D = (p1[0] - p0[0], p1[1] - p0[1])
        dist = math.hypot(*D)
        if dist < 0.03:
            return "failed: the robot barely moved (is it on the floor? battery?)"
        sgn = lambda v: 1 if v >= 0 else -1
        # 1. Forward: the motor's "forward" (40, 40) moved us toward the FRONT tag, or not.
        along = D[0] * math.cos(p0[2]) + D[1] * math.sin(p0[2])
        reverse = bool(along < 0)
        # 2. Turning: the motor's (-40, +40) is a LEFT turn (counterclockwise) for a robot whose
        #    sides are as assumed. If the camera saw a clockwise turn, the sides are swapped.
        dth = self.cam_turn
        if self.cam_frames < 5:
            return "failed: the camera lost the tags while the robot turned. Press d again"
        if abs(wrap(dth - wrap(p2[2] - p1[2]))) > math.radians(25):
            return "failed: the camera's turn readings don't add up (tags lost mid-turn?). Press d again"
        if abs(dth) < math.radians(15):
            return f"failed: the robot barely turned ({math.degrees(dth):.0f} deg). Battery? Wheels slipping?"
        swap = bool(dth < 0)
        probe = {"reverse": reverse, "swap_sides": swap}
        # 3. Encoders: which port is our left/right wheel, and the sign that counts up going forward.
        dLm, dRm = e1[0] - e0[0], e1[1] - e0[1]
        rl, rr = motor_to_robot_wheels(probe, dLm, dRm)
        moved = -1 if reverse else 1                     # we moved backward if the motor is reversed
        sl, sr = sgn(rl) * moved, sgn(rr) * moved
        wheel = (abs(dLm) + abs(dRm)) / 2
        k = dist / wheel if wheel > 1 else d["ring_per_wheel_deg"]
        # 4. Wheel spacing from the spin: (right - left wheel travel) / turn angle. Must be > 0.
        sl_, sr_ = motor_to_robot_wheels(probe, e2[0] - e1[0], e2[1] - e1[1])
        dl, dr = sl * sl_ * k, sr * sr_ * k
        track = (dr - dl) / dth
        if track <= 0:
            return ("failed: wheels and camera disagree about the turn (track < 0). "
                    "Check both wheels touch the floor, then press d again")
        # 5. Gyro: the axis that moved most, scaled to the camera's turn.
        gi = self.gyro_int
        axis = int(np.argmax(np.abs(gi)))
        ratio = math.degrees(dth) / gi[axis] if abs(gi[axis]) > 1e-6 else 0
        scale = abs(ratio)
        if 0.8 < scale < 1.25:
            scale = 1.0
        # Sanity: the wheel spacing should be about the robot's width.
        if self.geom is not None and not (0.5 * self.geom.width < track < 1.6 * self.geom.width):
            return (f"failed: wheel spacing came out {track:.2f} ring units but the robot is "
                    f"{self.geom.width:.2f} wide. Keep the robot clear and press d again")

        # Plain Python types: numpy's bool/float/int don't save as JSON.
        new = {"reverse": reverse, "swap_sides": swap, "enc_left_sign": int(sl), "enc_right_sign": int(sr),
               "ring_per_wheel_deg": float(k), "wheel_dps_per_pct": float(wheel / self.MOVE_S / self.SPEED),
               "track_ring": float(track), "gyro_axis": int(axis),
               "gyro_sign": 1 if ratio >= 0 else -1, "gyro_scale": float(scale or 1.0)}
        d.update(new)                # same dict the fusion's odometry and the motor commands use
        config.DRIVE_CALIBRATION_PATH.write_text(json.dumps(d, indent=2))
        print(f"[drive] calibration: {json.dumps(new)}")
        fixes = [w for w, on in (("forward was backward", reverse), ("left/right were swapped", swap)) if on]
        return (f"ok{': ' + ', '.join(fixes) + ' (fixed)' if fixes else ''}. "
                f"turned {math.degrees(dth):.0f} deg, track {track:.2f}, gyro {'xyz'[axis]}{new['gyro_sign']:+d}"
                + (f" x{scale:.2f}" if scale != 1.0 else ""))


# ---------------- overlay ----------------

def draw_ring_circle(img, ring, radius, color, thick, s):
    a = np.linspace(0, 2 * math.pi, 90)
    pts = ring.to_image(np.stack([radius * np.cos(a), radius * np.sin(a)], 1)) * s
    cv2.polylines(img, [pts.astype(np.int32)], True, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, label, value, color):
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x + 95, y - 10), (x + w, y + 2), (70, 70, 70), -1)
    v = max(0.0, min(1.0, value))
    cv2.rectangle(img, (x + 95, y - 10), (x + 95 + int((w - 95) * v), y + 2), color, -1)
    cv2.putText(img, f"{value:.2f}", (x + w + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def render(frame, vision, brain, cmd, meas, snap, app):
    if frame is None:
        view = np.full((540, VIEW_WIDTH, 3), 30, np.uint8)
        s = 1.0
        cv2.putText(view, vision.status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        s = VIEW_WIDTH / frame.shape[1]
        view = cv2.resize(frame, (VIEW_WIDTH, int(frame.shape[0] * s)))
    st = brain.state
    ring = vision.ring
    g = vision.geom
    P = lambda pts: (ring.to_image(pts) * s).astype(np.int32)
    if ring is not None and frame is not None:
        draw_ring_circle(view, ring, 1.0, (255, 160, 40), 2, s)
        draw_ring_circle(view, ring, config.DEFENSE_HOME_R, (107, 124, 15), 1, s)
        draw_ring_circle(view, ring, config.PUSH_NEAR_EDGE_R, (189, 71, 114), 1, s)
        draw_ring_circle(view, ring, config.LOSE_FUSED_R, (42, 50, 196), 1, s)
        if meas is not None and meas.tag_corners:
            for tid, c in meas.tag_corners.items():
                col = (80, 220, 80) if tid == g.front_id else (220, 80, 220) if tid == g.back_id else (150, 150, 150)
                cv2.polylines(view, [(c * s).astype(np.int32)], True, col, 2)
                name = "FRONT" if tid == g.front_id else "BACK" if tid == g.back_id else str(tid)
                cv2.putText(view, name, tuple((c[0] * s).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if brain.fusion.ekf.initialized:
            col = STATE_COLORS.get(cmd.state, (255, 255, 255))
            cv2.polylines(view, [P(footprint(st.x, st.y, st.theta, g.length, g.width))], True, col, 2)
            nose = (st.x + math.cos(st.theta) * g.length / 2, st.y + math.sin(st.theta) * g.length / 2)
            a, b = P([(st.x, st.y), nose])
            cv2.arrowedLine(view, tuple(a), tuple(b), col, 2, tipLength=0.3)
            if st.pushed:
                tip = (st.x + st.push_dir[0] * 0.25, st.y + st.push_dir[1] * 0.25)
                a, b = P([(st.x, st.y), tip])
                cv2.arrowedLine(view, tuple(a), tuple(b), (189, 71, 114), 3, tipLength=0.3)
        if st.opp is not None:
            ox, oy, vx, vy = st.opp
            a = np.linspace(0, 2 * math.pi, 24)
            rim = np.stack([ox + st.opp_radius * np.cos(a), oy + st.opp_radius * np.sin(a)], 1)
            cv2.polylines(view, [P(rim)], True, (60, 60, 230), 2, cv2.LINE_AA)
            a, b = P([(ox, oy), (ox + vx * 0.5, oy + vy * 0.5)])
            cv2.arrowedLine(view, tuple(a), tuple(b), (60, 60, 230), 2, tipLength=0.3)
        if cmd.target is not None:
            cv2.drawMarker(view, tuple(P([cmd.target])[0]), (0, 220, 255), cv2.MARKER_CROSS, 16, 2)
    det = vision.opponent
    if app.show_detector and det is not None and det.debug is not None and frame is not None:
        fg, roi, sc, cands = det.debug
        x0, y0, x1, y1 = roi
        big = cv2.resize(fg, (max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        vx0, vy0 = int(x0 * s), int(y0 * s)
        region = view[vy0:vy0 + big.shape[0], vx0:vx0 + big.shape[1]]
        big = big[:region.shape[0], :region.shape[1]]
        region[big] = (0.45 * region[big] + 0.55 * np.array([40, 40, 255])).astype(np.uint8)
        cv2.rectangle(view, (vx0, vy0), (int(x1 * s), int(y1 * s)), (90, 90, 255), 1)
        for st_, ok in cands:
            bx, by, bw, bh = (int(v / sc * s) for v in st_[:4])
            cv2.rectangle(view, (vx0 + bx, vy0 + by), (vx0 + bx + bw, vy0 + by + bh),
                          (80, 220, 80) if ok else (120, 120, 120), 1)
    for p in app.clicks:
        cv2.circle(view, (int(p[0] * s), int(p[1] * s)), 5, (0, 255, 255), -1)
    app.view_scale = s

    panel = np.full((view.shape[0], PANEL_W, 3), 24, np.uint8)
    y = 30
    col = STATE_COLORS.get(cmd.state, (255, 255, 255))
    cv2.putText(panel, cmd.state, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, cmd.note[:40], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 28
    bar(panel, 12, y, 270, "aggression", st.aggression, (34, 74, 204)); y += 22
    bar(panel, 12, y, 270, "push prob", st.push_prob, (189, 71, 114)); y += 22
    bar(panel, 12, y, 270, "edge risk", st.edge_risk, (5, 110, 184)); y += 26
    lines = [
        f"motors  L {cmd.left:+4.0f}  R {cmd.right:+4.0f}   cap {brain.ramp.cap:.0f}%"
        + ("  RESISTED" if brain.ramp.resisted else ""),
        f"color   {st.color}" + (f"  rgb {tuple(int(v) for v in snap.color.rgb)}" if math.isfinite(snap.color.rgb[0]) else ""),
        f"pose    ({st.x:+.2f}, {st.y:+.2f})  r {st.r:.2f}  {math.degrees(st.theta):+.0f} deg",
        f"        sigma {st.sigma_pos:.3f}  tags {st.tags_seen}  {'cam ok' if st.camera_ok else 'BLIND'}",
        f"opp     " + (f"({st.opp[0]:+.2f}, {st.opp[1]:+.2f}) closing {st.closing_speed:+.2f}" if st.opp else "not seen"),
        "cues    " + " ".join(f"{k[:3]} {v:.1f}" for k, v in st.cues.items()),
        "lose    " + " ".join(f"{k[:3]}{'*' if v else '-'}" for k, v in st.lose_votes.items()),
        f"shake   " + ("connected" if st.shake_connected else "no messages (defense)"),
        f"gyro    {snap.gyro_yaw_dps:+.0f} dps   enc {snap.enc_left:.0f} / {snap.enc_right:.0f}",
        f"camera  {vision.fps:.0f} fps   ring {g.ring_diameter_cm:.0f} cm",
        "view    " + ("from the side, corrected" if ring is not None and ring.camera else
                      "NOT perspective-corrected (p)" if ring is not None else "no ring yet (c)"),
        f"robot   {app.robot.status}",
        f"vision  {vision.status[:34]}",
        f"mqtt    {'ok' if app.mqtt else 'offline'}",
    ]
    if getattr(app, "wizard", None) is not None and app.wizard.active:
        lines = lines[1:2] + lines[3:4] + lines[-3:]    # room for the setup instructions
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        y += 19
    y += 8
    if st.drive_warning:
        cv2.putText(panel, "STOPPED: " + st.drive_warning[:34], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (60, 60, 255), 1, cv2.LINE_AA)
        y += 19
    for line in app.message.split("\n"):
        cv2.putText(panel, line[:44], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        y += 19
    y = panel.shape[0] - 12
    cv2.putText(panel, "space start  r reset  w setup  o detector  q quit", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    return np.hstack([view, panel])


# ---------------- app ----------------

class App:
    def __init__(self, args):
        self.args = args
        self.geom = Geometry()
        self.robot = Robot()
        if not args.no_robot:
            self.robot.connect()
        self.brain = Brain(self.geom, self.robot.drive)
        self.vision = Vision(self.geom).start()
        self.sound = LoseSound()
        self.mqtt = connect_mqtt(lambda topic, payload: self.brain.fusion.aggression.on_message(payload, time.monotonic()))
        self.stop = threading.Event()
        self.clicks = []
        self.click_mode = False
        self.view_scale = 1.0
        self.message = "" if self.vision.ring else "press c with the ring empty"
        self.color_stage = None          # None, GRAY, BLUE
        self.color_samples = None        # (until, label, readings)
        self.drive_cal = None
        self.show_detector = False
        self.log_file = None
        self.log = None
        self.start_pressed = -1e9        # second `space` within 3 s overrides the perspective warning
        self.color_result = None         # (label, ok) when a color recording finishes
        self.drive_result = None         # (ok, message) when a drive calibration finishes
        self.wizard = SetupWizard(self, fresh=args.fresh)

    # ---------- match log ----------

    LOG_FIELDS = ["t", "state", "note", "cmd_l", "cmd_r", "x", "y", "theta_deg", "r", "sigma", "camera_ok",
                  "tags", "v", "vel_x", "vel_y", "color", "reflection", "rgb", "edge_risk", "edge_end",
                  "push_prob", "pushed", "cues", "opp_x", "opp_y", "opp_r", "closing", "aggression",
                  "gyro_dps", "enc_l", "enc_r", "lose_votes", "drive_warning", "speed_cap", "resisted"]

    def open_log(self):
        self.close_log()
        folder = config.HERE / "logs"
        folder.mkdir(exist_ok=True)
        path = folder / f"match_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.log_file = open(path, "w", newline="", encoding="utf-8")
        self.log = csv.writer(self.log_file)
        self.log.writerow(self.LOG_FIELDS)
        self.log_t0 = time.monotonic()
        print(f"[log] {path}")

    def close_log(self):
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = self.log = None

    def write_log(self, now, cmd, st, snap):
        if self.log is None:
            return
        c = snap.color
        opp = st.opp or (None, None)
        self.log.writerow([
            f"{now - self.log_t0:.3f}", cmd.state, cmd.note, f"{cmd.left:.0f}", f"{cmd.right:.0f}",
            f"{st.x:.3f}", f"{st.y:.3f}", f"{math.degrees(st.theta):.1f}", f"{st.r:.3f}", f"{st.sigma_pos:.3f}",
            int(st.camera_ok), st.tags_seen, f"{st.v:.3f}", f"{st.vel[0]:.3f}", f"{st.vel[1]:.3f}",
            st.color, f"{c.reflection:.0f}" if math.isfinite(c.reflection) else "",
            "/".join(f"{v:.0f}" for v in c.rgb) if math.isfinite(c.rgb[0]) else "",
            f"{st.edge_risk:.2f}", st.edge_end, f"{st.push_prob:.2f}", int(st.pushed),
            " ".join(f"{k}={v:.2f}" for k, v in st.cues.items()),
            "" if opp[0] is None else f"{opp[0]:.3f}", "" if opp[1] is None else f"{opp[1]:.3f}",
            f"{st.opp_radius:.3f}" if st.opp else "", f"{st.closing_speed:.2f}", f"{st.aggression:.2f}",
            f"{snap.gyro_yaw_dps:.0f}", f"{snap.enc_left:.0f}", f"{snap.enc_right:.0f}",
            " ".join(k for k, v in st.lose_votes.items() if v), st.drive_warning or "",
            f"{self.brain.ramp.cap:.0f}", int(self.brain.ramp.resisted)])

    def publish(self, msg):
        if self.mqtt:
            try:
                self.mqtt.publish(config.MQTT_TOPIC, config.SUMO_PREFIX + msg)
            except Exception:
                pass

    def reflex_loop(self):
        """Edge reflex at REFLEX_HZ: tape under the nose -> reverse immediately (faster than braking)."""
        period = 1.0 / config.REFLEX_HZ
        while not self.stop.is_set():
            color = self.robot.read_color()
            cmd = self.brain.reflex(time.monotonic(), color.label)
            if cmd is not None:
                self.robot.tank(*cmd, force=True)
            time.sleep(period)

    def on_mouse(self, event, x, y, flags, param):
        if self.click_mode and event == cv2.EVENT_LBUTTONDOWN and x < VIEW_WIDTH:
            self.clicks.append((x / self.view_scale, y / self.view_scale))
            self.message = f"{len(self.clicks)} points (5+), Enter to fit, Esc to cancel"

    def start_drive_calibration(self, now):
        if self.brain.match == "running":
            self.message = "reset (r) before drive calibration"
        elif self.args.no_robot:
            self.message = "no robot connected"
        elif not self.robot.connected:
            self.message = "robot not connected yet: wait for 'connected'"
        elif self.vision.ring is None:
            self.message = "the ring must be set up first (c)"
        else:
            self.drive_cal = DriveCalibration(self.robot, now, self.geom)
            return True
        return False

    def handle_key(self, k, now):
        if self.wizard.active and self.wizard.on_key(k, now):
            return True
        if k in (ord("q"), ord("Q")):
            return False
        if k == ord("w"):
            self.wizard = SetupWizard(self)
            return True
        if k == ord(" "):
            again = now - self.start_pressed < 3.0
            self.start_pressed = now
            if self.vision.ring is None:
                self.message = "calibrate the ring first (c)"
            elif not drive_calibration_ok() and not self.args.no_robot:
                # Without it, "forward", wheel distances and turning can all be backwards, and the
                # robot can drive straight out of the ring thinking it is driving in.
                self.message = "run the drive calibration first:\nrobot near the center, press d"
            elif self.vision.ring.camera is None and not again:
                self.message = ("perspective not corrected (p): positions near\nthe edge can be "
                                "15-20 cm off. space again = start anyway")
            else:
                self.brain.start_match(now)
                self.message = ""
        elif k == ord("r"):
            self.brain.reset()
            self.brain.fusion.drive_check.reset()   # it re-checks within ~1.5 s of driving
            self.message = "reset: press space to start"
        elif k == ord("c"):
            self.vision.request_calibration("auto")
            self.message = "finding the ring..."
        elif k == ord("b"):
            self.vision.request_calibration("background")
        elif k == ord("o"):
            self.show_detector = not self.show_detector
        elif k == ord("p"):
            self.vision.request_calibration("perspective")
            self.message = "watching the robot's tags for 2 s..."
        elif k in (ord("m"), ord("M")):
            self.click_mode, self.clicks = True, []
            self.message = "click 5+ points on the tape, Enter to fit"
        elif k == 13 and self.click_mode:
            self.vision.request_calibration(list(self.clicks))
            self.click_mode, self.clicks = False, []
        elif k == 27 and self.click_mode:
            self.click_mode, self.clicks = False, []
            self.message = ""
        elif k == ord("k"):
            if self.color_stage is None:
                self.color_stage = GRAY
                self.message = "color: put the sensor on the GRAY floor,\nthen press k"
            else:
                self.color_samples = (now + 0.6, self.color_stage, [])
                self.message = f"recording {self.color_stage}..."
        elif k == ord("d"):
            self.start_drive_calibration(now)
        return True

    def update_color_calibration(self, now):
        if self.color_samples is None:
            return
        until, label, readings = self.color_samples
        r = self.robot.read_color()
        if not readings or r is not readings[-1]:
            readings.append(r)
        if now < until:
            return
        self.color_samples = None
        ok = self.robot.classifier.record(label, readings)
        self.color_result = (label, ok)
        if not ok:
            self.message = "no color readings: is the sensor connected?"
            self.color_stage = None
        elif label == GRAY:
            self.color_stage = BLUE
            self.message = "color: now put the sensor on the BLUE tape,\nthen press k"
        else:
            self.color_stage = None
            self.message = "color calibration saved"

    def run(self):
        threading.Thread(target=self.reflex_loop, daemon=True, name="reflex").start()
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        period = 1.0 / config.CONTROL_HZ
        last_seq = None
        try:
            while True:
                t0 = time.monotonic()
                snap = self.robot.snapshot()
                seq, meas, frame = self.vision.latest()
                fresh = meas if seq != last_seq else None
                last_seq = seq
                now = time.monotonic()
                cmd = self.brain.step(now, snap, fresh)
                if self.drive_cal is not None:
                    done = self.drive_cal.update(now, snap, fresh)
                    cmd.state, cmd.note = "CALIBRATING", self.drive_cal.note
                    if done:
                        self.message = "drive " + self.drive_cal.result
                        self.drive_result = (self.drive_cal.result.startswith("ok"), self.drive_cal.result)
                        self.drive_cal = None
                        self.brain.fusion.drive_check.reset()
                else:
                    self.robot.tank(cmd.left, cmd.right)
                st = self.brain.state
                self.vision.opp_hint = st.opp[:2] if st.opp is not None else None
                for e in cmd.events:
                    if e == "lost":
                        self.robot.stop()
                        self.sound.play()
                        self.publish("lost")
                        print("[match] LOST: left the ring")
                    elif e == "drive_check":
                        self.robot.stop()
                        self.message = st.drive_warning
                        print(f"[match] STOPPED: {st.drive_warning}")
                    elif e == "started":
                        self.publish("start")
                        self.open_log()
                        print("[match] started")
                self.write_log(now, cmd, st, snap)
                if self.brain.match in ("idle", "lost") and self.log is not None and not cmd.events:
                    self.close_log()
                self.update_color_calibration(now)
                if self.wizard.active:
                    self.wizard.update(now)
                    self.message = self.wizard.text
                cv2.imshow(WINDOW, render(frame, self.vision, self.brain, cmd, meas, snap, self))
                k = cv2.waitKey(1) & 0xFF
                if k != 255 and not self.handle_key(k, now):
                    break
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        finally:
            self.close()

    def close(self):
        self.close_log()
        self.stop.set()
        self.robot.stop()
        self.vision.stop()
        self.robot.close()
        if self.mqtt:
            self.mqtt.disconnect()
        self.sound.close()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-robot", action="store_true", help="don't connect the LEGO devices")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--fresh", action="store_true", help="forget all calibration and walk through setup")
    args = p.parse_args()
    if args.list_cameras:
        list_cameras()
        return
    App(args).run()


if __name__ == "__main__":
    main()
