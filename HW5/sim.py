"""Headless 2D sumo simulator: tests fusion + the decision tree without hardware.

It fakes everything the real program gets: wheel encoders, IMU (gyro + accel,
~38 Hz), the color sensor (BLUE on the tape), late noisy camera frames with the
FRONT/BACK tag pose and the opponent blob, and the MQTT shake value. The real
`Brain` (fusion.py + behavior.py) runs on it unchanged.

Usage:
    python HW5/sim.py test            # the checklist tests in PLAN.md section 7
    python HW5/sim.py watch [shake]   # watch one match top-down (shake 0..1, default 0.7)
    python HW5/sim.py matches         # win/loss against a chasing opponent at several shake levels
"""

import math
import random
import sys
from collections import deque
from dataclasses import dataclass

import config
from behavior import Brain
from fusion import clip, wrap
from robot_io import BLUE, GRAY, ColorReading, RobotSnapshot
from vision import Measurement, OurPose, fraction_outside

PHYS_DT = 0.005
RING_RADIUS_M = 0.385
# Real-hardware delays and speeds (the first sim had none of these, so it never drove out):
WHEEL_DPS_PER_PCT = 14.0      # the real robot spins ~550 deg/s at 70% (match log 2026-10-19)
CAM_EXTRA_LAG_S = 0.06        # the real camera is later than config.CAMERA_LATENCY_S assumes...
CAM_JITTER_S = 0.04           # ...and not by a fixed amount
ENC_DELAY_S = 0.04            # encoder notifications arrive this late over Bluetooth
GYRO_DELAY_S = 0.02           # IMU notifications (a different stream) this late
OPP_REACTION_S = 0.15         # the opponent robot's own sensing delay (it used to see us perfectly)
K = config.DEFAULT_DRIVE["ring_per_wheel_deg"]
TRACK = config.DEFAULT_DRIVE["track_ring"]
MOTOR_TAU = 0.12              # motor spin-up / spin-down
BLE_CMD_DELAY_S = 0.08        # laptop -> Double Motor command delay over Bluetooth
COLOR_DELAY_S = 0.04          # color sensor -> laptop delay
COLOR_PERIOD_S = 0.015        # color sensor notification period
TAPE_HALF = 0.032             # 2.5 cm tape on a 77 cm ring
OUT_R = 1.05          # a robot whose center passes this is out


class SimGeometry:
    def __init__(self):
        self.ring_diameter_cm = 2 * RING_RADIUS_M * 100
        self.length = 14 / 38.5
        self.width = 13 / 38.5
        self.radius = 0.5 * math.hypot(self.length, self.width) * 0.85
        self.front_offset = self.back_offset = 4.5 / 38.5


@dataclass
class Body:
    x: float
    y: float
    theta: float
    radius: float
    wl: float = 0.0          # wheel surface speeds, ring units/s
    wr: float = 0.0
    enc_l: float = 0.0       # wheel degrees
    enc_r: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    cmd: tuple = (0, 0)
    strength: float = 1.0    # pushing strength (traction)

    @property
    def r(self):
        return math.hypot(self.x, self.y)


class World:
    def __init__(self, seed=0, opponent="chaser", camera=True, shake=0.7, us=None, opp=None):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.geom = SimGeometry()
        g = self.geom
        self.us = us or Body(-0.4, 0.0, 0.0, g.radius)
        self.opp = opp or (Body(0.4, 0.0, math.pi, g.radius) if opponent else None)
        self.opponent_mode = opponent
        self.camera = camera
        self.shake = shake
        self.ghost = False                    # opponent the physics ignores (a moving target for tests)
        self.carry = None                     # (until, vx, vy) robot picked up and carried
        self.shove = None                     # (until, vx, vy) external velocity on us
        # what `d` (drive calibration) would have measured on this robot
        self.brain = Brain(g, dict(config.DEFAULT_DRIVE, wheel_dps_per_pct=WHEEL_DPS_PER_PCT))
        self.pending = []                     # (deliver_t, Measurement)
        self.meas = None
        self._next_cam = 0.0
        self._next_imu = 0.0
        self._imu = ((0.0, 0.0, 1000.0), 0.0, 0.0)
        self._prev_v = (0.0, 0.0)
        self._next_ctrl = 0.0
        self._next_shake = 0.0
        self.events = []
        self.log = []
        self._cmd_queue = []                  # (apply_t, (left, right)): commands in flight over BLE
        self.sent_cmd = (0, 0)
        self._color_hist = []                 # (t, label) true color under the nose
        self._sensor_hist = deque(maxlen=100)
        self._us_hist = deque(maxlen=30)       # (t, x, y) at control ticks, for the opponent's delay  # (t, enc_l, enc_r, omega) for delayed encoder/gyro readings
        self._color_reading = GRAY
        self._next_color = 0.0
        self.max_nose_r = 0.0                 # furthest the nose got from the center
        self.flip_drive = False               # motor mounted backwards and `d` never run
        self.swap_drive = False               # motor's left/right ports swapped and `d` never run

    # ---------- physics ----------

    def _motors(self, b, dt):
        tl = b.cmd[0] * WHEEL_DPS_PER_PCT * K
        tr = b.cmd[1] * WHEEL_DPS_PER_PCT * K
        a = 1 - math.exp(-dt / MOTOR_TAU)
        b.wl += a * (tl - b.wl)
        b.wr += a * (tr - b.wr)
        v = (b.wl + b.wr) / 2
        w = (b.wr - b.wl) / TRACK
        return v * math.cos(b.theta), v * math.sin(b.theta), w

    def physics(self, dt):
        real_opp = self.opp is not None and not self.ghost
        bodies = [self.us] + ([self.opp] if real_opp else [])
        intents = [self._motors(b, dt) for b in bodies]
        vels = [[ix, iy] for ix, iy, _ in intents]
        if self.shove and self.t < self.shove[0]:
            vels[0][0] += self.shove[1]
            vels[0][1] += self.shove[2]
        if self.carry and self.t < self.carry[0]:
            vels[0] = [self.carry[1], self.carry[2]]     # picked up: our wheels don't touch the floor
            intents[0] = (0.0, 0.0, 0.0)
        if real_opp:
            dx, dy = self.opp.x - self.us.x, self.opp.y - self.us.y
            d = math.hypot(dx, dy)
            if d < self.us.radius + self.opp.radius and d > 1e-6:
                n = (dx / d, dy / d)
                a = vels[0][0] * n[0] + vels[0][1] * n[1]
                b = vels[1][0] * n[0] + vels[1][1] * n[1]
                if a > b:   # pressing into each other: share one normal speed, weighted by strength
                    s0, s1 = self.us.strength, self.opp.strength
                    c = (a * s0 + b * s1) / (s0 + s1)
                    vels[0][0] += (c - a) * n[0]; vels[0][1] += (c - a) * n[1]
                    vels[1][0] += (c - b) * n[0]; vels[1][1] += (c - b) * n[1]
                overlap = self.us.radius + self.opp.radius - d
                self.us.x -= n[0] * overlap / 2; self.us.y -= n[1] * overlap / 2
                self.opp.x += n[0] * overlap / 2; self.opp.y += n[1] * overlap / 2
        for b, (ix, iy, w), (vx, vy) in zip(bodies, intents, vels):
            b.x += vx * dt
            b.y += vy * dt
            b.theta = wrap(b.theta + w * dt)
            b.vx, b.vy, b.omega = vx, vy, w
            b.enc_l += b.wl / K * dt        # wheels keep turning even if they slip
            b.enc_r += b.wr / K * dt

    # ---------- sensors ----------

    def nose(self, b=None):
        b = b or self.us
        L = self.geom.length / 2
        return b.x + L * math.cos(b.theta), b.y + L * math.sin(b.theta)

    def color(self):
        r = math.hypot(*self.nose())
        return BLUE if 1 - TAPE_HALF < r < 1 + TAPE_HALF else GRAY

    def color_reading(self):
        """What the laptop sees: sampled every COLOR_PERIOD_S, COLOR_DELAY_S late."""
        self._color_hist.append((self.t, self.color()))
        if self.t >= self._next_color:
            self._next_color = self.t + COLOR_PERIOD_S
            want = self.t - COLOR_DELAY_S
            for t, label in reversed(self._color_hist):
                if t <= want:
                    self._color_reading = label
                    break
            self._color_hist = [h for h in self._color_hist if h[0] > want - 0.05]
        return self._color_reading

    def send(self, left, right):
        """Command the motors; it reaches them BLE_CMD_DELAY_S later."""
        cmd = (int(round(clip(left, -100, 100))), int(round(clip(right, -100, 100))))
        self.sent_cmd = cmd
        self._cmd_queue.append((self.t + BLE_CMD_DELAY_S, cmd))

    def _delayed(self, delay):
        """(enc_l, enc_r, omega) as they were `delay` seconds ago."""
        want = self.t - delay
        for t, *vals in reversed(self._sensor_hist):
            if t <= want:
                return vals
        return self._sensor_hist[0][1:]

    def snapshot(self):
        u = self.us
        self._sensor_hist.append((self.t, u.enc_l, u.enc_r, u.omega))
        if self.t >= self._next_imu:
            self._next_imu = self.t + 1 / 38
            ax = (u.vx - self._prev_v[0]) * 38
            ay = (u.vy - self._prev_v[1]) * 38
            self._prev_v = (u.vx, u.vy)
            c, s = math.cos(-u.theta), math.sin(-u.theta)
            bx, by = ax * c - ay * s, ax * s + ay * c
            mg = RING_RADIUS_M / 9.81 * 1000
            n = self.rng.gauss
            self._imu = ((bx * mg + n(0, 15), by * mg + n(0, 15), 1000 + n(0, 15)),
                         math.degrees(self._delayed(GYRO_DELAY_S)[2]) + n(0, 1.5), self.t)
        accel, gyro, imu_t = self._imu
        enc_l, enc_r, _ = self._delayed(ENC_DELAY_S)
        snap = RobotSnapshot(t=self.t, enc_left=enc_l, enc_right=enc_r, gyro_yaw_dps=gyro,
                             accel_mg=accel, imu_t=imu_t, cmd_left=self.sent_cmd[0], cmd_right=self.sent_cmd[1])
        snap.color = ColorReading(label=self.color_reading(), t=self.t)
        return snap

    def camera_frame(self):
        if self.t >= self._next_cam:
            self._next_cam = self.t + 1 / 30
            if self.camera:
                g, u, n = self.geom, self.us, self.rng.gauss
                m = Measurement(t=self.t)
                roll = self.rng.random()
                if roll > 0.03:
                    m.our = OurPose(u.x + n(0, 0.008), u.y + n(0, 0.008),
                                    wrap(u.theta + n(0, math.radians(1.5))), 2 if roll > 0.08 else 1)
                    m.our_out_fraction = fraction_outside(m.our.x, m.our.y, m.our.theta, g.length, g.width)
                if self.opp and self.rng.random() > 0.05:
                    m.opp = (self.opp.x + n(0, 0.015), self.opp.y + n(0, 0.015), self.opp.radius)
                lag = config.CAMERA_LATENCY_S + CAM_EXTRA_LAG_S + self.rng.uniform(0, CAM_JITTER_S)
                self.pending.append((self.t + lag, m))
                self.pending.sort(key=lambda p: p[0])
        while self.pending and self.pending[0][0] <= self.t:
            self.meas = self.pending.pop(0)[1]
        return self.meas

    # ---------- opponent AI ----------

    def opponent_ai(self):
        o = self.opp
        if o is None or self.opponent_mode in (None, "still"):
            if o is not None:
                o.cmd = (0, 0)
            return
        nose_r = math.hypot(*self.nose(o))
        to_center = math.atan2(-o.y, -o.x)
        if nose_r > 0.9 and abs(wrap(to_center - o.theta)) > math.pi / 2:
            err = wrap(to_center - o.theta)
            s = 60 if err > 0 else -60
            o.cmd = (-s, s)
            return
        # it reacts to where we were OPP_REACTION_S ago (its own sensing delay, like ours)
        seen = self._us_hist[0] if self._us_hist else (self.us.x, self.us.y)
        for t, x, y in self._us_hist:
            if t <= self.t - OPP_REACTION_S:
                seen = (x, y)
        target = math.atan2(seen[1] - o.y, seen[0] - o.x)
        err = wrap(target - o.theta)
        turn = clip(80 * err, -60, 60)
        fwd = 65 * max(0.0, math.cos(err)) if abs(err) < 1.0 else 0
        o.cmd = (fwd - turn, fwd + turn)

    # ---------- run ----------

    def start(self):
        self.brain.start_match(self.t)
        self.brain.countdown_until = self.t

    def tick(self):
        """Advance one physics step; brain at CONTROL_HZ, reflex every step."""
        if self.t >= self._next_shake:
            self._next_shake = self.t + 0.1
            if self.shake is not None:
                self.brain.fusion.aggression.on_message(f"shake:{self.shake:.2f}", self.t)
        snap = self.snapshot()
        reflex = self.brain.reflex(self.t, snap.color.label)
        if reflex is not None:
            self.send(*reflex)
        if self.t >= self._next_ctrl:
            self._next_ctrl = self.t + 1 / config.CONTROL_HZ
            meas = self.camera_frame()
            cmd = self.brain.step(self.t, snap, meas)
            self.send(cmd.left, cmd.right)
            self.events += [(self.t, e) for e in cmd.events]
            self._us_hist.append((self.t, self.us.x, self.us.y))
            self.opponent_ai()
            st = self.brain.state
            self.log.append((self.t, cmd.state, st.pushed, st.push_prob,
                             math.hypot(st.x - self.us.x, st.y - self.us.y),
                             abs(wrap(st.theta - self.us.theta))))
        else:
            self.camera_frame()
        while self._cmd_queue and self._cmd_queue[0][0] <= self.t:
            c = self._cmd_queue.pop(0)[1]
            c = (-c[1], -c[0]) if self.flip_drive else c
            self.us.cmd = (c[1], c[0]) if self.swap_drive else c
        self.physics(PHYS_DT)
        self.t += PHYS_DT
        self.max_nose_r = max(self.max_nose_r, math.hypot(*self.nose()))

    def run(self, seconds, stop_when_out=True):
        end = self.t + seconds
        while self.t < end:
            self.tick()
            if stop_when_out and self.outcome():
                break
        return self.outcome()

    def outcome(self):
        if self.us.r > OUT_R:
            return "lost"
        if self.opp is not None and self.opp.r > OUT_R:
            return "won"
        return None


# ---------------- tests ----------------

def test_ekf_converges():
    w = World(seed=1, opponent=None, shake=0.0, us=Body(0.3, -0.2, 1.0, SimGeometry().radius))
    w.start()
    w.run(1.5)
    _, _, _, _, perr, therr = w.log[-1]
    ok = perr < 0.03 and therr < math.radians(5)
    print(f"[{'PASS' if ok else 'FAIL'}] EKF converges: position error {perr:.3f}, heading error {math.degrees(therr):.1f} deg")
    # Camera drops out for 1 s: dead reckoning should hold.
    w.camera = False
    w.shake = 0.8
    w.run(1.0)
    _, state, _, _, perr2, therr2 = w.log[-1]
    ok2 = perr2 < 0.15
    print(f"[{'PASS' if ok2 else 'FAIL'}] dead reckoning through a 1 s camera dropout: error {perr2:.3f} ({state})")
    return ok and ok2


def test_never_leaves_alone(runs=100, seconds=20):
    outs = 0
    for i in range(runs):
        rng = random.Random(100 + i)
        camera = i % 2 == 0          # half with the camera, half BLIND (color reflex only)
        body = Body(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-math.pi, math.pi), SimGeometry().radius)
        w = World(seed=i, opponent=None, camera=camera, shake=rng.uniform(0.2, 1.0), us=body)
        if camera:
            # With the camera on and no opponent the robot just searches at the center, so give
            # it a target to chase near the edge: a ghost opponent the physics ignores.
            w.opp = Body(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), 0, 0.0)
            w.opponent_mode = "still"
            w.ghost = True
        w.start()
        end = w.t + seconds
        while w.t < end:
            if camera and w.opp is not None and rng.random() < 0.002:
                a, rad = rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 0.95)
                w.opp.x, w.opp.y = rad * math.cos(a), rad * math.sin(a)
            w.tick()
            if w.us.r > OUT_R:
                outs += 1
                break
    ok = outs == 0
    print(f"[{'PASS' if ok else 'FAIL'}] alone in the ring: left it in {outs}/{runs} runs of {seconds} s "
          f"(half camera, half color-only)")
    return ok


def test_push_detection():
    g = SimGeometry()
    w = World(seed=3, opponent=None, shake=0.0, us=Body(0.0, 0.0, 0.0, g.radius))
    w.start()
    w.run(2.0)
    w.shove = (w.t + 1.0, -0.35, 0.0)          # shoved backward at 0.35 ring units/s for 1 s
    t0 = w.t
    w.run(1.0)
    fired = [t for t, _, pushed, *_ in w.log if t >= t0 and pushed]
    latency = (fired[0] - t0) if fired else None
    ok1 = latency is not None and latency < 0.5
    print(f"[{'PASS' if ok1 else 'FAIL'}] push detected {'in %.2f s' % latency if latency is not None else 'never'}")

    # Free driving (BLIND wander with the camera on is not possible, so attack a ghost target).
    w2 = World(seed=4, opponent=None, shake=0.9, us=Body(-0.3, 0.2, 0.5, g.radius))
    w2.opp = Body(0.5, -0.4, 0, 0.0)
    w2.opponent_mode = "still"
    w2.ghost = True
    w2.start()
    rng = random.Random(9)
    end = w2.t + 20
    while w2.t < end:
        if rng.random() < 0.003:
            a = rng.uniform(-math.pi, math.pi)
            w2.opp.x, w2.opp.y = 0.7 * math.cos(a), 0.7 * math.sin(a)
        w2.tick()
    frac = sum(1 for _, _, pushed, *_ in w2.log if pushed) / len(w2.log)
    ok2 = frac < 0.02
    print(f"[{'PASS' if ok2 else 'FAIL'}] no false push while driving freely: pushed {100 * frac:.1f}% of the time")
    return ok1 and ok2


def test_drive_check():
    g = SimGeometry()
    w = World(seed=8, opponent=None, shake=0.9, us=Body(0.0, 0.0, 0.0, g.radius))
    w.opp = Body(0.6, 0.3, 0, 0.0); w.opponent_mode = "still"; w.ghost = True
    w.flip_drive = True
    w.start()
    w.run(6.0)
    hit = [t for t, e in w.events if e == "drive_check"]
    ok1 = bool(hit)
    print(f"[{'PASS' if ok1 else 'FAIL'}] motor driving backwards is caught "
          f"{'after %.1f s, nose got to r %.2f' % (hit[0], w.max_nose_r) if hit else '- NOT caught'}")
    w3 = World(seed=9, opponent="chaser", shake=0.0, us=Body(0.0, 0.0, 0.0, g.radius))
    w3.swap_drive = True
    w3.start()
    w3.run(8.0)
    hit3 = [t for t, e in w3.events if e == "drive_check"]
    ok3 = bool(hit3)
    print(f"[{'PASS' if ok3 else 'FAIL'}] left/right swapped (spins and gets confused) is caught "
          f"{'after %.1f s: %s' % (hit3[0], w3.brain.state.drive_warning) if hit3 else '- NOT caught'}")
    ok1 = ok1 and ok3
    false = 0
    for i in range(20):
        rng = random.Random(70 + i)
        w2 = World(seed=70 + i, opponent="chaser", shake=rng.uniform(0, 1))
        w2.start()
        w2.run(20)
        false += any(e == "drive_check" for _, e in w2.events)
    ok2 = false == 0
    print(f"[{'PASS' if ok2 else 'FAIL'}] no false drive-check stops in 20 normal matches ({false})")
    return ok1 and ok2


def test_lose_detection():
    g = SimGeometry()
    w = World(seed=5, opponent=None, shake=0.0, us=Body(0.6, 0.0, 0.0, g.radius))
    w.start()
    w.run(1.0)
    w.carry = (w.t + 3.0, 0.5, 0.0)            # picked up and carried straight out
    t0 = w.t
    w.run(3.0, stop_when_out=False)
    lost = [t for t, e in w.events if e == "lost"]
    ok = bool(lost)
    print(f"[{'PASS' if ok else 'FAIL'}] carried out of the ring -> lost "
          f"{'%.2f s after the shove started' % (lost[0] - t0) if lost else 'NOT detected'}")
    # No false loss in a normal 20 s match.
    w2 = World(seed=6, opponent="chaser", shake=0.6)
    w2.start()
    while w2.t < 20 and not w2.outcome():
        w2.tick()
    false_loss = any(e == "lost" for _, e in w2.events) and w2.us.r < 1.0
    print(f"[{'PASS' if not false_loss else 'FAIL'}] no false 'lost' while still inside the ring")
    return ok and not false_loss


def matches(n=20, seconds=40):
    for shake in (0.0, 0.5, 1.0):
        res = {"won": 0, "lost": 0, None: 0}
        for i in range(n):
            w = World(seed=1000 + i, opponent="chaser", shake=shake)
            rng = random.Random(i)
            w.us.x, w.us.y, w.us.theta = rng.uniform(-0.5, -0.3), rng.uniform(-0.2, 0.2), rng.uniform(-0.5, 0.5)
            w.opp.x, w.opp.y = rng.uniform(0.3, 0.5), rng.uniform(-0.2, 0.2)
            w.start()
            res[w.run(seconds)] += 1
        print(f"shake {shake:.1f}: won {res['won']}, lost {res['lost']}, draw {res[None]} (of {n})")


def watch(shake=0.7):
    import cv2
    import numpy as np
    w = World(seed=random.randrange(10000), opponent="chaser", shake=shake)
    w.start()
    S, C = 500, 250
    def px(x, y):
        return int(C + x * 200), int(C - y * 200)
    while True:
        for _ in range(int(1 / 30 / PHYS_DT)):
            w.tick()
        img = np.full((S, S, 3), 40, np.uint8)
        cv2.circle(img, (C, C), 200, (200, 120, 30), 6)
        for b, col in ((w.us, (255, 200, 80)), (w.opp, (60, 90, 230))):
            cv2.circle(img, px(b.x, b.y), int(b.radius * 200), col, -1)
            cv2.line(img, px(b.x, b.y), px(*w.nose(b)), (255, 255, 255), 2)
        st, cmd = w.brain.state, w.brain.last
        cv2.circle(img, px(st.x, st.y), 4, (0, 255, 0), -1)
        if cmd.target:
            cv2.drawMarker(img, px(*cmd.target), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(img, f"{cmd.state} {cmd.note}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"push {st.push_prob:.2f}  edge {st.edge_risk:.2f}  shake {st.aggression:.2f}  t {w.t:.1f}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        out = w.outcome()
        if out:
            cv2.putText(img, out.upper(), (C - 60, C), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.imshow("sumo sim (q quits, n new match)", img)
        k = cv2.waitKey(1 if not out else 0) & 0xFF
        if k == ord("q"):
            break
        if out or k == ord("n"):
            w = World(seed=random.randrange(10000), opponent="chaser", shake=shake)
            w.start()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        results = [test_ekf_converges(), test_push_detection(), test_lose_detection(), test_drive_check(),
                   test_never_leaves_alone()]
        print(f"\n{sum(results)}/{len(results)} test groups passed")
        sys.exit(0 if all(results) else 1)
    elif mode == "watch":
        watch(float(sys.argv[2]) if len(sys.argv) > 2 else 0.7)
    elif mode == "matches":
        matches()
    else:
        print(__doc__)
