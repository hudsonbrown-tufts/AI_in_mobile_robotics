"""Sensor fusion for the sumo robot.

    PoseEKF          our pose from wheel encoders + IMU gyro (predict) and the
                     camera's FRONT/BACK tags (update). It also estimates an
                     "external velocity": motion the wheels don't explain, which
                     is what being pushed (or wheel slip) looks like.
    OpponentTracker  constant-velocity Kalman filter on the camera's opponent blob.
    PushDetector     log-odds sum of five cues -> push probability + direction.
    edge_risk        fused distance / time to the tape for both ends + color sensor.
    LoseDetector     camera footprint, fused position and color history must agree.
    Aggression       MQTT shake intensity, smoothed, decaying to 0 if the controller goes quiet.

`Fusion.step()` runs all of them once per control tick and returns a `FusedState`.
Every function takes explicit timestamps (`now`), so sim.py can drive it with
simulated time.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from robot_io import BLUE, GRAY


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def _finite(*vals):
    return all(v is not None and math.isfinite(v) for v in vals)


def unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)


# ---------------- odometry ----------------

class Odometry:
    """Dead reckoning from wheel encoders (distance) and the gyro (turning).

    Keeps a short history so a late camera frame can be moved forward to "now".
    """

    def __init__(self, drive):
        self.drive = drive
        self.prev = None            # (t, enc_left, enc_right)
        self.x = self.y = self.theta = 0.0
        self.history = deque(maxlen=200)
        self.ds = self.dtheta = 0.0
        self.v = 0.0                # forward speed from the wheels, ring units/s
        self.omega_enc = 0.0        # turn rate from the wheels, rad/s
        self.omega_gyro = math.nan  # turn rate from the gyro, rad/s
        self.encoders_ok = False

    def step(self, snap, dt):
        d = self.drive
        k = d["ring_per_wheel_deg"]
        ds = dth_enc = 0.0
        self.encoders_ok = _finite(snap.enc_left, snap.enc_right)
        if self.encoders_ok:
            if self.prev is not None:
                dl = (snap.enc_left - self.prev[1]) * k
                dr = (snap.enc_right - self.prev[2]) * k
                if abs(dl) < 0.5 and abs(dr) < 0.5:          # ignore encoder wrap/glitch
                    ds = (dl + dr) / 2
                    dth_enc = (dr - dl) / d["track_ring"]
            self.prev = (snap.t, snap.enc_left, snap.enc_right)
        else:
            # No encoders: estimate from the commanded speeds.
            wl = snap.cmd_left * d["wheel_dps_per_pct"] * k * dt
            wr = snap.cmd_right * d["wheel_dps_per_pct"] * k * dt
            ds, dth_enc = (wl + wr) / 2, (wr - wl) / d["track_ring"]

        self.omega_gyro = math.radians(snap.gyro_yaw_dps) if _finite(snap.gyro_yaw_dps) else math.nan
        dth = self.omega_gyro * dt if math.isfinite(self.omega_gyro) else dth_enc
        self.ds, self.dtheta = ds, dth
        self.v = ds / dt if dt > 0 else 0.0
        self.omega_enc = dth_enc / dt if dt > 0 else 0.0
        mid = self.theta + dth / 2
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta += dth
        self.history.append((snap.t, self.x, self.y, self.theta))

    def pose_at(self, t):
        """Dead-reckoned (x, y, theta) at time t (nearest stored sample at or before t)."""
        if not self.history:
            return 0.0, 0.0, 0.0
        past = self.history[0]
        for h in self.history:
            if h[0] > t:
                break
            past = h
        return past[1], past[2], past[3]

    def motion_since(self, t):
        """Motion from time t to now, in the robot's body frame at time t: (forward, left, dtheta)."""
        if not self.history:
            return 0.0, 0.0, 0.0
        past = self.history[0]
        for h in self.history:
            if h[0] > t:
                break
            past = h
        now = self.history[-1]
        dx, dy = now[1] - past[1], now[2] - past[2]
        c, s = math.cos(-past[3]), math.sin(-past[3])
        return dx * c - dy * s, dx * s + dy * c, now[3] - past[3]


# ---------------- pose EKF ----------------

class PoseEKF:
    """State [x, y, theta, ex, ey]: pose in ring units + external (unexplained) velocity."""

    TAU_EXT = 0.8      # external velocity fades back to 0 with this time constant
    Q_EXT = 0.6        # ...and can change this fast (ring units/s per sqrt(s))

    def __init__(self):
        self.s = np.zeros(5)
        self.P = np.diag([1.0, 1.0, math.pi ** 2, 0.01, 0.01])
        self.initialized = False
        self._suspect = 0
        self.last_innovation = np.zeros(3)

    @property
    def sigma_pos(self):
        return math.sqrt(max(self.P[0, 0], self.P[1, 1]))

    def predict(self, ds, dtheta, dt):
        x, y, th, ex, ey = self.s
        mid = th + dtheta / 2
        a = math.exp(-dt / self.TAU_EXT) if dt > 0 else 1.0
        self.s = np.array([x + ds * math.cos(mid) + ex * dt,
                           y + ds * math.sin(mid) + ey * dt,
                           wrap(th + dtheta), ex * a, ey * a])
        F = np.eye(5)
        F[0, 2], F[1, 2] = -ds * math.sin(mid), ds * math.cos(mid)
        F[0, 3] = F[1, 4] = dt
        F[3, 3] = F[4, 4] = a
        qp = config.EKF_Q_POS ** 2 * dt + (0.1 * ds) ** 2
        qt = config.EKF_Q_THETA ** 2 * dt + (0.05 * dtheta) ** 2
        qe = self.Q_EXT ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag([qp, qp, qt, qe, qe])

    def update_camera(self, x, y, theta, n_tags, age=0.0):
        """Camera pose moved forward to now by wheel odometry; `age` = how old the frame is.

        The frame doesn't include external motion during those `age` seconds, so it
        measures (x - ex*age, y - ey*age, theta). Returns the innovation.
        """
        z = np.array([x, y, theta])
        if not self.initialized:
            self.s[:3] = z
            self.P = np.diag([0.02, 0.02, 0.05, 0.01, 0.01]) ** 2
            self.initialized = True
            return np.zeros(3)
        scale = 1.0 if n_tags >= 2 else config.EKF_SINGLE_TAG_SCALE
        R = np.diag([config.EKF_R_CAM_POS, config.EKF_R_CAM_POS, config.EKF_R_CAM_THETA]) ** 2 * scale ** 2
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        H[0, 3] = H[1, 4] = -age
        nu = z - H @ self.s
        nu[2] = wrap(nu[2])
        self.last_innovation = nu
        # A big jump while we're confident is probably a misread tag: skip it, unless it repeats.
        if math.hypot(nu[0], nu[1]) > config.EKF_GATE_POS and self.sigma_pos < config.EKF_GATE_POS / 2:
            self._suspect += 1
            if self._suspect < config.EKF_GATE_RESET_COUNT:
                return nu
            self.s[:3] = z
            self.s[3:] = 0.0
            self.P = np.diag([0.03, 0.03, 0.08, 0.05, 0.05]) ** 2
            self._suspect = 0
            return nu
        self._suspect = 0
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ nu
        self.s[2] = wrap(self.s[2])
        self.P = (np.eye(5) - K @ H) @ self.P
        return nu


# ---------------- opponent tracker ----------------

class OpponentTracker:
    """Constant-velocity Kalman filter, state [x, y, vx, vy]."""

    def __init__(self):
        self.s = None
        self.P = None
        self.t = None
        self.radius = 0.15
        self.last_seen = -1e9

    def update(self, x, y, radius, t):
        if self.s is None or t - self.last_seen > config.OPP_TIMEOUT_S:
            self.s = np.array([x, y, 0.0, 0.0])
            self.P = np.diag([0.05, 0.05, 0.5, 0.5]) ** 2
            self.t = self.last_seen = t
            self.radius = radius
            return
        self._predict(max(0.0, t - self.t))
        self.t = max(self.t, t)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * config.OPP_R_POS ** 2
        nu = np.array([x, y]) - H @ self.s
        if math.hypot(*nu) > 0.6:              # jumped: a different blob; start over there
            self.s = np.array([x, y, 0.0, 0.0])
            self.P = np.diag([0.05, 0.05, 0.5, 0.5]) ** 2
        else:
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.s = self.s + K @ nu
            self.P = (np.eye(4) - K @ H) @ self.P
        self.radius += 0.2 * (radius - self.radius)
        self.last_seen = t

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = config.OPP_Q_ACCEL ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + G @ G.T * q

    def state(self, now):
        """(x, y, vx, vy) extrapolated to now, or None if not seen recently."""
        if self.s is None or now - self.last_seen > config.OPP_TIMEOUT_S:
            return None
        dt = max(0.0, now - self.t)
        x, y, vx, vy = self.s
        return (x + vx * dt, y + vy * dt, vx, vy)


# ---------------- aggression ----------------

class Aggression:
    """Shake intensity from the UNO Q over MQTT -> aggression 0..1."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0.0
        self._last_t = -1e9
        self._decay_from = None
        self.raw = None
        self.fault = False

    def on_message(self, payload, now):
        if not payload.startswith(config.SHAKE_PREFIX):
            return
        body = payload[len(config.SHAKE_PREFIX):].strip()
        with self._lock:
            if body == "err":
                self.fault = True
                self._last_t = now
                self.raw = None
                self._value *= 0.5
                return
            try:
                x = clip(float(body), 0.0, 1.0)
            except ValueError:
                return
            self.fault = False
            self.raw = x
            self._value += config.SHAKE_EMA * (x - self._value)
            self._last_t = now

    def value(self, now):
        with self._lock:
            quiet = now - self._last_t
            if quiet <= config.SHAKE_TIMEOUT_S:
                return self._value
            return self._value * math.exp(-(quiet - config.SHAKE_TIMEOUT_S) / config.SHAKE_DECAY_S)

    def connected(self, now):
        return now - self._last_t <= config.SHAKE_TIMEOUT_S


# ---------------- fused state ----------------

@dataclass
class FusedState:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0                  # forward speed from the wheels
    vel: tuple = (0.0, 0.0)         # true world velocity: wheels + external
    ext: tuple = (0.0, 0.0)         # external (pushed) velocity
    sigma_pos: float = 1.0
    pose_confident: bool = False
    camera_ok: bool = False
    tags_seen: int = 0
    opp: tuple = None               # (x, y, vx, vy)
    opp_radius: float = 0.15
    closing_speed: float = 0.0
    edge_risk: float = 0.0
    edge_end: str = "front"         # which end is in danger
    edge_dist_risk: float = 0.0
    color: str = "?"
    push_prob: float = 0.0
    pushed: bool = False
    push_dir: tuple = (0.0, 0.0)    # direction we're being moved
    cues: dict = field(default_factory=dict)
    aggression: float = 0.0
    shake_connected: bool = False
    lost: bool = False
    lose_votes: dict = field(default_factory=dict)
    drive_warning: str = None
    pushing_out: bool = False       # set by the Brain: shoving the opponent out (tape rules relaxed)
    our_radius: float = 0.2
    length: float = 0.4

    @property
    def r(self):
        return math.hypot(self.x, self.y)

    @property
    def heading(self):
        return (math.cos(self.theta), math.sin(self.theta))


# ---------------- push detector ----------------

class PushDetector:
    def __init__(self):
        self.prob = 0.0
        self.pushed = False
        self._grav = None
        self._jolt = 0.0
        self._yaw = 0.0
        self._w_gyro = self._w_enc = 0.0
        self._stall = 0.0
        self._last_imu_t = None
        self._prev_v = 0.0

    def step(self, now, dt, snap, odo, resid, camera_ok, us, opp, opp_r, our_r, closing, true_speed, ring_radius_m):
        cues = {}
        # 1. motion the wheels don't explain: camera displacement minus odometry displacement
        #    over the last PUSH_WINDOW_S (None when the camera hasn't seen us long enough)
        resid_speed = math.hypot(*resid) if resid is not None else 0.0
        turn_rate = abs(odo.omega_gyro) if math.isfinite(odo.omega_gyro) else abs(odo.omega_enc)
        resid_unit = config.PUSH_RESID_SPEED * (1 + 0.15 * turn_rate)      # rad/s
        cues["resid"] = clip(resid_speed / resid_unit, 0, 2) if camera_ok else 0.0

        # 2. IMU jolt beyond what our own driving explains
        a = snap.accel_mg
        a_own_mg = abs(odo.v - self._prev_v) / dt * ring_radius_m / 9.81 * 1000 if dt > 0 else 0.0
        self._prev_v = odo.v
        if _finite(*a) and snap.imu_t != self._last_imu_t:
            self._last_imu_t = snap.imu_t
            av = np.array(a)
            self._grav = av if self._grav is None else self._grav + 0.05 * (av - self._grav)
            dyn = float(np.linalg.norm(av - self._grav))
            self._jolt = max(self._jolt, max(0.0, dyn - a_own_mg - 60) / config.PUSH_ACCEL_MG)
        self._jolt *= math.exp(-dt / 0.3)
        cues["accel"] = clip(self._jolt, 0, 2)

        # 3. turning the wheels don't explain (gyro vs encoders)
        if math.isfinite(odo.omega_gyro) and odo.encoders_ok:
            k = 1 - math.exp(-dt / 0.15)
            self._w_gyro += k * (odo.omega_gyro - self._w_gyro)
            self._w_enc += k * (odo.omega_enc - self._w_enc)
            rate = max(abs(self._w_gyro), abs(self._w_enc))
            yaw_err = max(0.0, abs(math.degrees(self._w_gyro - self._w_enc)) - 0.3 * math.degrees(rate))
            self._yaw += (1 - math.exp(-dt / 0.2)) * (yaw_err - self._yaw)
        else:
            self._yaw *= math.exp(-dt / 0.2)
        cues["yaw"] = clip(self._yaw / config.PUSH_YAW_DPS, 0, 2)

        # 4. opponent touching us (and closing in)
        if opp is not None:
            gap = math.hypot(opp[0] - us[0], opp[1] - us[1]) - (opp_r + our_r)
            touch = clip((config.PUSH_CONTACT_GAP - gap) / config.PUSH_CONTACT_GAP, 0, 1)
            cues["contact"] = touch * (1 + clip(closing / 0.2, 0, 1))
        else:
            cues["contact"] = 0.0

        # 5. stalled: driving hard but going nowhere
        cmd = (abs(snap.cmd_left) + abs(snap.cmd_right)) / 2
        stalled = cmd >= config.PUSH_STALL_CMD and true_speed < config.PUSH_STALL_SPEED and camera_ok
        self._stall += (1 - math.exp(-dt / 0.3)) * ((1.0 if stalled else 0.0) - self._stall)
        cues["stall"] = self._stall

        L = config.PUSH_BIAS + sum(config.PUSH_WEIGHTS[k] * v for k, v in cues.items())
        if opp is not None:
            gap = math.hypot(opp[0] - us[0], opp[1] - us[1]) - (opp_r + our_r)
            far = clip((gap - config.PUSH_FAR_GAP) / config.PUSH_FAR_GAP, 0, 1)
            cues["far"] = -far
            L -= config.PUSH_FAR_WEIGHT * far
        self.prob = 1 / (1 + math.exp(-L))
        if self.pushed and self.prob < config.PUSH_OFF:
            self.pushed = False
        elif not self.pushed and self.prob > config.PUSH_ON:
            self.pushed = True

        if resid_speed > 0.08 and camera_ok:
            direction = unit(resid)
        elif opp is not None:
            direction = unit((us[0] - opp[0], us[1] - opp[1]))
        else:
            direction = (0.0, 0.0)
        return self.prob, self.pushed, direction, cues


# ---------------- edge risk ----------------

def edge_risk(st, color, half_len, use_position):
    """Fused risk 0..1 of going over the tape, and which end ('front'/'back') is at risk."""
    hx, hy = st.heading
    sigma = min(st.sigma_pos, 0.1)
    best = (0.0, "front", 0.0)
    for end, sign in (("front", 1), ("back", -1)):
        risk = dist_risk = 0.0
        if use_position:
            px, py = st.x + sign * half_len * hx, st.y + sign * half_len * hy
            r = math.hypot(px, py)
            dist = 1.0 - r - sigma
            dist_risk = clip(1 - dist / config.EDGE_SAFE_DIST, 0, 1.2) * 0.75
            if r > 1e-6:
                vr = (st.vel[0] * px + st.vel[1] * py) / r
                if vr > 0.02:
                    tte = max(dist, 0.0) / vr
                    risk = clip(1 - tte / config.EDGE_TTE_SAFE_S, 0, 1)
            risk = max(risk, dist_risk)
        if end == "front" and color == BLUE:
            risk = 1.0
        if risk > best[0]:
            best = (risk, end, dist_risk)
    return best


# ---------------- lose detector ----------------

class LoseDetector:
    def __init__(self):
        self.out_since = None
        self.blue_t = -1e9
        self.color_vote_t = -1e9
        self.lost = False
        self.votes = {}

    def reset(self):
        self.__init__()

    def step(self, now, cam_out_fraction, cam_fresh, st, color, moving_forward, nose_outward, nose_r):
        if self.lost:
            return True
        cam_out = cam_fresh and cam_out_fraction is not None and cam_out_fraction > config.LOSE_OUT_FRACTION
        if cam_out:
            self.out_since = now if self.out_since is None else self.out_since
        elif cam_fresh and cam_out_fraction is not None:
            self.out_since = None
        if color == BLUE:
            self.blue_t = now
        elif (color == GRAY and now - self.blue_t < config.LOSE_COLOR_WINDOW_S
              and moving_forward and nose_outward and nose_r > 0.9):
            self.color_vote_t = now
        votes = {
            "camera": bool(cam_out),
            "position": st.r > config.LOSE_FUSED_R and st.sigma_pos < 0.15,
            "color": now - self.color_vote_t < config.LOSE_COLOR_WINDOW_S,
        }
        self.votes = votes
        alone = self.out_since is not None and now - self.out_since >= config.LOSE_CAMERA_ALONE_S
        self.lost = alone or sum(votes.values()) >= 2
        return self.lost


# ---------------- drive check ----------------

class DriveCheck:
    """Compares what we told the motors with what the camera saw.

    If "forward" makes the robot go backward, a left-turn command turns it right, or the
    gyro turns the opposite way to the camera, the drive calibration is missing or wrong.
    Fusion and steering are then working from a mirror image of reality: the robot spins
    "confused" or drives straight out. The match is stopped with a message to press `d`.

    Each check is a correlation over the last second between the command (as it was
    DRIVE_CHECK_LAG_S earlier, the Bluetooth + motor delay) and the camera's motion, so the
    back-and-forth of a confused robot doesn't average itself away.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.cmds = deque(maxlen=400)    # (t, cmd forward, cmd turn)
        self.cams = deque(maxlen=200)    # (t, x, y, theta, odometry theta)
        self.scores = {"dir": 0, "turn": 0, "gyro": 0}
        self.last_eval = -1e9
        self.warning = None

    def add_cmd(self, t, fwd, turn):
        self.cmds.append((t, fwd, turn))

    def _cmd_at(self, t):
        best = None
        for c in self.cmds:
            if c[0] > t:
                break
            best = c
        return best

    @staticmethod
    def _corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        den = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
        return float(np.sum(a * b)) / den if den > 1e-12 else 0.0

    def add(self, t, x, y, theta, odo_theta):
        self.cams.append((t, x, y, theta, odo_theta))
        if t - self.last_eval < config.DRIVE_CHECK_WINDOW_S:
            return
        win = [c for c in self.cams if t - c[0] <= config.DRIVE_CHECK_SPAN_S]
        if len(win) < 8:
            return
        self.last_eval = t
        cf, ct, va, wc, wo = [], [], [], [], []
        for a, b in zip(win, win[1:]):
            dt = b[0] - a[0]
            if dt < 0.015:
                continue
            cmd = self._cmd_at((a[0] + b[0]) / 2 - config.DRIVE_CHECK_LAG_S)
            if cmd is None:
                continue
            cf.append(cmd[1])
            ct.append(cmd[2])
            va.append(((b[1] - a[1]) * math.cos(a[3]) + (b[2] - a[2]) * math.sin(a[3])) / dt)
            wc.append(wrap(b[3] - a[3]) / dt)
            wo.append((b[4] - a[4]) / dt)
        if len(cf) < 6:
            return
        rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
        mean_abs = lambda v: sum(abs(x) for x in v) / len(v)
        checks = (
            ("dir", cf, va, mean_abs(cf) >= config.DRIVE_CHECK_MIN_CMD and rms(va) >= config.DRIVE_CHECK_MIN_SPEED),
            ("turn", ct, wc, mean_abs(ct) >= config.DRIVE_CHECK_MIN_TURN_CMD and rms(wc) >= config.DRIVE_CHECK_MIN_TURN),
            ("gyro", wo, wc, rms(wc) >= config.DRIVE_CHECK_MIN_TURN),
        )
        for name, cmd, seen, excited in checks:
            if not excited:
                continue
            c = self._corr(cmd, seen)
            step = 2 if abs(c) > 0.6 else 1           # a clear answer counts double
            if c > 0.3:
                self.scores[name] = min(self.scores[name] + step, 10)
            elif c < -0.3:
                self.scores[name] = max(self.scores[name] - step, -10)
        fail = -config.DRIVE_CHECK_FAIL
        if self.scores["turn"] <= fail:
            self.warning = "robot turns the WRONG WAY (left/right swapped): press d"
        elif self.scores["dir"] <= fail:
            self.warning = "robot drives BACKWARD when told forward: press d"
        elif self.scores["gyro"] <= fail:
            self.warning = "turning is measured backwards (gyro/wheels): press d"


# ---------------- all together ----------------

class Fusion:
    def __init__(self, geometry, drive):
        self.geom = geometry
        self.odo = Odometry(drive)
        self.ekf = PoseEKF()
        self.opp = OpponentTracker()
        self.push = PushDetector()
        self.lose = LoseDetector()
        self.pushing_out = False             # the Brain's flag from the last tick
        self.aggression = Aggression()
        self.drive_check = DriveCheck()
        self.state = FusedState()
        self._last_t = None
        self._last_cam_t = -1e9
        self._last_meas_t = None
        self._cam_out = None
        self._tags_seen = 0
        self._cam_hist = deque(maxlen=60)     # (t, camera x, y, odometry x, y) for the push residual

    def step(self, now, snap, meas=None):
        dt = 1.0 / config.CONTROL_HZ if self._last_t is None else clip(now - self._last_t, 1e-3, 0.2)
        self._last_t = now
        self.odo.step(snap, dt)
        self.ekf.predict(self.odo.ds, self.odo.dtheta, dt)
        self.drive_check.add_cmd(now, (snap.cmd_left + snap.cmd_right) / 2, (snap.cmd_right - snap.cmd_left) / 2)

        cam_fresh = False
        if meas is not None and meas.t != self._last_meas_t:
            self._last_meas_t = meas.t
            cam_fresh = True
            if meas.our is not None:
                # Move the late frame forward to now by our own wheel motion since then. Any
                # external (pushed) motion during that time is in the EKF's measurement model.
                fwd, left, dth = self.odo.motion_since(meas.t)
                th = meas.our.theta
                x = meas.our.x + fwd * math.cos(th) - left * math.sin(th)
                y = meas.our.y + fwd * math.sin(th) + left * math.cos(th)
                self.ekf.update_camera(x, y, wrap(th + dth), meas.our.n_tags, max(0.0, now - meas.t))
                self._last_cam_t = now
                self._tags_seen = meas.our.n_tags
                ox, oy, oth = self.odo.pose_at(meas.t)
                self._cam_hist.append((meas.t, meas.our.x, meas.our.y, ox, oy))
                self.drive_check.add(meas.t, meas.our.x, meas.our.y, meas.our.theta, oth)
            else:
                self._tags_seen = 0
            self._cam_out = meas.our_out_fraction
            if meas.opp is not None:
                self.opp.update(meas.opp[0], meas.opp[1], meas.opp[2], meas.t)

        g = self.geom
        st = FusedState(t=now)
        x, y, th, ex, ey = self.ekf.s
        st.x, st.y, st.theta = float(x), float(y), float(th)
        st.ext = (float(ex), float(ey))
        st.v = self.odo.v
        st.vel = (st.v * math.cos(th) + ex, st.v * math.sin(th) + ey)
        st.sigma_pos = self.ekf.sigma_pos
        st.camera_ok = self.ekf.initialized and now - self._last_cam_t < config.TAG_TIMEOUT_S
        st.pose_confident = st.camera_ok and st.sigma_pos < config.POSE_CONFIDENT_SIGMA
        st.tags_seen = self._tags_seen if st.camera_ok else 0
        st.our_radius, st.length = g.radius, g.length

        st.opp = self.opp.state(now)
        st.opp_radius = self.opp.radius
        if st.opp is not None:
            # How fast the OPPONENT is coming at us (our own charge doesn't count: ramming
            # it isn't being pushed, and defense wants to know if it is charging).
            u = unit((st.x - st.opp[0], st.y - st.opp[1]))
            st.closing_speed = st.opp[2] * u[0] + st.opp[3] * u[1]

        st.color = snap.color.label
        st.aggression = self.aggression.value(now)
        st.shake_connected = self.aggression.connected(now)

        resid = self._push_residual(now, st.theta)
        if resid is not None and st.camera_ok:
            # true world velocity: wheels + what the camera says they don't explain
            st.vel = (st.v * math.cos(th) + resid[0], st.v * math.sin(th) + resid[1])
        true_speed = math.hypot(*st.vel)
        ring_radius_m = g.ring_diameter_cm / 200.0
        st.push_prob, st.pushed, st.push_dir, st.cues = self.push.step(
            now, dt, snap, self.odo, resid, st.camera_ok, (st.x, st.y), st.opp, st.opp_radius,
            st.our_radius, st.closing_speed, true_speed, ring_radius_m)

        st.edge_risk, st.edge_end, st.edge_dist_risk = edge_risk(st, st.color, g.length / 2, st.camera_ok)

        hx, hy = st.heading
        nose = (st.x + hx * g.length / 2, st.y + hy * g.length / 2)
        nose_r = math.hypot(*nose)
        nose_outward = nose_r > 1e-6 and (st.vel[0] * nose[0] + st.vel[1] * nose[1]) > 0
        # Pushing the opponent out takes our color sensor over the tape on purpose: that isn't us leaving.
        st.lost = self.lose.step(now, self._cam_out, cam_fresh, st, st.color, st.v > 0.05,
                                 nose_outward and not self.pushing_out, nose_r)
        st.lose_votes = dict(self.lose.votes)
        st.drive_warning = self.drive_check.warning
        self.state = st
        return st

    def _push_residual(self, now, theta):
        """Velocity the wheels don't explain, from the camera over the last PUSH_WINDOW_S."""
        h = self._cam_hist
        if len(h) < 4 or now - h[-1][0] > config.TAG_TIMEOUT_S:
            return None
        t_end = h[-1][0]
        start = None
        for item in h:
            if t_end - item[0] <= config.PUSH_WINDOW_S:
                start = item
                break
        if start is None or t_end - start[0] < 0.6 * config.PUSH_WINDOW_S:
            return None
        end = h[-1]
        # odometry lives in its own frame; rotate its displacement into the ring frame
        rot = theta - self.odo.theta
        c, s = math.cos(rot), math.sin(rot)
        odx, ody = end[3] - start[3], end[4] - start[4]
        odx, ody = odx * c - ody * s, odx * s + ody * c
        span = end[0] - start[0]
        return ((end[1] - start[1] - odx) / span, (end[2] - start[2] - ody) / span)
