"""The sumo decision tree (see PLAN.md section 5) plus the `Brain` that ties it to fusion.

Every control tick, top to bottom, the first check that says yes decides:
    1 waiting for start      -> IDLE / COUNTDOWN
    2 already lost           -> HOLD
    3 left the ring          -> LOST (stop, lose.mp3, sumo:lost)
    4 edge escape underway   -> keep escaping
    5 on the tape / about to -> EDGE_ESCAPE (brake, back off, turn to center or spin 135)
    6 being pushed           -> PUSHED_SLIP (near edge) / PUSHED_BACK
    7 camera lost our tags   -> BLIND
    8 shake < 0.15           -> DEFENSE_HOME / DEFENSE_FACE / DEFENSE_HOLD (stay still at the center)
    9 otherwise              -> ATTACK_SEARCH / ATTACK_CHARGE / ATTACK_AIM
Then every moving output goes through the edge speed limiter and the speed ramp: commands
stay at CRUISE_SPEED or below unless the opponent is resisting us (touching, and our wheels
drive but we barely move, or it is pushing us); then the cap ramps up toward MAX_DRIVE_SPEED.

The color-sensor reflex (`Brain.reflex`) runs much faster than the tick, from its
own thread in sumo.py, and can start an edge escape at any moment.
"""

import math
import threading
from dataclasses import dataclass, field

import config
from fusion import Fusion, clip, unit, wrap
from robot_io import BLUE


@dataclass
class Command:
    left: float = 0.0
    right: float = 0.0
    state: str = "IDLE"
    note: str = ""
    target: tuple = None            # point we're steering at (for the overlay)
    events: list = field(default_factory=list)


# ---------------- steering helpers ----------------

def drive_heading(st, target_angle, speed, allow_reverse=False, min_fwd=0.0, turn_max=None):
    """Tank speeds that turn toward `target_angle` and drive at `speed` once roughly facing it.

    min_fwd keeps that fraction of the speed even while still turning (an arc, not a spin).
    """
    err = wrap(target_angle - st.theta)
    if allow_reverse and abs(err) > math.radians(100):
        err = wrap(err + math.pi)       # back up toward it instead of turning all the way round
        speed = -speed
    tmax = config.TURN_MAX if turn_max is None else turn_max
    turn = clip(config.TURN_KP * err, -tmax, tmax)
    fwd = speed * max(0.0, math.cos(err)) if abs(err) < math.radians(60) else 0.0
    if abs(fwd) < abs(speed) * min_fwd:
        fwd = speed * min_fwd
    return fwd - turn, fwd + turn


def bearing(st, point):
    return math.atan2(point[1] - st.y, point[0] - st.x)


def angle_of(v):
    return math.atan2(v[1], v[0])


def ray_to_circle(p, d, radius):
    """Distance along unit direction d from point p to the circle of `radius` (0 if already outside)."""
    pd = p[0] * d[0] + p[1] * d[1]
    c = p[0] * p[0] + p[1] * p[1] - radius * radius
    if c >= 0:
        return 0.0
    return -pd + math.sqrt(pd * pd - c)


def allowed_speed(dist):
    """Fastest speed (ring units/s) that can still stop within `dist`, counting the command delay."""
    if dist <= 0:
        return 0.0
    a, L = config.STOP_DECEL, config.STOP_LATENCY_S
    return a * (-L + math.sqrt(L * L + 2 * dist / a))


def pushing_ahead(st):
    """Our front is against the opponent (it's right in front of us, touching)."""
    if st.opp is None:
        return False
    dx, dy = st.opp[0] - st.x, st.opp[1] - st.y
    gap = math.hypot(dx, dy) - (st.opp_radius + st.our_radius)
    ahead = abs(wrap(math.atan2(dy, dx) - st.theta)) < math.radians(35)
    return ahead and gap < 2 * config.PUSH_CONTACT_GAP


def limit_speed(st, left, right, drive):
    """Edge speed limiter, applied to every moving command.

    With the camera: cap the forward/backward speed so the robot could still stop before its
    leading end (nose going forward, tail going backward) reaches the tape along the way it is
    heading. Without the camera: a flat cap, slow enough for the color reflex to catch the tape.
    """
    v, w = (left + right) / 2, (right - left) / 2
    if not st.camera_ok:
        cap = config.BLIND_MAX_SPEED
    else:
        if abs(v) < 1e-6:
            return left, right
        hx, hy = st.heading
        sign = 1 if v > 0 else -1
        end = (st.x + sign * st.length / 2 * hx, st.y + sign * st.length / 2 * hy)
        dist = ray_to_circle(end, (sign * hx, sign * hy), 1.0 - config.STOP_MARGIN)
        ring_per_s_per_pct = drive["wheel_dps_per_pct"] * drive["ring_per_wheel_deg"]
        # An arc that is swinging us inward has more room than the straight line (sideways at
        # the tape, a straight line hits it at once but the arc is how we get away).
        omega = 2 * w * ring_per_s_per_pct / max(drive["track_ring"], 1e-6)
        th2 = st.theta + omega * config.LIMIT_TURN_LOOKAHEAD_S
        d2 = (sign * math.cos(th2), sign * math.sin(th2))
        dist = max(dist, ray_to_circle(end, d2, 1.0 - config.STOP_MARGIN))
        cap = allowed_speed(dist) / max(ring_per_s_per_pct, 1e-6)
        if cap < 4:
            cap = 0.0
        if sign > 0 and dist > 0 and pushing_ahead(st):
            # Shoving the opponent: full power is fine while we're actually moving slowly enough
            # to stop in time. If it drops out and we speed up, the normal cap returns next tick.
            speed = st.vel[0] * hx + st.vel[1] * hy
            if speed < allowed_speed(dist):
                cap = config.MAX_DRIVE_SPEED
    if abs(v) > cap:
        v = math.copysign(cap, v)
    return v - w, v + w


def cap_speed(left, right, cap):
    """Scale both sides down together so neither is faster than `cap`."""
    m = max(abs(left), abs(right))
    if m > cap:
        left, right = left * cap / m, right * cap / m
    return left, right


def escape_room(st, direction):
    """How far (ring units) we can drive straight forward (+1) or backward (-1) before the
    leading end reaches the tape; 0 if the opponent is touching us on that side."""
    hx, hy = st.heading
    d = (direction * hx, direction * hy)
    end = (st.x + d[0] * st.length / 2, st.y + d[1] * st.length / 2)
    room = ray_to_circle(end, d, 1.0 - config.STOP_MARGIN)
    if st.opp is not None:
        dx, dy = st.opp[0] - st.x, st.opp[1] - st.y
        gap = math.hypot(dx, dy) - (st.opp_radius + st.our_radius)
        if gap < config.RESIST_TOUCH_GAP and (dx * d[0] + dy * d[1]) > 0.5 * math.hypot(dx, dy):
            room = 0.0
    return room


# ---------------- timed maneuvers ----------------

class EdgeEscape:
    """Brake, back away from the endangered end, then turn to face the center (or spin 135)."""

    def __init__(self, now, end, st, color_front=False):
        self.t0 = now
        self.end = end
        self.away = -1 if end == "front" else 1      # drive away from the tape
        self.arc = False
        if not color_front and st.camera_ok:
            # Back off whichever way has more room: sideways to the tape, "away from the risky
            # end" can still be blocked by the tape or by the opponent shoving us from there.
            room = {d: escape_room(st, d) for d in (1, -1)}
            self.away = 1 if room[1] > room[-1] else -1
            # sideways at the tape: no straight line gets away, so arc toward the center
            self.arc = max(room.values()) < config.ESCAPE_MIN_ROOM
        self.turn_t0 = None
        self.spun = 0.0
        self.prev_theta = st.theta
        self.blind = not st.pose_confident
        self.spin_dir = 1

    def step(self, now, st):
        """(left, right, phase) or None when finished."""
        t = now - self.t0
        if t < config.ESCAPE_BRAKE_S:
            return 0, 0, "brake"
        if t < config.ESCAPE_BRAKE_S + config.ESCAPE_BACK_S:
            # being shoved toward the tape: back off as hard as the speed ramp allows
            speed = config.PUSH_INTENT_SPEED if st.pushed else config.ESCAPE_BACK_SPEED
            if self.arc:
                l, r = drive_heading(st, math.atan2(-st.y, -st.x), speed, allow_reverse=True, min_fwd=0.6,
                                     turn_max=config.PUSHED_TURN_MAX)
                return l, r, "arc inward"
            s = self.away * speed
            return s, s, "back off"
        if self.turn_t0 is None:
            self.turn_t0, self.prev_theta = now, st.theta
            self.blind = not st.pose_confident
            if not self.blind:
                err = wrap(math.atan2(-st.y, -st.x) - st.theta)
                self.spin_dir = 1 if err >= 0 else -1
        if now - self.turn_t0 > config.ESCAPE_TURN_TIMEOUT_S:
            return None
        self.spun += abs(wrap(st.theta - self.prev_theta))
        self.prev_theta = st.theta
        if self.blind:
            if self.spun >= config.ESCAPE_BLIND_SPIN:
                return None
            s = self.spin_dir * config.ESCAPE_TURN_SPEED
            return -s, s, "spin 135"
        err = wrap(math.atan2(-st.y, -st.x) - st.theta)
        if abs(err) < config.ESCAPE_TURN_TOL:
            return None
        if abs(err) < math.radians(90):
            self.spin_dir = 1 if err > 0 else -1      # close: turn the short way
        # far (facing away): keep the direction picked at the start, or it flip-flops
        s = config.ESCAPE_TURN_SPEED * self.spin_dir
        return -s, s, "turn to center"


# ---------------- decision tree ----------------

class Behavior:
    def __init__(self):
        self.escape = None
        self.slipping = False         # PUSHED near the edge (with hysteresis on r)
        self.homing = False           # DEFENSE: creeping back to the center (hysteresis on r)
        self.facing = False           # DEFENSE: turning to face the opponent (hysteresis on angle)

    def reset(self):
        self.escape = None
        self.slipping = False
        self.homing = False
        self.facing = False

    def start_escape(self, now, end, st, color_front=False):
        self.escape = EdgeEscape(now, end, st, color_front)

    def decide(self, now, st):
        """Steps 4-9 of the tree. Returns a Command (before the speed limiter)."""
        # 4. edge escape already underway
        if self.escape is not None:
            out = self.escape.step(now, st)
            if out is not None and st.pushed and self.escape.turn_t0 is not None:
                out = None       # a slow turn while being shoved loses: fight the push instead
            if out is not None:
                return Command(out[0], out[1], "EDGE_ESCAPE", out[2])
            self.escape = None

        # 5. nose on the tape, or about to reach it
        if st.color == BLUE or st.edge_risk > config.EDGE_ESCAPE_RISK:
            end = "front" if st.color == BLUE else st.edge_end
            self.start_escape(now, end, st, color_front=st.color == BLUE)
            out = self.escape.step(now, st)
            return Command(out[0], out[1], "EDGE_ESCAPE", f"{end} at tape")

        # 6. being pushed
        if st.pushed:
            return self._pushed(st)
        self.slipping = False

        # 7. camera lost our tags
        if not st.camera_ok:
            speed = config.BLIND_BASE_SPEED + config.BLIND_GAIN_SPEED * st.aggression
            turn = 15 * math.sin(now * 0.8)
            return Command(speed - turn, speed + turn, "BLIND", "color reflex only")

        # 8. no shake -> defense
        if st.aggression < config.DEFENSE_BELOW:
            return self._defense(now, st)

        # 9. otherwise -> attack
        self.homing = self.facing = False
        return self._attack(st)

    def _pushed(self, st):
        if st.r > config.PUSH_NEAR_EDGE_R:
            self.slipping = True
        elif st.r < config.PUSH_NEAR_EDGE_R - 0.1:
            self.slipping = False
        if self.slipping:
            p = unit((st.x, st.y))
            t1 = (-p[1], p[0])
            t2 = (p[1], -p[0])
            # Slip along the edge, away from the push's sideways part, angled back inward.
            push = st.push_dir
            tan = t1 if (t1[0] * push[0] + t1[1] * push[1]) <= (t2[0] * push[0] + t2[1] * push[1]) else t2
            want = unit((tan[0] - 0.6 * p[0], tan[1] - 0.6 * p[1]))
            l, r = drive_heading(st, angle_of(want), config.PUSH_INTENT_SPEED, allow_reverse=True,
                                 min_fwd=config.PUSHED_MIN_FWD, turn_max=config.PUSHED_TURN_MAX)
            return Command(l, r, "PUSHED_SLIP", f"p={st.push_prob:.2f}",
                           target=(st.x + want[0] * 0.3, st.y + want[1] * 0.3))
        if st.opp is not None:
            face = bearing(st, st.opp)
            target = st.opp[:2]
        else:
            face = angle_of((-st.push_dir[0], -st.push_dir[1])) if st.push_dir != (0.0, 0.0) else st.theta
            target = None
        # Defense, hit from the side: pivot to square up (driving forward would just arc us
        # sideways). Otherwise keep driving into it while correcting (in attack, keeping the
        # momentum wins more shoves than stopping to pivot -- sim, 2026-10-19).
        side_hit = st.aggression < config.DEFENSE_BELOW and abs(wrap(face - st.theta)) > config.PUSHED_PIVOT_ANGLE
        l, r = drive_heading(st, face, config.PUSH_INTENT_SPEED, min_fwd=0.0 if side_hit else config.PUSHED_MIN_FWD,
                             turn_max=config.PUSHED_TURN_MAX)
        return Command(l, r, "PUSHED_BACK", f"p={st.push_prob:.2f}", target=target)

    def _defense(self, now, st):
        """Stay still at the center. Creep back if knocked off it; turn slowly to face the
        opponent only when it's well off to the side (so a push lands on our front)."""
        if st.r > config.DEFENSE_HOME_R:
            self.homing = True
        elif st.r < config.DEFENSE_SETTLE_R:
            self.homing = False
        if self.homing:
            speed = config.DEFENSE_SPEED * clip(st.r / config.DEFENSE_HOME_R, 0.6, 1.0)
            l, r = drive_heading(st, bearing(st, (0, 0)), speed, allow_reverse=True)
            return Command(l, r, "DEFENSE_HOME", f"r={st.r:.2f}", target=(0.0, 0.0))
        if st.opp is not None:
            err = abs(wrap(bearing(st, st.opp) - st.theta))
            if err > config.DEFENSE_FACE_TOL:
                self.facing = True
            elif err < config.DEFENSE_FACE_DONE:
                self.facing = False
            if self.facing:
                l, r = drive_heading(st, bearing(st, st.opp), 0, turn_max=config.DEFENSE_TURN_MAX)
                return Command(l, r, "DEFENSE_FACE", "", target=st.opp[:2])
        else:
            self.facing = False
        return Command(0, 0, "DEFENSE_HOLD", "holding the center")

    def _attack(self, st):
        a = st.aggression
        speed = config.ATTACK_BASE_SPEED + config.ATTACK_GAIN_SPEED * a
        if st.opp is None:
            if st.r > config.DEFENSE_HOME_R:
                l, r = drive_heading(st, bearing(st, (0, 0)), config.ATTACK_BASE_SPEED)
                return Command(l, r, "ATTACK_SEARCH", "to center", target=(0.0, 0.0))
            s = config.SEARCH_SPIN_SPEED
            return Command(-s, s, "ATTACK_SEARCH", "spinning")
        ox, oy, vx, vy = st.opp
        lead = (ox + vx * config.ATTACK_LEAD_S, oy + vy * config.ATTACK_LEAD_S)
        u = unit(lead) if math.hypot(*lead) > 0.1 else unit((lead[0] - st.x, lead[1] - st.y))
        stand = st.opp_radius + st.our_radius + 0.08
        behind = (lead[0] - u[0] * stand, lead[1] - u[1] * stand)       # center side of the opponent
        from_opp = unit((st.x - lead[0], st.y - lead[1]))
        in_position = (from_opp[0] * -u[0] + from_opp[1] * -u[1]) > math.cos(math.radians(50)) \
            or math.hypot(st.x - behind[0], st.y - behind[1]) < 0.12
        target = lead if in_position else behind
        err = wrap(bearing(st, target) - st.theta)
        if abs(err) < config.ALIGN_TOL:
            if pushing_ahead(st):
                speed = config.PUSH_INTENT_SPEED          # shoving: the resist ramp sets the real speed
            l, r = drive_heading(st, bearing(st, target), speed)
            return Command(l, r, "ATTACK_CHARGE", f"speed {speed:.0f}", target=target)
        l, r = drive_heading(st, bearing(st, target), 0.35 * speed)
        return Command(l, r, "ATTACK_AIM", "lining up", target=target)


# ---------------- speed ramp ----------------

class SpeedRamp:
    """Speed cap: CRUISE_SPEED normally, ramping up only while another robot resists us.

    Resisted (fused): we're being pushed, or the opponent is touching us on the side we're
    driving toward and we move at under RESIST_SPEED_RATIO of the commanded speed for
    RESIST_HOLD_S. Speed comes from the camera when it tracks us (wheels can slip), else
    from the wheel encoders (a loaded motor slows down too).
    """

    def __init__(self):
        self.cap = config.CRUISE_SPEED
        self.resisted = False
        self._slow_since = None

    def reset(self):
        self.__init__()

    def step(self, now, dt, st, last_cmd, drive):
        v_cmd = (last_cmd[0] + last_cmd[1]) / 2
        expected = abs(v_cmd) * drive["wheel_dps_per_pct"] * drive["ring_per_wheel_deg"]
        hx, hy = st.heading
        moving = (st.vel[0] * hx + st.vel[1] * hy) if st.camera_ok else st.v
        moving *= 1 if v_cmd >= 0 else -1
        slow = abs(v_cmd) >= config.RESIST_MIN_CMD and moving < config.RESIST_SPEED_RATIO * expected
        if slow:
            self._slow_since = now if self._slow_since is None else self._slow_since
        else:
            self._slow_since = None
        stalled = self._slow_since is not None and now - self._slow_since >= config.RESIST_HOLD_S
        touching = False
        if st.opp is not None:
            dx, dy = st.opp[0] - st.x, st.opp[1] - st.y
            gap = math.hypot(dx, dy) - (st.opp_radius + st.our_radius)
            ahead = (dx * hx + dy * hy) * (1 if v_cmd >= 0 else -1) > 0
            touching = gap < config.RESIST_TOUCH_GAP and ahead
        self.resisted = st.pushed or (stalled and touching)
        if self.resisted:
            self.cap = max(self.cap, config.RESIST_START_SPEED)     # friction: start strong, then ramp
            self.cap = min(config.MAX_DRIVE_SPEED, self.cap + config.RAMP_UP_PCT_S * dt)
        else:
            self.cap = max(config.CRUISE_SPEED, self.cap - config.RAMP_DOWN_PCT_S * dt)
        return self.cap


# ---------------- brain: fusion + tree + match state ----------------

class Brain:
    """Owns fusion and the decision tree; thread-safe so the reflex thread can cut in."""

    def __init__(self, geometry, drive):
        self.drive = drive
        self.fusion = Fusion(geometry, drive)
        self.behavior = Behavior()
        self.ramp = SpeedRamp()
        self._last_step_t = None
        self.lock = threading.Lock()
        self.match = "idle"            # idle -> countdown -> running -> lost
        self.countdown_until = None
        self.last = Command()

    @property
    def state(self):
        return self.fusion.state

    def start_match(self, now):
        with self.lock:
            if self.match in ("idle", "lost"):
                self.reset_locked()
                self.match = "countdown"
                self.countdown_until = now + config.COUNTDOWN_S

    def reset(self):
        with self.lock:
            self.reset_locked()

    def reset_locked(self):
        self.match = "idle"
        self.fusion.lose.reset()
        self.behavior.reset()
        self.ramp.reset()

    def reflex(self, now, color_label):
        """Fast edge reflex. Returns a (left, right) command to send right now, or None."""
        with self.lock:
            if (self.match == "running" and color_label == BLUE and self.behavior.escape is None):
                self.behavior.start_escape(now, "front", self.fusion.state, color_front=True)
                s = -config.REFLEX_REVERSE_SPEED          # reverse at once: stops faster than braking
                return (s, s)
        return None

    def step(self, now, snap, meas=None):
        with self.lock:
            st = self.fusion.step(now, snap, meas)
            dt = 0.0 if self._last_step_t is None else clip(now - self._last_step_t, 0.0, 0.2)
            self._last_step_t = now
            cap = self.ramp.step(now, dt, st, (snap.cmd_left, snap.cmd_right), self.drive)
            events = []
            # 1. waiting for the match to start
            if self.match == "idle":
                cmd = Command(0, 0, "IDLE", "press space to start")
            elif self.match == "countdown":
                left = self.countdown_until - now
                if left > 0:
                    cmd = Command(0, 0, "COUNTDOWN", f"{left:.1f} s")
                else:
                    self.match = "running"
                    self.fusion.lose.reset()
                    events.append("started")
                    cmd = None
            else:
                cmd = None
            if cmd is None:
                # 2. already lost / 3. lose detected
                if self.match == "lost":
                    cmd = Command(0, 0, "HOLD", "press r to reset")
                elif st.drive_warning:
                    # Driving from a mirror image of reality is how robots leave the ring: stop.
                    self.match = "lost"
                    self.behavior.reset()
                    events.append("drive_check")
                    cmd = Command(0, 0, "DRIVE_CHECK", st.drive_warning)
                elif st.lost:
                    self.match = "lost"
                    self.behavior.reset()
                    events.append("lost")
                    cmd = Command(0, 0, "LOST", "left the ring")
                else:
                    cmd = self.behavior.decide(now, st)
                    # Escapes move away from the tape, but backing off can still head at the
                    # tape behind us, so they're limited too.
                    cmd.left, cmd.right = limit_speed(st, cmd.left, cmd.right, self.drive)
                    # escapes always get at least their own speed (backing off the tape matters)
                    floor = config.ESCAPE_BACK_SPEED if cmd.state == "EDGE_ESCAPE" else 0
                    cmd.left, cmd.right = cap_speed(cmd.left, cmd.right, max(cap, floor))
                    cmd.left, cmd.right = cap_speed(cmd.left, cmd.right, config.MAX_DRIVE_SPEED)
            cmd.events = events
            self.last = cmd
            return cmd
