"""Pure ping pong game logic -- no camera, motor or pygame in here.

World coordinates are in meters, centered on the table:
    x  across the table, -TABLE_HALF_W (my left) .. +TABLE_HALF_W (my right)
    y  height above the table surface
    z  along the table, 0 = my end, TABLE_LEN = CPU's end

The ball flies on scripted arcs (Wii-style arcade physics): each shot goes
from a hit point, bounces once on the receiver's half, and arrives at the
receiver's hitting plane at a known time. Knowing the arrival time up front
is what makes the swing-timing check simple.

The game reports what happened through Game.events, a list of
(name, data) tuples that pingpong.py drains each frame to trigger haptics,
sounds and pop-up text.
"""

import math
import random
from dataclasses import dataclass, field

TABLE_LEN = 2.74
TABLE_HALF_W = 0.7625
NET_Z = TABLE_LEN / 2
NET_H = 0.1525
BALL_R = 0.02

Z_PLAYER = -0.20               # my hitting plane (just behind my end of the table)
Z_CPU = TABLE_LEN + 0.20       # CPU's hitting plane

POINTS_TO_WIN = 11
SERVES_EACH = 2                # serve switches every 2 points

# --- Ball speed from level ---
BASE_BALL_SPEED = 3.0          # m/s along the table at level 1 (~1 s per crossing)
SPEED_PER_LEVEL = 0.15         # +15% per level above 1
MAX_LEVEL = 10

# --- My hit test ---
TIMING_WINDOW = 0.22           # s: swing must land within +/- this of the ball arriving
EARLY_HINT_WINDOW = 0.6        # s: swings this early just show "Too early!" (not consumed)
HIT_RADIUS_X = 0.38            # m: how far the paddle can be from the ball across the table
HIT_RADIUS_Y = 0.30            # m: ...and in height
PADDLE_HISTORY_S = 0.30        # use the paddle's closest point to the ball over this long
PERFECT_QUALITY = 0.75
GOOD_QUALITY = 0.40

# --- Shot shapes ---
ARC_HEIGHT_OVER_NET = 0.30     # m: extra arc on the first half of a shot (clears the net)
ARC_HEIGHT_AFTER_BOUNCE = 0.14
ARRIVAL_HEIGHT = (0.18, 0.30)  # m: height the ball arrives at a hitting plane
REACHABLE_X = 0.85             # m: CPU never aims the ball wider than this at my plane
CPU_REACH = 0.15               # m: CPU paddle within this of the ball returns it (unless it fumbles)

# --- My shot direction from the paddle's IMU orientation ---
# Kept gentle on purpose: the face angle just steers the ball left/right, and it always
# stays on the table, so aiming never makes you miss.
AIM_DEADZONE_DEG = 10.0        # face turned less than this = straight down the middle
AIM_FULL_DEG = 50.0            # face turned this far (or more) = the widest shot
AIM_MAX_X = 0.55               # widest shot lands this fraction of the way to the sideline
OUT_AIM_DEG = None             # None = aiming can never put the ball out (set e.g. 80 to allow it)
TILT_FULL_DEG = 60.0           # face opened/closed this far = maximum (small) loft/drive effect
TILT_EFFECT = 0.25             # 0 = tilt ignored, 1 = strong lob/drive
STROKE_DEADZONE = 0.10         # m: balls this close to my body's center line take either stroke

# --- Timing of the flow between points ---
COUNTDOWN_S = 3.0
POINT_PAUSE_S = 1.4
CPU_SERVE_DELAY_S = 1.0
BALL_GONE_S = 0.6              # after a miss, how long the ball keeps flying before the point ends


def level_ball_speed(level):
    return BASE_BALL_SPEED * (1 + SPEED_PER_LEVEL * (level - 1))


def cpu_move_speed(level):
    return 1.0 + 0.25 * level         # m/s the CPU paddle can slide sideways


def cpu_miss_chance(level):
    return max(0.03, 0.22 - 0.025 * level)


def cpu_spread(level):
    return min(0.70, 0.25 + 0.06 * level)   # how wide the CPU aims at me


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def aim_landing_x(aim_deg):
    """Where (x, meters) a shot with this paddle-face angle bounces on the CPU's side.
    Returns (x, out)."""
    if OUT_AIM_DEG is not None and abs(aim_deg) > OUT_AIM_DEG:
        return math.copysign(TABLE_HALF_W + 0.30, aim_deg), True
    beyond = max(0.0, abs(aim_deg) - AIM_DEADZONE_DEG)
    k = min(1.0, beyond / (AIM_FULL_DEG - AIM_DEADZONE_DEG))
    return math.copysign(k * AIM_MAX_X * TABLE_HALF_W, aim_deg), False


@dataclass
class Shot:
    """One flight of the ball from a hitter to the receiver's hitting plane."""
    t0: float
    x0: float
    y0: float
    z0: float
    xb: float          # bounce point
    zb: float
    x1: float          # arrival point at the receiver's plane
    y1: float
    z1: float
    speed: float       # m/s along z
    to_player: bool
    bounced: bool = False
    arc: float = ARC_HEIGHT_OVER_NET   # m: arc height of the first half (higher = loftier)
    out: bool = False                  # aimed wide: lands off the table
    swung: bool = False   # I already used my one swing on this ball
    cpu_decided: bool = False  # CPU already rolled hit/miss on this ball

    @property
    def duration(self):
        return abs(self.z1 - self.z0) / self.speed

    @property
    def t_arrive(self):
        return self.t0 + self.duration

    def position(self, t):
        """Ball (x, y, z) at time t. Keeps flying (and dropping) past the plane."""
        dist = (t - self.t0) * self.speed
        direction = 1 if self.z1 > self.z0 else -1
        z = self.z0 + direction * dist
        x = self.x0 + (self.x1 - self.x0) * (z - self.z0) / (self.z1 - self.z0)
        seg_a = abs(self.zb - self.z0)
        seg_b = abs(self.z1 - self.zb)
        if dist <= seg_a:
            s = dist / seg_a
            y = self.y0 * (1 - s) + 4 * self.arc * s * (1 - s)
        elif self.out:
            # Missed the table: keeps dropping past the table's height.
            extra = (dist - seg_a) / self.speed
            y = -4.9 * extra * extra - 2.0 * extra
        elif dist <= seg_a + seg_b:
            s = (dist - seg_a) / seg_b
            y = self.y1 * s + 4 * ARC_HEIGHT_AFTER_BOUNCE * s * (1 - s)
        else:
            # Past the receiver: fall away under gravity.
            extra = (dist - seg_a - seg_b) / self.speed
            y = self.y1 - 4.9 * extra * extra
        return x, y, z


def make_shot(t0, x0, y0, z0, zb, xb, speed, to_player, arc=ARC_HEIGHT_OVER_NET):
    """Build a shot that bounces at (xb, zb) and continues straight to the far plane."""
    z1 = Z_PLAYER if to_player else Z_CPU
    x1 = x0 + (xb - x0) * (z1 - z0) / (zb - z0)
    if to_player:
        x1 = clamp(x1, -REACHABLE_X, REACHABLE_X)
    y1 = random.uniform(*ARRIVAL_HEIGHT)
    return Shot(t0, x0, y0, z0, xb, zb, x1, y1, z1, speed, to_player, arc=arc)


@dataclass
class Swing:
    t: float           # time of the swing's peak (time.monotonic clock)
    strength: float    # 0..1
    forehand: bool     # None = either (keyboard test swing)
    aim_deg: float = 0.0    # paddle face turned right (+) / left (-) from facing the CPU
    tilt_deg: float = 0.0   # paddle face opened up (+) / closed down (-)


def required_stroke(ball_x):
    """'forehand' if the ball comes to my right side (I'm right-handed), 'backhand'
    on my left, None near my center line. Paddle x is measured from my body's
    center, so x = 0 is always straight in front of me."""
    if ball_x > STROKE_DEADZONE:
        return "forehand"
    if ball_x < -STROKE_DEADZONE:
        return "backhand"
    return None


@dataclass
class Game:
    level: int = 1
    points_to_win: int = POINTS_TO_WIN
    # WAITING -> SETUP_FLAT (hold paddle flat: zero the angles) -> SETUP_STROKES (3 forehands
    # + 3 backhands) -> COUNTDOWN -> SERVE <-> RALLY <-> POINT_OVER -> GAME_OVER
    state: str = "WAITING"
    state_t: float = 0.0       # time the current state started
    score_player: int = 0
    score_cpu: int = 0
    server_is_player: bool = True
    shot: Shot = None
    cpu_x: float = 0.0
    cpu_target_x: float = 0.0
    rally: int = 0
    best_rally: int = 0
    winner_is_player: bool = None
    last_point_player: bool = None
    paddle_history: list = field(default_factory=list)   # [(t, x, y)]
    events: list = field(default_factory=list)
    _elapsed_now: float = 0.0

    # ---------------- flow ----------------

    def start(self, now, level):
        self.level = clamp(level, 1, MAX_LEVEL)
        self.score_player = self.score_cpu = 0
        self.server_is_player = True
        self.rally = self.best_rally = 0
        self.shot = None
        self.winner_is_player = None
        self._set_state("SETUP_FLAT", now)
        self.events.append(("game_start", None))

    def begin_strokes(self, now):
        self._set_state("SETUP_STROKES", now)

    def begin_countdown(self, now):
        self._set_state("COUNTDOWN", now)

    def reset(self):
        self.state = "WAITING"
        self.shot = None

    def _set_state(self, state, now):
        self.state = state
        self.state_t = now

    @property
    def ball_speed(self):
        return level_ball_speed(self.level)

    @property
    def countdown_left(self):
        return COUNTDOWN_S - self._elapsed

    def update(self, now, dt, paddle_xy):
        """Advance the game. paddle_xy is my paddle (x, y) in meters, or None."""
        self._elapsed_now = now
        if paddle_xy is not None:
            self.paddle_history.append((now, paddle_xy[0], paddle_xy[1]))
        self.paddle_history = [p for p in self.paddle_history if now - p[0] <= PADDLE_HISTORY_S]

        if self.state == "COUNTDOWN" and self._elapsed >= COUNTDOWN_S:
            self._begin_serve(now)
        elif self.state == "SERVE" and not self.server_is_player:
            self._move_cpu(dt, 0.0)
            if self._elapsed >= CPU_SERVE_DELAY_S:
                self._cpu_hit(now, self.cpu_x, 0.25, Z_CPU, serve=True)
                self._set_state("RALLY", now)
        elif self.state == "RALLY":
            self._update_rally(now, dt)
        elif self.state == "POINT_OVER" and self._elapsed >= POINT_PAUSE_S:
            if self._game_won():
                self.winner_is_player = self.score_player > self.score_cpu
                self._set_state("GAME_OVER", now)
                self.events.append(("game_over", self.winner_is_player))
            else:
                self._begin_serve(now)

    @property
    def _elapsed(self):
        return self._elapsed_now - self.state_t

    def _begin_serve(self, now):
        total = self.score_player + self.score_cpu
        self.server_is_player = (total // SERVES_EACH) % 2 == 0
        # At deuce (10-10 and beyond) the serve alternates every point.
        if self.score_player >= self.points_to_win - 1 and self.score_cpu >= self.points_to_win - 1:
            self.server_is_player = total % 2 == 0
        self.shot = None
        self.rally = 0
        self._set_state("SERVE", now)
        self.events.append(("serve_start", self.server_is_player))

    def _game_won(self):
        hi, lo = max(self.score_player, self.score_cpu), min(self.score_player, self.score_cpu)
        return hi >= self.points_to_win and hi - lo >= 2

    def _award_point(self, now, to_player):
        if to_player:
            self.score_player += 1
        else:
            self.score_cpu += 1
        self.best_rally = max(self.best_rally, self.rally)
        self.last_point_player = to_player
        self._set_state("POINT_OVER", now)
        self.events.append(("point", to_player))

    # ---------------- rally ----------------

    def _update_rally(self, now, dt):
        shot = self.shot
        if shot is None:
            return
        if not shot.bounced and now - shot.t0 >= abs(shot.zb - shot.z0) / shot.speed:
            shot.bounced = True
            self.events.append(("out", None) if shot.out else ("bounce", shot.to_player))

        if shot.to_player:
            self._move_cpu(dt, 0.0)   # CPU drifts back to the middle
            if now > shot.t_arrive + TIMING_WINDOW + BALL_GONE_S:
                self._award_point(now, to_player=False)
                self.events.append(("missed", None))
        else:
            self._move_cpu(dt, self.cpu_target_x)
            if now >= shot.t_arrive and not shot.cpu_decided:
                shot.cpu_decided = True
                if shot.out:
                    return
                reach = abs(self.cpu_x - shot.x1) <= CPU_REACH
                if reach and random.random() > cpu_miss_chance(self.level):
                    self._cpu_hit(now, shot.x1, shot.y1, Z_CPU)
            elif shot.cpu_decided and now > shot.t_arrive + BALL_GONE_S:
                self._award_point(now, to_player=not shot.out)

    def _move_cpu(self, dt, target):
        step = cpu_move_speed(self.level) * dt
        self.cpu_x += clamp(target - self.cpu_x, -step, step)

    def _cpu_hit(self, now, x, y, z, serve=False):
        spread = cpu_spread(self.level)
        xb = random.uniform(-spread, spread) * TABLE_HALF_W
        zb = random.uniform(0.35, 0.80)
        speed = self.ball_speed * random.uniform(0.92, 1.08) * (0.85 if serve else 1.0)
        self.shot = make_shot(now, x, y, z, zb, xb, speed, to_player=True)
        self.events.append(("cpu_hit", None))

    def _player_hit(self, now, swing, quality, x, y):
        perfect = quality >= PERFECT_QUALITY
        # Paddle face angle (IMU) steers the ball left/right -- gently, see aim_landing_x.
        xb, out = aim_landing_x(swing.aim_deg)
        # Tilt: an open face lofts the ball a little; a closed face drives it a little flatter.
        tilt = clamp(swing.tilt_deg / TILT_FULL_DEG, -1, 1) * TILT_EFFECT
        speed = (self.ball_speed * (0.85 + 0.5 * swing.strength) * (1.15 if perfect else 1.0)
                 * (1 - 0.15 * tilt))
        arc = ARC_HEIGHT_OVER_NET * (1 + 0.6 * tilt)
        zb = TABLE_LEN - clamp(0.75 - 0.40 * swing.strength - 0.25 * tilt, 0.15, 1.0)   # harder = deeper
        self.shot = make_shot(now, x, y, Z_PLAYER, zb, xb, speed, to_player=False, arc=arc)
        self.shot.out = out
        # The CPU guesses where it's going, with a level-based error.
        err = random.gauss(0, max(0.03, 0.12 - 0.01 * self.level))
        self.cpu_target_x = self.shot.x1 + err
        self.rally += 1

    # ---------------- my swing ----------------

    def on_swing(self, swing):
        """Called once per swing detected by the IMU (or the space bar)."""
        now = swing.t
        if self.state == "SERVE" and self.server_is_player:
            x, _ = self._paddle_now()
            self._player_hit(now, swing, quality=0.5, x=x, y=0.25)
            self._set_state("RALLY", now)
            self.events.append(("hit", "SERVE"))
            return

        shot = self.shot
        if self.state != "RALLY" or shot is None or not shot.to_player or shot.swung:
            return

        dt = swing.t - shot.t_arrive
        if dt < -TIMING_WINDOW:
            if dt > -EARLY_HINT_WINDOW:
                self.events.append(("early", None))
            return
        if dt > TIMING_WINDOW:
            return

        shot.swung = True
        need = required_stroke(shot.x1)
        if need and swing.forehand is not None and swing.forehand != (need == "forehand"):
            self.events.append(("wrong_stroke", need))
            return
        dist, _ = self._paddle_distance(shot.x1, shot.y1)
        if dist > 1.0:
            self.events.append(("whiff", None))
            return

        timing_score = 1 - abs(dt) / TIMING_WINDOW
        center_score = 1 - dist
        quality = 0.6 * timing_score + 0.4 * center_score
        if quality >= PERFECT_QUALITY:
            label = "PERFECT"
        elif quality >= GOOD_QUALITY:
            label = "GOOD"
        else:
            label = "EARLY" if dt < 0 else "LATE"
        self._player_hit(swing.t, swing, quality, x=shot.x1, y=shot.y1)
        self.events.append(("hit", label))

    def _paddle_now(self):
        if self.paddle_history:
            _, x, y = self.paddle_history[-1]
            return clamp(x, -REACHABLE_X, REACHABLE_X), y
        return 0.0, 0.25

    def _paddle_distance(self, bx, by):
        """Smallest normalized paddle-to-ball distance over the recent history.

        Returns (distance, ball_x - paddle_x). distance <= 1 means in reach.
        """
        best = (math.inf, 0.0)
        for _, px, py in self.paddle_history:
            d = math.hypot((bx - px) / HIT_RADIUS_X, (by - py) / HIT_RADIUS_Y)
            if d < best[0]:
                best = (d, bx - px)
        return best

    def ball_position(self, now):
        if self.state == "SERVE":
            bob = 0.30 + 0.05 * math.sin(now * 6)   # ball tossed and waiting to be served
            if self.server_is_player:
                return self._paddle_now()[0], bob, Z_PLAYER + 0.15
            return self.cpu_x, bob, Z_CPU - 0.15
        if self.shot is None or self.state not in ("RALLY", "POINT_OVER"):
            return None
        if self.state == "POINT_OVER" and now - self.state_t > 0.5:
            return None
        return self.shot.position(now)


if __name__ == "__main__":
    # This file is only the game logic; running it starts the real game.
    from pingpong import main
    main()
