"""Drawing for the ping pong GUI (pygame, 1280x720).

    +------------------+-----------------------------------+
    | camera + pose    |                                   |
    | (512x288)        |      3D table, seen from behind   |
    +------------------+      my end, Wii-style (768x720)  |
    | player card,     |                                   |
    | swing meter,     |                                   |
    | status, keys     |                                   |
    +------------------+-----------------------------------+

The table is drawn with a tiny pinhole camera (project()) in the same
meters as game.py, so the ball, its shadow and both paddles line up.
"""

import math
import time
from collections import deque

import numpy as np
import pygame

import game as g
from paddle_imu import CALIBRATION_SWINGS, HOLD_STILL_S, SETUP_READ_S, SWING_FULL, SWING_MIN_PEAK
from vision import DISPLAY_SIZE, PADDLE_WRIST, SKELETON

WIDTH, HEIGHT = 1280, 720
LEFT_W = DISPLAY_SIZE[0]
CAM_H = DISPLAY_SIZE[1]
TABLE_X0 = LEFT_W
TABLE_W = WIDTH - LEFT_W

# --- Virtual camera behind my end of the table ---
VCAM_Y = 1.0        # m above the table
VCAM_Z = -1.9       # m behind my end
VCAM_PITCH = math.radians(20)
FOCAL = 880
CX, CY = TABLE_W / 2, 330

PADDLE_R = 0.085    # m (drawn size)
BALL_DRAW_SCALE = 1.8

# --- Colors ---
BG_TOP = (28, 38, 66)
BG_FLOOR = (120, 86, 60)
TABLE_TOP = (30, 92, 170)
TABLE_SIDE = (18, 52, 100)
LINE = (240, 240, 240)
NET = (235, 235, 235)
BALL = (255, 250, 235)
SHADOW = (14, 50, 100)
RED_RUBBER = (210, 40, 40)          # forehand side
BLACK_RUBBER = (35, 35, 40)         # backhand side
BLUE_RUBBER = (40, 70, 200)
WOOD = (170, 120, 70)
PANEL = (22, 24, 30)
TEXT = (235, 235, 235)
DIM = (150, 155, 165)
GOOD = (90, 220, 120)
WARN = (250, 190, 60)
BAD = (240, 90, 80)

POPUP_COLORS = {"PERFECT": (255, 215, 60), "GOOD": GOOD, "EARLY": WARN, "LATE": WARN,
                "SERVE": TEXT, "Too early!": WARN, "Miss!": BAD, "Point!": GOOD,
                "CPU point": BAD, "Forehand!": BAD, "Backhand!": BAD, "Out!": BAD}
SHOW_AIM_GUIDE = True   # faint line on the table showing where the paddle face points


def project(x, y, z):
    """World meters -> (screen x, screen y, pixels per meter) in the table panel."""
    Y = y - VCAM_Y
    Z = z - VCAM_Z
    yc = Y * math.cos(VCAM_PITCH) + Z * math.sin(VCAM_PITCH)
    zc = -Y * math.sin(VCAM_PITCH) + Z * math.cos(VCAM_PITCH)
    zc = max(zc, 0.05)
    s = FOCAL / zc
    return TABLE_X0 + CX + x * s, CY - yc * s, s


def pt(x, y, z):
    sx, sy, _ = project(x, y, z)
    return int(sx), int(sy)


def mouse_to_paddle(mx, my):
    """Keyboard test mode: mouse over the table panel -> paddle (x, y) meters."""
    x = (mx - TABLE_X0 - TABLE_W / 2) / (TABLE_W / 2) * 0.95
    y = (HEIGHT - my) / HEIGHT * 0.65
    return g.clamp(x, -1.0, 1.0), g.clamp(y, -0.1, 0.6)


class Sounds:
    """Tiny synthesized sound effects (no audio files needed)."""

    def __init__(self):
        self.ok = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
            self.rate, _, self.channels = pygame.mixer.get_init()
            self.hit = self._tone(1500, 0.045, noise=0.5)
            self.cpu_hit = self._tone(1200, 0.045, noise=0.5)
            self.bounce = self._tone(800, 0.03, noise=0.3)
            self.point = self._tone(660, 0.12, then=990)
            self.miss = self._tone(300, 0.25, then=200)
            self.start = self._tone(523, 0.1, then=784)
            self.ok = True
        except Exception as exc:
            print(f"[sound] disabled: {exc}")

    def _tone(self, freq, dur, noise=0.0, then=None):
        rate = self.rate
        t = np.arange(int(rate * dur)) / rate
        if then:
            half = len(t) // 2
            f = np.where(np.arange(len(t)) < half, freq, then)
            wave = np.sin(2 * np.pi * f * t)
        else:
            wave = np.sin(2 * np.pi * freq * t)
        wave = wave * (1 - noise) + noise * np.random.uniform(-1, 1, len(t))
        wave *= np.exp(-t / (dur / 3))
        samples = (wave * 12000).astype(np.int16)
        if self.channels > 1:
            samples = np.repeat(samples[:, None], self.channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(samples))

    def play(self, name):
        if self.ok and hasattr(self, name):
            getattr(self, name).play()


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font_s = pygame.font.SysFont("arial", 16)
        self.font_m = pygame.font.SysFont("arial", 22, bold=True)
        self.font_l = pygame.font.SysFont("arial", 40, bold=True)
        self.font_xl = pygame.font.SysFont("arial", 110, bold=True)
        self.popups = []                 # [text, t_start]
        self.trail = deque(maxlen=8)
        self.background = self._make_background()
        self.mqtt_status = None          # set by pingpong.py when publishing the score

    def popup(self, text):
        self.popups.append([text, time.monotonic()])

    # ---------------- main entry ----------------

    def draw(self, game, now, vis, vision_status, orientation, paddle, player, paddle_xy, keyboard_mode):
        self.screen.fill(PANEL)
        self._draw_table_view(game, now, paddle_xy, orientation)
        self._draw_hud(game, now)
        self._draw_overlay(game, player, paddle, now)
        self._draw_popups(now)
        self._draw_camera(vis, vision_status)
        self._draw_info(game, vis, vision_status, orientation, paddle, player, keyboard_mode)

    # ---------------- table view ----------------

    def _make_background(self):
        surf = pygame.Surface((TABLE_W, HEIGHT))
        horizon = pt(0, 0, g.TABLE_LEN + 4)[1]
        for y in range(HEIGHT):
            if y < horizon:
                k = y / max(1, horizon)
                c = [int(BG_TOP[i] * (0.7 + 0.3 * k)) for i in range(3)]
            else:
                k = (y - horizon) / max(1, HEIGHT - horizon)
                c = [int(BG_FLOOR[i] * (0.55 + 0.45 * k)) for i in range(3)]
            pygame.draw.line(surf, c, (0, y), (TABLE_W, y))
        return surf

    def _draw_table_view(self, game, now, paddle_xy, orientation):
        scr = self.screen
        scr.blit(self.background, (TABLE_X0, 0))
        clip = scr.get_clip()
        scr.set_clip(pygame.Rect(TABLE_X0, 0, TABLE_W, HEIGHT))

        hw, L = g.TABLE_HALF_W, g.TABLE_LEN
        # Legs and side face, then the top with its white lines.
        for lx in (-hw + 0.1, hw - 0.1):
            for lz in (0.2, L - 0.2):
                pygame.draw.line(scr, (40, 40, 45), pt(lx, -0.03, lz), pt(lx, -0.76, lz), 6)
        pygame.draw.polygon(scr, TABLE_SIDE, [pt(-hw, 0, 0), pt(hw, 0, 0), pt(hw, -0.04, 0), pt(-hw, -0.04, 0)])
        pygame.draw.polygon(scr, TABLE_TOP, [pt(-hw, 0, 0), pt(hw, 0, 0), pt(hw, 0, L), pt(-hw, 0, L)])
        pygame.draw.polygon(scr, LINE, [pt(-hw, 0, 0), pt(hw, 0, 0), pt(hw, 0, L), pt(-hw, 0, L)], 3)
        pygame.draw.line(scr, LINE, pt(0, 0, 0), pt(0, 0, L), 2)

        ball = game.ball_position(now)
        if ball is None:
            self.trail.clear()
        else:
            self.trail.append(ball)

        self._draw_paddle(game.cpu_x, 0.25, g.Z_CPU, BLUE_RUBBER, alpha=255)
        if ball and ball[2] > g.NET_Z:
            self._draw_ball(ball)
        self._draw_net()
        if ball and ball[2] <= g.NET_Z:
            self._draw_ball(ball)
        if paddle_xy is not None:
            o = orientation or {"aim": 0.0, "tilt": 0.0, "twist": 0.0}
            if SHOW_AIM_GUIDE and orientation and game.state in ("SERVE", "RALLY"):
                self._draw_aim_guide(paddle_xy[0], o["aim"])
            # Red forehand rubber on my right side, black backhand rubber on my left.
            rubber = RED_RUBBER if paddle_xy[0] >= 0 else BLACK_RUBBER
            self._draw_paddle(paddle_xy[0], paddle_xy[1], g.Z_PLAYER, rubber, alpha=170,
                              aim=o["aim"], tilt=o["tilt"], twist=o["twist"])

        scr.set_clip(clip)

    def _draw_net(self):
        hw = g.TABLE_HALF_W + 0.08
        net = pygame.Surface((TABLE_W, HEIGHT), pygame.SRCALPHA)
        poly = [pt(-hw, 0, g.NET_Z), pt(hw, 0, g.NET_Z), pt(hw, g.NET_H, g.NET_Z), pt(-hw, g.NET_H, g.NET_Z)]
        poly = [(x - TABLE_X0, y) for x, y in poly]
        pygame.draw.polygon(net, (230, 230, 230, 70), poly)
        self.screen.blit(net, (TABLE_X0, 0))
        pygame.draw.line(self.screen, NET, pt(-hw, g.NET_H, g.NET_Z), pt(hw, g.NET_H, g.NET_Z), 4)
        for x in (-hw, hw):
            pygame.draw.line(self.screen, (60, 60, 60), pt(x, -0.02, g.NET_Z), pt(x, g.NET_H, g.NET_Z), 4)

    def _draw_ball(self, ball):
        x, y, z = ball
        # Shadow on the table straight below the ball.
        if abs(x) <= g.TABLE_HALF_W and 0 <= z <= g.TABLE_LEN and y >= 0:
            sx, sy, s = project(x, 0, z)
            r = max(2, g.BALL_R * BALL_DRAW_SCALE * s * max(0.4, 1 - y))
            pygame.draw.ellipse(self.screen, SHADOW, (sx - r * 1.4, sy - r * 0.5, r * 2.8, r))
        # Motion trail.
        for i, (tx, ty, tz) in enumerate(list(self.trail)[:-1]):
            sx, sy, s = project(tx, ty, tz)
            r = max(1, g.BALL_R * BALL_DRAW_SCALE * s * (i + 1) / len(self.trail) * 0.7)
            pygame.draw.circle(self.screen, (200, 200, 190), (int(sx), int(sy)), int(r))
        sx, sy, s = project(x, y, z)
        r = max(3, int(g.BALL_R * BALL_DRAW_SCALE * s))
        pygame.draw.circle(self.screen, BALL, (int(sx), int(sy)), r)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(sx - r * 0.3), int(sy - r * 0.3)), max(1, r // 3))

    def _draw_paddle(self, x, y, z, rubber, alpha, aim=0.0, tilt=0.0, twist=0.0):
        """Paddle blade centered at (x, y, z). Turning the face (aim) narrows it, tilting
        it (tilt) flattens it, and twist rotates the whole paddle on screen."""
        sx, sy, s = project(x, y, z)
        r = PADDLE_R * s
        rx = max(r * 0.2, r * abs(math.cos(math.radians(aim))))
        ry = max(r * 0.25, r * abs(math.cos(math.radians(tilt))))
        w, h = int(r * 2 + 6), int(r * 4 + 6)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        c = (w // 2, h // 2)
        pygame.draw.rect(surf, (*WOOD, alpha), (c[0] - r * 0.22, c[1] + ry * 0.85, r * 0.44, r * 1.0),
                         border_radius=int(r * 0.15))
        pygame.draw.ellipse(surf, (40, 30, 20, alpha), (c[0] - rx, c[1] - ry, rx * 2, ry * 2))
        pygame.draw.ellipse(surf, (*rubber, alpha), (c[0] - rx * 0.92, c[1] - ry * 0.92, rx * 1.84, ry * 1.84))
        if twist:
            surf = pygame.transform.rotate(surf, -twist)
        self.screen.blit(surf, surf.get_rect(center=(int(sx), int(sy))))

    def _draw_aim_guide(self, x, aim):
        """Dotted line from my paddle to where a shot with this face angle would land."""
        target, out = g.aim_landing_x(aim)
        color = BAD if out else (255, 255, 255)
        z_land = g.TABLE_LEN - 0.55
        for i in range(1, 12):
            k = i / 12
            p = pt(x + (target - x) * k, 0.0, z_land * k)
            pygame.draw.circle(self.screen, color, p, 2)
        pygame.draw.circle(self.screen, color, pt(target, 0.0, z_land), 6, 2)

    # ---------------- HUD / overlays ----------------

    def _text(self, text, font, color, center=None, topleft=None, shadow=True):
        img = font.render(text, True, color)
        rect = img.get_rect()
        if center:
            rect.center = center
        else:
            rect.topleft = topleft
        if shadow:
            self.screen.blit(font.render(text, True, (0, 0, 0)), rect.move(2, 2))
        self.screen.blit(img, rect)
        return rect

    def _draw_hud(self, game, now):
        if game.state == "WAITING":
            return
        cx = TABLE_X0 + TABLE_W // 2
        box = pygame.Rect(0, 0, 300, 56)
        box.center = (cx, 40)
        pygame.draw.rect(self.screen, (10, 12, 20), box, border_radius=12)
        pygame.draw.rect(self.screen, (90, 95, 110), box, 2, border_radius=12)
        self._text(f"YOU  {game.score_player} : {game.score_cpu}  CPU", self.font_m, TEXT, center=box.center)
        if game.state == "SERVE":
            who = "Your serve - swing!" if game.server_is_player else "CPU serving..."
            self._text(who, self.font_m, WARN, center=(cx, 95))
        if game.rally >= 3 and game.state == "RALLY":
            self._text(f"Rally {game.rally}", self.font_m, TEXT, center=(cx, 95))
        # Stroke cue for the incoming ball: forehand on my right, backhand on my left.
        shot = game.shot
        if game.state == "RALLY" and shot and shot.to_player and not shot.swung:
            need = g.required_stroke(shot.x1)
            if need == "forehand":
                self._text("FOREHAND  >", self.font_m, (255, 140, 120), center=(TABLE_X0 + TABLE_W - 110, HEIGHT - 60))
            elif need == "backhand":
                self._text("<  BACKHAND", self.font_m, (190, 190, 200), center=(TABLE_X0 + 110, HEIGHT - 60))

    def _draw_overlay(self, game, player, paddle, now):
        cx, cy = TABLE_X0 + TABLE_W // 2, HEIGHT // 2
        if game.state in ("WAITING", "GAME_OVER", "SETUP_FLAT", "SETUP_STROKES"):
            dim = pygame.Surface((TABLE_W, HEIGHT), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 150))
            self.screen.blit(dim, (TABLE_X0, 0))
        if game.state == "WAITING":
            self._text("VIRTUAL PING PONG", self.font_l, TEXT, center=(cx, cy - 90))
            self._text("Hold up your AprilTag player card to start", self.font_m, WARN, center=(cx, cy - 20))
            self._text("Hold the Double Motor like a paddle in your right hand", self.font_s, DIM, center=(cx, cy + 25))
            self._text("Swing when the ball reaches you  -  your wrist aims the paddle", self.font_s, DIM, center=(cx, cy + 50))
        elif game.state == "SETUP_FLAT":
            self._text("Step 1: zero the paddle angle", self.font_l, TEXT, center=(cx, cy - 90))
            self._text("Hold the paddle UPRIGHT, face toward the screen, and keep it still", self.font_m, WARN,
                       center=(cx, cy - 30))
            held = paddle.held_still_since(now, game.state_t) if paddle else 0.0
            done = min(1.0, held / HOLD_STILL_S)
            bar = pygame.Rect(0, 0, 360, 22)
            bar.center = (cx, cy + 25)
            pygame.draw.rect(self.screen, (50, 52, 60), bar, border_radius=8)
            fill = bar.copy()
            fill.width = int(bar.width * done)
            pygame.draw.rect(self.screen, GOOD, fill, border_radius=8)
            if now - game.state_t < SETUP_READ_S:
                status = "get the paddle into position..."
            else:
                status = "still..." if done > 0 else "moving - hold it steady"
            self._text(status, self.font_s, DIM, center=(cx, cy + 60))
        elif game.state == "SETUP_STROKES":
            self._text("Step 2: calibrate your strokes", self.font_l, TEXT, center=(cx, cy - 110))
            prompt = paddle.calibration_prompt if paddle else None
            if prompt:
                stroke = "FOREHAND" if "FOREHAND" in prompt else "BACKHAND"
                count = prompt.split("(")[-1].rstrip(")")
                self._text(f"Swing a {stroke}", self.font_l, (255, 140, 120) if stroke == "FOREHAND" else (200, 200, 210),
                           center=(cx, cy - 40))
                self._text(f"{count}", self.font_m, WARN, center=(cx, cy + 10))
                hint = ("right side, swing across your body to the left" if stroke == "FOREHAND"
                        else "left side, swing across your body to the right")
                self._text(hint, self.font_s, DIM, center=(cx, cy + 45))
                self._text(f"{CALIBRATION_SWINGS} of each - the swing back doesn't count", self.font_s, DIM,
                           center=(cx, cy + 70))
        elif game.state == "COUNTDOWN":
            n = max(1, math.ceil(game.countdown_left))
            self._text(str(n), self.font_xl, TEXT, center=(cx, cy))
            self._text("Get ready!", self.font_m, TEXT, center=(cx, cy - 110))
            if player:
                self._text(f"Level {player['level']}", self.font_m, WARN, center=(cx, cy + 90))
        elif game.state == "GAME_OVER":
            won = game.winner_is_player
            self._text("YOU WIN!" if won else "CPU WINS", self.font_l, GOOD if won else BAD, center=(cx, cy - 70))
            self._text(f"{game.score_player} : {game.score_cpu}   (best rally {game.best_rally})",
                       self.font_m, TEXT, center=(cx, cy - 15))
            if player and "prev_level" in player:
                self._text(f"Level {player['prev_level']}  ->  {player['level']}", self.font_m, WARN,
                           center=(cx, cy + 30))
            self._text("Show your card again to play another game", self.font_s, DIM, center=(cx, cy + 80))

    def _draw_popups(self, now):
        cx = TABLE_X0 + TABLE_W // 2
        keep = []
        for text, t0 in self.popups:
            age = now - t0
            if age > 0.9:
                continue
            keep.append([text, t0])
            grow = min(1.0, age / 0.12)
            font = self.font_l if grow >= 1 else pygame.font.SysFont("arial", int(20 + 20 * grow), bold=True)
            img = font.render(text, True, POPUP_COLORS.get(text, TEXT))
            img.set_alpha(int(255 * min(1.0, (0.9 - age) / 0.3)))
            self.screen.blit(img, img.get_rect(center=(cx, 210 - age * 40)))
        self.popups = keep

    # ---------------- left panel ----------------

    def _draw_camera(self, vis, vision_status):
        rect = pygame.Rect(0, 0, LEFT_W, CAM_H)
        if vis is None or vis.frame_rgb is None:
            pygame.draw.rect(self.screen, (8, 8, 10), rect)
            self._text(f"Camera: {vision_status}", self.font_s, DIM, center=rect.center, shadow=False)
            return
        frame = vis.frame_rgb
        surf = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0))

        def to_px(uv):
            return int(uv[0] * LEFT_W), int(uv[1] * CAM_H)

        if vis.landmarks:
            for a, b in SKELETON:
                if a in vis.landmarks and b in vis.landmarks:
                    pygame.draw.line(self.screen, (255, 230, 80), to_px(vis.landmarks[a]), to_px(vis.landmarks[b]), 3)
        if vis.wrist_uv:
            p = to_px(vis.wrist_uv)
            pygame.draw.circle(self.screen, RED_RUBBER, p, 16)
            pygame.draw.circle(self.screen, (255, 255, 255), p, 16, 2)
        if vis.tag_corners:
            pts = [to_px(c) for c in vis.tag_corners]
            pygame.draw.polygon(self.screen, GOOD, pts, 3)
            self._text(f"Tag {vis.tag_id}", self.font_s, GOOD, topleft=(pts[0][0], pts[0][1] - 20))

    def _draw_info(self, game, vis, vision_status, orientation, paddle, player, keyboard_mode):
        x, y = 16, CAM_H + 12

        # Player card
        card = pygame.Rect(10, y, LEFT_W - 20, 120)
        pygame.draw.rect(self.screen, (34, 38, 50), card, border_radius=10)
        if player:
            self._text(f"Player  (Tag #{player['tag_id']})", self.font_m, TEXT, topleft=(x + 8, y + 8), shadow=False)
            self._text(f"Level {player['level']}", self.font_l, WARN, topleft=(x + 8, y + 40), shadow=False)
            self._text(f"Ball speed {g.level_ball_speed(player['level']):.1f} m/s", self.font_s, DIM,
                       topleft=(x + 200, y + 46), shadow=False)
            self._text(f"Record {player['wins']}W - {player['losses']}L   best rally {player['best_rally']}",
                       self.font_s, DIM, topleft=(x + 200, y + 70), shadow=False)
        else:
            self._text("No player yet", self.font_m, DIM, topleft=(x + 8, y + 8), shadow=False)
            self._text("Show your AprilTag card to the camera", self.font_s, DIM, topleft=(x + 8, y + 44), shadow=False)
        y += 134

        # Swing meter
        self._text("Swing", self.font_m, TEXT, topleft=(x, y), shadow=False)
        bar = pygame.Rect(x + 80, y + 4, LEFT_W - 120, 20)
        pygame.draw.rect(self.screen, (40, 42, 50), bar, border_radius=6)
        full = 2 * SWING_FULL
        omega = paddle.omega if paddle else 0.0
        fill = bar.copy()
        fill.width = int(bar.width * min(1.0, omega / full))
        pygame.draw.rect(self.screen, GOOD if omega > SWING_MIN_PEAK else (90, 120, 200), fill, border_radius=6)
        tick = bar.x + int(bar.width * SWING_MIN_PEAK / full)
        pygame.draw.line(self.screen, WARN, (tick, bar.y - 3), (tick, bar.bottom + 3), 2)
        y += 32
        last = "-"
        if paddle and paddle.calibrating:
            last = paddle.calibration_prompt
        elif paddle and paddle.last_swing:
            swing, peak = paddle.last_swing
            kind = "FOREHAND" if swing.forehand else "BACKHAND"
            last = f"last: {kind}  strength {swing.strength:.2f}  aim {swing.aim_deg:+.0f}  tilt {swing.tilt_deg:+.0f}"
        self._text(last, self.font_s, WARN if paddle and paddle.calibrating else DIM, topleft=(x, y), shadow=False)
        y += 34

        # Status
        def status_line(label, text, ok):
            nonlocal y
            pygame.draw.circle(self.screen, GOOD if ok else BAD, (x + 6, y + 9), 6)
            self._text(f"{label}: {text}", self.font_s, TEXT, topleft=(x + 20, y), shadow=False)
            y += 24

        cam_ok = vision_status == "running"
        status_line("Camera", f"{vision_status}  {vis.fps:.0f} fps" if vis and cam_ok else vision_status, cam_ok)
        tracking = bool(vis and vis.paddle is not None)
        wrist = "right" if PADDLE_WRIST == 16 else "left"
        status_line("Pose", f"tracking {wrist} wrist" if tracking else "no body / wrist in view", tracking)
        motor_ok = bool(paddle and paddle.connected)
        motor_txt = f"connected  {paddle.sample_rate:.0f} Hz IMU" if motor_ok else (paddle.status if paddle else "off")
        status_line("Motor", motor_txt[:60], motor_ok)
        if orientation:
            status_line("Paddle", f"aim {orientation['aim']:+4.0f}   tilt {orientation['tilt']:+4.0f}   "
                        f"twist {orientation['twist']:+4.0f} deg   (Z = re-zero)", True)
        if paddle and paddle.connected:
            cal = paddle.calibration is not None
            status_line("Strokes", "forehand/backhand calibrated" if cal
                        else "not calibrated - press C (using yaw-turn guess)", cal)
        if self.mqtt_status:
            status_line("MQTT", f"ME193/Rogers/Hudson  {self.mqtt_status}"[:62],
                        self.mqtt_status.startswith("connected"))
        if keyboard_mode:
            status_line("Mode", "KEYBOARD (mouse = paddle, space/F/B = swing, T = tag)", True)

        self._text("Q quit  C calibrate strokes  Z zero paddle  R reset  K keyboard  M mirror",
                   self.font_s, DIM, topleft=(x, HEIGHT - 26), shadow=False)
