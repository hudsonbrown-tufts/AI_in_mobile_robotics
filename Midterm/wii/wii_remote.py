"""Use the LEGO Double Motor as a Wii Remote in the Dolphin emulator (Wii Sports).

    LEGO hub IMU ----------------------> Wii Remote accelerometer / gyro (real motion)
    motor shafts (turned by hand) -----> B (left shaft, the "trigger") and A (right shaft)
    phone camera + pose: right wrist --> Wii pointer
    pose: left-hand gestures ----------> A, HOME, D-pad left/right
    keyboard (this window) ------------> every button, as a backup

Everything is served to Dolphin as a DSU ("cemuhook") motion controller on
127.0.0.1:26760 -- enable it in Dolphin under Controllers > Alternate Input
Sources > DSU Client, then load the profile in profiles/ (see README.md).

Usage:
    python wii_remote.py                 # run the controller
    python wii_remote.py --recalibrate   # redo the two-pose axis calibration
    python wii_remote.py --map           # pulse one control at a time, to map it in Dolphin
    python wii_remote.py --no-camera     # motion + shaft buttons + keyboard only
    python wii_remote.py --no-motor      # pointer + gestures + keyboard only (motion = resting remote)
    python wii_remote.py --buzz          # buzz the hub after each swing (shakes the motion data a little)

Keys (with this window focused):
    A / Enter = A    B / Space = B    1, 2    = (+)    - (minus)    H = HOME    arrows = D-pad
    C = recalibrate axes   G = gestures on/off   Q / Esc = quit   (in --map: N / P = next / previous)
"""

import argparse
import math
import time

import pygame

import _paths  # noqa: F401
from dsu_server import PORT, DSUServer, PadState
from gestures import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, Gestures
from hub import STILL_S, TIP_MIN_ANGLE, Hub, angle_deg, wiimote_to_dsu

WIDTH, HEIGHT = 960, 600
NEUTRAL_ACCEL = (0.0, 0.0, 1.0)        # remote lying flat, in g (Wii Remote frame)
SWING_BUZZ_DPS = 500.0                 # --buzz: a swing peaks above this...
SWING_BUZZ_END_DPS = 200.0             # ...and buzzes once it slows below this

KEY_BUTTONS = {
    pygame.K_a: "A", pygame.K_RETURN: "A", pygame.K_b: "B", pygame.K_SPACE: "B",
    pygame.K_1: "1", pygame.K_2: "2", pygame.K_EQUALS: "+", pygame.K_PLUS: "+", pygame.K_KP_PLUS: "+",
    pygame.K_MINUS: "-", pygame.K_KP_MINUS: "-", pygame.K_h: "HOME",
    pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT",
}
BUTTON_ORDER = ["A", "B", "1", "2", "-", "HOME", "+", "UP", "DOWN", "LEFT", "RIGHT"]

# --map: (Dolphin field to click, what to pulse). Motion pulses are in Wii Remote axes
# (x left, y back, z up); gyro pulses are physical rotations (deg/s).
G, W = 3.0, 500.0
MAP_ITEMS = (
    [(f"Buttons > {b}", ("button", b)) for b in ["A", "B", "1", "2", "-", "+", "Home"]]
    + [(f"D-Pad > {d.title()}", ("button", d)) for d in ["UP", "DOWN", "LEFT", "RIGHT"]]
    + [("Point > Up", ("stick", (0, 1))), ("Point > Down", ("stick", (0, -1))),
       ("Point > Left", ("stick", (-1, 0))), ("Point > Right", ("stick", (1, 0)))]
    + [("Accelerometer > Up", ("accel", (0, 0, G))), ("Accelerometer > Down", ("accel", (0, 0, -G))),
       ("Accelerometer > Left", ("accel", (G, 0, 0))), ("Accelerometer > Right", ("accel", (-G, 0, 0))),
       ("Accelerometer > Forward", ("accel", (0, -G, 0))), ("Accelerometer > Backward", ("accel", (0, G, 0)))]
    + [("Gyroscope > Pitch Up", ("gyro", (-W, 0, 0))), ("Gyroscope > Pitch Down", ("gyro", (W, 0, 0))),
       ("Gyroscope > Roll Left", ("gyro", (0, W, 0))), ("Gyroscope > Roll Right", ("gyro", (0, -W, 0))),
       ("Gyroscope > Yaw Left", ("gyro", (0, 0, W))), ("Gyroscope > Yaw Right", ("gyro", (0, 0, -W)))]
)
MAP_ON_S, MAP_OFF_S = 0.7, 0.5

# Colors
BG, PANEL, TEXT, DIM = (20, 22, 28), (32, 35, 45), (235, 235, 235), (150, 155, 165)
GOOD, WARN, BAD, LIT = (90, 220, 120), (250, 190, 60), (240, 90, 80), (90, 170, 255)


class App:
    def __init__(self, args):
        self.args = args
        self.server = DSUServer(port=args.port).start()
        self.hub = None if args.no_motor else Hub().start()
        self.vision = None
        if not args.no_camera:
            from vision import DISPLAY_SIZE, Vision
            self.vision = Vision().start()
            self.gestures = Gestures(aspect=DISPLAY_SIZE[0] / DISPLAY_SIZE[1])
        else:
            self.gestures = Gestures()
        self.mode = "MAP" if args.map else "WAIT_HUB"
        self.mode_t = time.monotonic()
        self.map_index = 0
        self.flat_accel = None
        self.cal_message = ""
        self.swing_peak = 0.0
        self.sources = {}          # button -> where it came from (for the display)

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("LEGO Wii Remote")
        self.font_s = pygame.font.SysFont("arial", 15)
        self.font_m = pygame.font.SysFont("arial", 20, bold=True)
        self.font_l = pygame.font.SysFont("arial", 30, bold=True)

    def set_mode(self, mode):
        self.mode, self.mode_t = mode, time.monotonic()

    # ---------------- main loop ----------------

    def run(self):
        clock = pygame.time.Clock()
        running = True
        try:
            while running:
                clock.tick(60)
                now = time.monotonic()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        running = self.on_key(event.key)

                vis = self.vision.result if self.vision else None
                fresh = vis is not None and now - vis.t < 0.5
                self.gestures.update(vis.landmarks if fresh else None, now)
                self.step_calibration(now)
                self.server.set_state(self.map_state(now) if self.mode == "MAP" else self.play_state(now))
                self.draw(vis if fresh else None, now)
                pygame.display.flip()
        finally:
            self.server.close()
            if self.vision:
                self.vision.close()
            if self.hub:
                self.hub.close()
            pygame.quit()

    def on_key(self, key):
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        if key == pygame.K_c and self.hub:
            self.set_mode("CAL_FLAT")
        elif key == pygame.K_g:
            self.gestures.enabled = not self.gestures.enabled
        elif self.mode == "MAP" and key in (pygame.K_n, pygame.K_TAB):
            self.map_index = (self.map_index + 1) % len(MAP_ITEMS)
            self.mode_t = time.monotonic()
        elif self.mode == "MAP" and key == pygame.K_p:
            self.map_index = (self.map_index - 1) % len(MAP_ITEMS)
            self.mode_t = time.monotonic()
        return True

    # ---------------- axis calibration ----------------

    def step_calibration(self, now):
        hub = self.hub
        if self.mode == "WAIT_HUB":
            if hub is None:
                self.set_mode("PLAY")
            elif hub.connected:
                self.set_mode("CAL_FLAT" if hub.calibration is None or self.args.recalibrate else "PLAY")
        elif self.mode == "CAL_FLAT":
            if hub.held_still_since(now, self.mode_t) >= STILL_S:
                self.flat_accel = hub.still_average()
                hub.reset_gyro_integral()
                self.set_mode("CAL_UP")
        elif self.mode == "CAL_UP":
            avg = hub.still_average()
            if (hub.held_still_since(now, self.mode_t) >= STILL_S and avg
                    and angle_deg(avg, self.flat_accel) >= TIP_MIN_ANGLE):
                cal = hub.solve_calibration(self.flat_accel, avg, hub.gyro_integral)
                self.cal_message = (f"calibrated: tip {cal['tip_angle']:.0f} deg, "
                                    f"gyro {cal['gyro_scale']:g} deg/s per unit")
                print(f"[hub] {self.cal_message}")
                self.set_mode("PLAY")

    # ---------------- controller state ----------------

    def play_state(self, now):
        keys = pygame.key.get_pressed()
        self.sources = {}
        for k, name in KEY_BUTTONS.items():
            if keys[k]:
                self.sources[name] = "key"
        for name in self.gestures.buttons:
            self.sources.setdefault(name, "pose")
        if self.hub:
            for name in self.hub.buttons:
                self.sources.setdefault(name, "shaft")

        accel, gyro, motion_t = NEUTRAL_ACCEL, (0.0, 0.0, 0.0), now
        hub = self.hub
        if hub and hub.connected and hub.calibration and self.mode == "PLAY":
            accel, gyro, motion_t = hub.accel, hub.gyro, hub.motion_t
            if self.args.buzz:
                self.buzz_after_swing(math.sqrt(sum(c * c for c in gyro)))
        dsu_accel, dsu_gyro = wiimote_to_dsu(accel, gyro)
        return PadState(buttons=set(self.sources), right_stick=self.gestures.pointer or (0.0, 0.0),
                        accel=dsu_accel, gyro=dsu_gyro, motion_t=motion_t)

    def buzz_after_swing(self, dps):
        self.swing_peak = max(self.swing_peak, dps)
        if self.swing_peak > SWING_BUZZ_DPS and dps < SWING_BUZZ_END_DPS:
            self.hub.haptic("hit")
            self.swing_peak = 0.0
        elif dps < SWING_BUZZ_END_DPS:
            self.swing_peak = 0.0

    def map_state(self, now):
        """Pulse only the selected control (nothing else moves, not even gravity)."""
        _, (kind, value) = MAP_ITEMS[self.map_index]
        phase = (now - self.mode_t) % (MAP_ON_S + MAP_OFF_S)
        on = phase < MAP_ON_S
        st = PadState(motion_t=now)
        if on and kind == "button":
            st.buttons = {"HOME" if value == "Home" else value}
        elif on and kind == "stick":
            st.right_stick = value
        elif on and kind == "accel":
            st.accel, _ = wiimote_to_dsu(value, (0, 0, 0))
        elif on and kind == "gyro":
            _, st.gyro = wiimote_to_dsu((0, 0, 0), value)
        self.sources = {"pulse": on}
        return st

    # ---------------- drawing ----------------

    def text(self, s, font, color, pos, center=False):
        img = font.render(s, True, color)
        r = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        self.screen.blit(img, r)
        return r

    def draw(self, vis, now):
        scr = self.screen
        scr.fill(BG)
        self.draw_camera(vis)
        self.draw_pointer_screen()
        self.draw_buttons()
        self.draw_motion()
        self.draw_status()
        self.draw_prompt(now)
        self.text("Keys: A/Enter=A  B/Space=B  1 2  = +  - -  H=Home  arrows=D-pad  |  "
                  "C recalibrate  G gestures  Q quit", self.font_s, DIM, (14, HEIGHT - 24))

    def draw_camera(self, vis):
        rect = pygame.Rect(10, 10, 512, 288)
        pygame.draw.rect(self.screen, (8, 8, 10), rect)
        if vis is None or vis.frame_rgb is None:
            status = self.vision.status if self.vision else "off (--no-camera)"
            self.text(f"Camera: {status}", self.font_s, DIM, rect.center, center=True)
            return
        f = vis.frame_rgb
        self.screen.blit(pygame.image.frombuffer(f.tobytes(), (f.shape[1], f.shape[0]), "RGB"), rect.topleft)
        to_px = lambda uv: (rect.x + int(uv[0] * rect.w), rect.y + int(uv[1] * rect.h))
        lm = vis.landmarks or {}
        for a, b in [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, 13), (13, L_WRIST), (R_SHOULDER, 14), (14, R_WRIST),
                     (L_SHOULDER, 23), (R_SHOULDER, 24), (23, 24)]:
            if a in lm and b in lm:
                pygame.draw.line(self.screen, (255, 230, 80), to_px(lm[a]), to_px(lm[b]), 3)
        if R_WRIST in lm:
            pygame.draw.circle(self.screen, LIT, to_px(lm[R_WRIST]), 12, 3)
        if L_WRIST in lm:
            pygame.draw.circle(self.screen, WARN if self.gestures.buttons else DIM, to_px(lm[L_WRIST]), 10, 3)

    def draw_pointer_screen(self):
        rect = pygame.Rect(540, 10, 410, 231)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        self.text("Wii pointer (right wrist)", self.font_s, DIM, (rect.x + 8, rect.y + 6))
        p = self.gestures.pointer
        if p is None:
            self.text("no right wrist in view", self.font_s, DIM, rect.center, center=True)
            return
        x = rect.centerx + p[0] * (rect.w / 2 - 12)
        y = rect.centery - p[1] * (rect.h / 2 - 12)
        pygame.draw.circle(self.screen, LIT, (int(x), int(y)), 10)
        pygame.draw.circle(self.screen, TEXT, (int(x), int(y)), 10, 2)

    def draw_buttons(self):
        x0, y = 540, 252
        self.text("Buttons", self.font_m, TEXT, (x0, y))
        y += 28
        for i, name in enumerate(BUTTON_ORDER):
            r = pygame.Rect(x0 + (i % 6) * 68, y + (i // 6) * 40, 62, 32)
            src = self.sources.get(name) if self.mode != "MAP" else None
            pygame.draw.rect(self.screen, LIT if src else PANEL, r, border_radius=6)
            label = {"UP": "Up", "DOWN": "Down", "LEFT": "Left", "RIGHT": "Right"}.get(name, name)
            self.text(label, self.font_s, BG if src else TEXT, (r.centerx, r.centery - (6 if src else 0)), center=True)
            if src:
                self.text(src, pygame.font.SysFont("arial", 11), BG, (r.centerx, r.centery + 9), center=True)

    def draw_motion(self):
        x0, y = 540, 372
        hub = self.hub
        cal = hub.calibration if hub else None
        accel = hub.accel if hub and cal else NEUTRAL_ACCEL
        gyro = hub.gyro if hub and cal else (0, 0, 0)
        self.text("Motion (Wii Remote axes)", self.font_m, TEXT, (x0, y))
        y += 28
        for label, vals, full, unit in (("accel", accel, 3.0, "g"), ("gyro", gyro, 1000.0, "dps")):
            for axis, v in zip(("left", "back", "up"), vals):
                bar = pygame.Rect(x0 + 110, y + 3, 250, 12)
                pygame.draw.rect(self.screen, PANEL, bar, border_radius=4)
                w = int(min(1.0, abs(v) / full) * bar.w / 2)
                fill = pygame.Rect(bar.centerx - (w if v < 0 else 0), bar.y, w, bar.h)
                pygame.draw.rect(self.screen, GOOD if label == "accel" else LIT, fill, border_radius=4)
                pygame.draw.line(self.screen, DIM, (bar.centerx, bar.y - 2), (bar.centerx, bar.bottom + 2))
                self.text(f"{label} {axis}", self.font_s, DIM, (x0, y))
                self.text(f"{v:+6.2f} {unit}" if unit == "g" else f"{v:+5.0f}", self.font_s, TEXT, (x0 + 365, y))
                y += 18

    def draw_status(self):
        x, y = 14, 310
        def line(label, text, ok):
            nonlocal y
            pygame.draw.circle(self.screen, GOOD if ok else BAD, (x + 6, y + 9), 6)
            self.text(f"{label}: {text}", self.font_s, TEXT, (x + 20, y))
            y += 24
        clients = self.server.active_clients
        line("Dolphin (DSU)", f"connected ({len(clients)})" if clients
             else f"waiting on 127.0.0.1:{self.server.port}", bool(clients))
        if self.hub:
            ok = self.hub.connected
            line("LEGO hub", f"connected  {self.hub.sample_rate:.0f} Hz" if ok else self.hub.status, ok)
            cal = self.hub.calibration
            line("Axes", self.cal_message or ("calibrated (C to redo)" if cal else "not calibrated"), bool(cal))
            offs = self.hub.shaft_offsets
            shafts = "  ".join(f"{n} {offs.get(i, float('nan')):+.0f} deg"
                               for i, n in sorted(((0, "B"), (1, "A"))))
            line("Shafts", shafts, ok)
        else:
            line("LEGO hub", "off (--no-motor)", False)
        line("Pose", ("pointer + gestures" if self.gestures.enabled else "pointer only (G = gestures on)")
             if self.gestures.pointer else "no body in view", self.gestures.pointer is not None)

    def draw_prompt(self, now):
        msg = None
        if self.mode == "WAIT_HUB":
            msg = ("Connecting to the LEGO hub...", "turn it on; it pairs with the purple card")
        elif self.mode == "CAL_FLAT":
            held = self.hub.held_still_since(now, self.mode_t)
            msg = ("Axis setup 1/2: hold it like a Wii Remote",
                   f"aimed at the screen, top facing UP, still  ({min(held, STILL_S):.1f}/{STILL_S:.0f} s)")
        elif self.mode == "CAL_UP":
            held = self.hub.held_still_since(now, self.mode_t)
            msg = ("Axis setup 2/2: point it straight UP at the ceiling",
                   f"tip the front up slowly, then hold still  ({min(held, STILL_S):.1f}/{STILL_S:.0f} s)")
        elif self.mode == "MAP":
            name, _ = MAP_ITEMS[self.map_index]
            msg = (f"Map {self.map_index + 1}/{len(MAP_ITEMS)}:  {name}",
                   "in Dolphin, click that field now - it's pulsing.  N = next, P = previous")
        if msg:
            r = pygame.Rect(10, 470, 520, 96)
            pygame.draw.rect(self.screen, (60, 50, 20), r, border_radius=10)
            self.text(msg[0], self.font_m, WARN, (r.x + 12, r.y + 14))
            self.text(msg[1], self.font_s, TEXT, (r.x + 12, r.y + 50))
            if self.mode == "MAP":
                lit = self.sources.get("pulse")
                pygame.draw.circle(self.screen, GOOD if lit else PANEL, (r.right - 26, r.y + 26), 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", action="store_true", help="pulse one control at a time for Dolphin mapping")
    parser.add_argument("--recalibrate", action="store_true", help="redo the two-pose axis calibration")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-motor", action="store_true")
    parser.add_argument("--buzz", action="store_true", help="buzz the hub after each swing")
    parser.add_argument("--port", type=int, default=PORT, help="DSU server port (Dolphin default 26760)")
    App(parser.parse_args()).run()


if __name__ == "__main__":
    main()
