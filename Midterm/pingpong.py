"""Virtual ping pong, Wii Sports style.

  * Hold the LEGO Double Motor by itself in your right hand like a paddle.
    Its IMU (gyroscope) detects WHEN you swing and how hard.
  * The phone camera + MediaPipe Pose track your right wrist, which is
    WHERE the paddle is over your end of the table.
  * The IMU's yaw/pitch/roll turn and tilt the paddle: where the face
    points is where your shot goes. Balls on your right need a FOREHAND,
    balls on your left a BACKHAND (told apart by the gyro).
  * Show your AprilTag player card to start a game. The tag ID is your
    player profile: your level (saved in players.json) sets the ball speed.
    Win a game -> level up, lose -> level down.
  * The Double Motor buzzes in your hand when you hit the ball.
  * Your score -- your record number of continuous hits, as a float like
    "7.0" -- is published live to MQTT topic ME193/Rogers/Hudson on
    test.mosquitto.org (same broker as the other assignments).

Usage:
    python pingpong.py                 # full game: phone camera + Double Motor
    python pingpong.py --keyboard      # test without hardware (mouse = paddle, space = swing,
                                       #   F/B = forehand/backhand swing, T = tag)
    python pingpong.py --points 5      # shorter games
    python pingpong.py --list-cameras

Keys: Q/Esc quit, C calibrate forehand/backhand, R reset to the start screen,
      K toggle keyboard test mode, M toggle camera mirroring, Z re-zero paddle angle,
      space = swing (either stroke), F/B = forehand/backhand swing.
"""

import argparse
import json
import time
from pathlib import Path

import pygame

from game import MAX_LEVEL, POINTS_TO_WIN, Game, Swing
from paddle_imu import HOLD_STILL_S, HOLD_STILL_TIMEOUT_S
from render import HEIGHT, WIDTH, Renderer, Sounds, mouse_to_paddle

PLAYERS_PATH = Path(__file__).with_name("players.json")
GAME_OVER_SCREEN_S = 4.0
STALE_VISION_S = 0.5       # ignore camera results older than this
KEYBOARD_SWING_STRENGTH = 0.6
KEYBOARD_TAG_ID = 0

KEYBOARD_SWINGS = {pygame.K_SPACE: None, pygame.K_f: True, pygame.K_b: False}

HIT_HAPTIC = {"PERFECT": "perfect", "GOOD": "good", "EARLY": "hit", "LATE": "hit", "SERVE": "soft"}


class PlayerStore:
    """Level and stats per AprilTag ID, saved to players.json."""

    def __init__(self, path=PLAYERS_PATH):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def get(self, tag_id):
        key = str(tag_id)
        if key not in self.data:
            self.data[key] = {"level": 1, "wins": 0, "losses": 0, "best_rally": 0}
            self.save()
        return {"tag_id": tag_id, **self.data[key]}

    def record_game(self, player, won, best_rally):
        player["prev_level"] = player["level"]
        player["level"] = min(MAX_LEVEL, player["level"] + 1) if won else max(1, player["level"] - 1)
        player["wins" if won else "losses"] += 1
        player["best_rally"] = max(player["best_rally"], best_rally)
        self.data[str(player["tag_id"])] = {k: player[k] for k in ("level", "wins", "losses", "best_rally")}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyboard", action="store_true", help="mouse/keyboard test mode, no hardware")
    parser.add_argument("--no-motor", action="store_true", help="don't connect the Double Motor")
    parser.add_argument("--no-camera", action="store_true", help="don't open the camera")
    parser.add_argument("--no-mqtt", action="store_true", help="don't publish the score to MQTT")
    parser.add_argument("--points", type=int, default=POINTS_TO_WIN, help="points to win a game")
    parser.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    args = parser.parse_args()

    if args.list_cameras:
        from vision import list_cameras
        list_cameras()
        return

    keyboard_mode = args.keyboard
    vision = paddle = None
    if not (args.no_camera or args.keyboard):
        from vision import Vision
        vision = Vision().start()
    if not (args.no_motor or args.keyboard):
        from paddle_imu import Paddle
        paddle = Paddle().start()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Ping Pong")
    clock = pygame.time.Clock()
    renderer = Renderer(screen)
    scores = None
    if not args.no_mqtt:
        from score_mqtt import ScorePublisher
        scores = ScorePublisher().start()
    sounds = Sounds()
    store = PlayerStore()
    game = Game(points_to_win=args.points)

    player = None
    tag_was_visible = False
    simulated_tag = False
    running = True

    def haptic(name):
        if paddle:
            paddle.haptic(name)

    try:
        while running:
            dt = clock.tick(60) / 1000
            now = time.monotonic()

            # ---------- input ----------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_c and paddle:
                        paddle.calibrate()
                    elif event.key == pygame.K_r:
                        game.reset()
                        if paddle:
                            paddle.cancel_calibration()
                    elif event.key == pygame.K_k:
                        keyboard_mode = not keyboard_mode
                    elif event.key == pygame.K_m and vision:
                        vision.camera_mirrored = not vision.camera_mirrored
                    elif event.key == pygame.K_t:
                        simulated_tag = True
                    elif event.key == pygame.K_z and paddle:
                        paddle.zero_orientation()
                    elif event.key in KEYBOARD_SWINGS:
                        game.on_swing(Swing(t=now, strength=KEYBOARD_SWING_STRENGTH,
                                            forehand=KEYBOARD_SWINGS[event.key]))
                elif event.type == pygame.KEYUP and event.key == pygame.K_t:
                    simulated_tag = False

            vis = vision.result if vision else None
            fresh = vis is not None and now - vis.t < STALE_VISION_S

            # ---------- AprilTag: start a game the moment the card appears ----------
            tag_id = vis.tag_id if fresh else None
            if tag_id is None and simulated_tag:
                tag_id = KEYBOARD_TAG_ID
            tag_visible = tag_id is not None
            if game.state == "WAITING" and tag_visible and not tag_was_visible:
                player = store.get(tag_id)
                game.start(now, player["level"])
                print(f"Player tag {tag_id}: level {player['level']}, ball speed {game.ball_speed:.1f} m/s")
            tag_was_visible = tag_visible

            # ---------- per-game setup: zero the paddle angles, then calibrate strokes ----------
            paddle_ready = paddle is not None and paddle.connected and not keyboard_mode
            if game.state == "SETUP_FLAT":
                if not paddle_ready:
                    game.begin_countdown(now)
                elif (paddle.held_still_since(now, game.state_t) >= HOLD_STILL_S
                      or now - game.state_t > HOLD_STILL_TIMEOUT_S):
                    paddle.zero_orientation()      # held upright, facing the screen = neutral
                    paddle.calibrate()             # 3 forehands, then 3 backhands
                    haptic("soft")
                    sounds.play("bounce")
                    game.begin_strokes(now)
            elif game.state == "SETUP_STROKES" and (not paddle_ready or not paddle.calibrating):
                game.begin_countdown(now)

            if game.state == "GAME_OVER" and now - game.state_t > GAME_OVER_SCREEN_S:
                game.reset()

            # ---------- paddle position and swings ----------
            if keyboard_mode:
                paddle_xy = mouse_to_paddle(*pygame.mouse.get_pos())
            else:
                paddle_xy = vis.paddle if fresh else None
            if paddle:
                for swing in paddle.get_swings():
                    game.on_swing(swing)

            game.update(now, dt, paddle_xy)

            # ---------- game events -> haptics, sounds, pop-ups ----------
            for name, data in game.events:
                if name == "game_start":
                    haptic("start")
                    sounds.play("start")
                elif name == "serve_start":
                    if paddle:
                        paddle.zero_orientation(yaw_only=True)   # cancel gyro yaw drift
                elif name == "hit":
                    renderer.popup(data)
                    haptic(HIT_HAPTIC.get(data, "hit"))
                    sounds.play("hit")
                elif name == "cpu_hit":
                    sounds.play("cpu_hit")
                elif name == "bounce":
                    sounds.play("bounce")
                elif name == "early":
                    renderer.popup("Too early!")
                elif name == "whiff":
                    renderer.popup("Miss!")
                elif name == "wrong_stroke":
                    renderer.popup("Backhand!" if data == "backhand" else "Forehand!")
                    sounds.play("miss")
                elif name == "out":
                    renderer.popup("Out!")
                elif name == "point":
                    renderer.popup("Point!" if data else "CPU point")
                    sounds.play("point" if data else "miss")
                    if not data:
                        haptic("miss")
                elif name == "game_over":
                    store.record_game(player, won=data, best_rally=game.best_rally)
                    haptic("win" if data else "lose")
                    print(f"Game over: {'won' if data else 'lost'} {game.score_player}-{game.score_cpu}. "
                          f"Level {player['prev_level']} -> {player['level']}")
            game.events.clear()

            # ---------- live score to MQTT: record number of continuous hits ----------
            if scores:
                record = None
                if player:
                    record = max(player["best_rally"], game.best_rally, game.rally)
                scores.update(record, now)
                renderer.mqtt_status = (f"{scores.status}  sent {scores.last_sent}" if scores.last_sent
                                        else scores.status)

            # Angles measured from the stroke this side of my body needs (forehand on my right).
            side_forehand = None if paddle_xy is None else paddle_xy[0] >= 0
            orientation = paddle.orientation(side_forehand) if paddle and not keyboard_mode else None
            renderer.draw(game, now, vis, vision.status if vision else "off", orientation,
                          paddle, player, paddle_xy, keyboard_mode)
            pygame.display.flip()
    finally:
        if vision:
            vision.close()
        if paddle:
            paddle.close()
        if scores:
            scores.close()
        pygame.quit()


if __name__ == "__main__":
    main()
