"""All HW5 sumo tunables in one place.

Units: positions are in RING UNITS, where the ring center is (0, 0) and the
blue tape is radius 1.0. Angles are radians, counterclockwise positive, in the
ring frame (x = image right, y = image up). Motor speeds are LEGO percent
(-100..100).
"""

import math
from pathlib import Path

import legoeducation as le

HERE = Path(__file__).resolve().parent

# --- Files ---
ROBOT_TAGS_PATH = HERE / "robot_tags.json"            # written by make_tags.py
RING_CALIBRATION_PATH = HERE / "ring_calibration.json"
RING_BACKGROUND_PATH = HERE / "ring_background.npz"   # empty-ring background (opponent detection)
COLOR_CALIBRATION_PATH = HERE / "color_calibration.json"
DRIVE_CALIBRATION_PATH = HERE / "drive_calibration.json"
LOSE_SOUND_PATH = HERE / "lose.mp3"

# --- LEGO devices (both on the purple connection card) ---
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "5164"
NOTIFICATION_MS = 15            # fastest the devices allow (~38-66 Hz); default 100 ms is too slow
BLE_DISCONNECT_TIMEOUT_S = 3.0
MOTOR_MIN_CHANGE = 3            # send a new tank command only if a side changes by this much...
MOTOR_MIN_INTERVAL_S = 0.1      # ...and at most 10 per second. Faster floods the hub's Bluetooth
                                # (each command waits for a reply) until it disconnects. Stops,
                                # direction reversals and the edge reflex are always sent at once.
RECONNECT_EVERY_S = 3.0         # a dropped device is reconnected in the background this often

# --- MQTT ---
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/hudson"
SHAKE_PREFIX = "shake:"         # from the UNO Q: "shake:0.63" (or "shake:err")
SUMO_PREFIX = "sumo:"           # we publish "sumo:lost", "sumo:start"
MQTT_CONNECT_RETRIES = 3
MQTT_RETRY_DELAY_SECONDS = 3

# --- Sound ---
OUTPUT_DEVICE_NAME_HINT = "Speakers"   # same as HW3; None = system default

# --- Camera ---
CAMERA_NAME_HINT = "Windows Virtual Camera"   # the phone through Phone Link
CAMERA_INDEX = 0                              # fallback if no camera matches the hint
CAMERA_LATENCY_S = 0.12         # phone -> Phone Link -> OpenCV delay; measurements are back-dated by this
TAG_TIMEOUT_S = 0.6             # no tag for this long = camera tracking lost (BLIND)

# Ring calibration (c) looks for the most ring-shaped bluish thing in the frame, so it
# works for deep blue and light blue painter's tape alike. It tries every hue window
# x saturation step below (OpenCV hue 0-180: ~90 = sky blue, ~110 = blue, ~125 = deep blue)
# and keeps the best ring fit. The saturation step is measured above the floor's own
# saturation, so a dark gray floor drops out.
TAPE_HUE_WINDOWS = [(80, 135), (85, 115), (95, 130), (100, 125)]
TAPE_SAT_STEPS = [12, 25, 45, 75, 110]
TAPE_MIN_VALUE = 40
TAPE_MIN_SCORE = 0.35           # best ring fit (coverage x precision) must beat this

# Opponent detection (opponent.py): difference from the empty-ring background
OPP_SIZE_CM = 15.0              # roughly how wide the opponent is (blobs from 0.35x to 3x count)
OPP_MAX_HEIGHT_CM = 15.0        # tallest the opponent might be (sizes the search area)
OPP_Z = 4.0                     # a pixel differs from the background by this many noise levels = "something"
OPP_EDGE_Z = 3.0                # ...or its texture (edges) changed this much (dark robot on dark floor)
OPP_NOISE_L = 2.0               # minimum noise assumed per pixel: brightness (Lab L)
OPP_NOISE_C = 2.5               # ...color (Lab a, b)
OPP_NOISE_EDGE = 6.0            # ...edge strength
OPP_SHADOW_DISCOUNT = 0.6       # soft-shadow pixels count this much
BG_ADAPT_RATE = 0.002           # very slow background update, only where nothing is detected
BG_FRAMES = 15                  # frames averaged for the empty-ring background (`c` / `b`)
OPP_TAPE_BAND = 0.06            # ignore this close to the tape line (ring units) when finding the opponent
OPP_SEARCH_RADIUS = 1.3         # look for the opponent out to this radius (it can be on/over the tape)
OUR_MASK_SCALE = 1.35           # our footprint is grown by this much before removing it from the mask
OPP_TAG_HEIGHT_CM = 8.0         # if the opponent wears any AprilTag (not ours), it is used instead

# --- Perspective (camera on a chair, looking at the ring from the side) ---
PERSPECTIVE_COLLECT_S = 2.0     # `p` watches the robot's tags this long
PERSPECTIVE_MAX_OBS = 300       # tag views kept for the fit (presses of `p` add up)
DEFAULT_TAG_HEIGHT_CM = 8.0     # tag height above the floor if robot_tags.json doesn't say
OPP_CENTROID_HEIGHT_CM = 4.0    # the opponent blob's middle is about this high

# --- Ring size (to convert cm -> ring units) ---
# None = estimate it from the AprilTag's size in the image; set a number (tape diameter,
# measured across the middle of the tape) to override.
RING_DIAMETER_CM = None
RING_DIAMETER_FALLBACK_CM = 77.0

# --- Color sensor: "not the floor" = tape ---
COLOR_BRIGHTER = 12             # reflection (0-100) this much above the floor = tape (light blue tape)
COLOR_BLUER = 0.05              # blue share of R+G+B this much above the floor = tape (deep blue tape)

# --- Drive check: does the robot move the way it's told? (catches a missing/wrong `d`) ---
DRIVE_CHECK_WINDOW_S = 0.3      # judged this often...
DRIVE_CHECK_SPAN_S = 0.8        # ...over the last this long
DRIVE_CHECK_MIN_CMD = 25        # only judge while commanding at least this forward/backward
DRIVE_CHECK_MIN_TURN_CMD = 10   # ...or this much turn (turning is slow now: at most ~22)
DRIVE_CHECK_MIN_SPEED = 0.08    # ...and the camera saw at least this speed (ring units/s)
DRIVE_CHECK_MIN_TURN = 0.5      # ...or turn rate (rad/s) for the turning checks
DRIVE_CHECK_LAG_S = 0.2         # commands take this long to show up as motion
DRIVE_CHECK_FAIL = 4            # wrong-way score (clear wrong-way windows count 2) that stops the match

# --- Control loop ---
CONTROL_HZ = 30
REFLEX_HZ = 200                 # color-sensor edge reflex thread

# --- Shake -> aggression ---
SHAKE_EMA = 0.6                 # per-message smoothing (20 Hz messages: ~0.05 s time constant)
SHAKE_TIMEOUT_S = 1.0           # no message for this long -> aggression decays to 0 (defense)
SHAKE_DECAY_S = 0.4             # ...with this time constant
DEFENSE_BELOW = 0.5             # aggression under this = DEFENSE (a hand at rest reads ~0.1)

# --- Pose EKF noise ---
EKF_Q_POS = 0.02                # ring units / sqrt(s): unmodeled motion (pushes, slip)
EKF_Q_THETA = math.radians(8)   # rad / sqrt(s): gyro noise / bias
EKF_Q_V = 0.8                   # ring units/s / sqrt(s): speed random walk
EKF_R_ENC_V = 0.05              # encoder speed measurement noise (ring units/s)
EKF_R_CAM_POS = 0.015           # camera position noise with both tags (ring units)
EKF_R_CAM_THETA = math.radians(4)
EKF_SINGLE_TAG_SCALE = 2.0      # one tag only: noise x this
EKF_GATE_POS = 0.45             # innovation bigger than this = suspect (tag misread)...
EKF_GATE_RESET_COUNT = 3        # ...unless it repeats this many frames, then re-center on the camera
POSE_CONFIDENT_SIGMA = 0.08     # position std below this = "pose confident"

# --- Opponent Kalman filter ---
OPP_Q_ACCEL = 1.5               # ring units/s^2
OPP_R_POS = 0.03
OPP_TIMEOUT_S = 1.0             # not seen this long = opponent unknown

# --- Edge risk ---
EDGE_SAFE_DIST = 0.25           # risk starts rising this far (ring units) inside the tape
EDGE_TTE_SAFE_S = 0.6           # time-to-tape under this raises risk
EDGE_ESCAPE_RISK = 0.8          # risk above this triggers EDGE_ESCAPE
EDGE_LIMIT_FLOOR = 0.15         # speed limiter never scales outward speed below this
ESCAPE_BRAKE_S = 0.0            # no pause: reversing at once stops the robot faster than braking
ESCAPE_BACK_S = 0.5
ESCAPE_BACK_SPEED = 40
REFLEX_REVERSE_SPEED = 50       # the color reflex sends this reverse command the moment it sees tape

# Stopping-distance speed limit (needs the camera). The robot must be able to stop before
# its leading end reaches the tape, counting the command delay and how fast it can brake.
STOP_LATENCY_S = 0.18           # sensor/camera -> laptop -> Bluetooth -> motor, plus a margin
STOP_DECEL = 2.0                # ring units/s^2 the robot can brake at (~0.8 m/s^2)
STOP_MARGIN = 0.06              # stop this far (ring units, ~2 cm) inside the tape
MAX_DRIVE_SPEED = 80            # no command is ever faster than this (%)
BLIND_MAX_SPEED = 22            # camera not tracking us: the color reflex alone can only stop this fast
ESCAPE_MIN_ROOM = 0.12          # less straight-line room than this both ways = arc inward instead
LIMIT_TURN_LOOKAHEAD_S = 0.4    # the speed limiter also checks the heading the turn reaches by then
ESCAPE_TURN_SPEED = 22          # slow: at 55% the robot overshot and flip-flopped left/right
ESCAPE_TURN_TOL = math.radians(25)
ESCAPE_TURN_TIMEOUT_S = 2.5
ESCAPE_BLIND_SPIN = math.radians(135)

# --- Lose detection ---
LOSE_OUT_FRACTION = 0.6         # camera: this much of our footprint past r = 1
LOSE_CAMERA_ALONE_S = 0.5       # ...for this long is enough by itself
LOSE_FUSED_R = 1.1              # fused center past this radius = a vote
LOSE_COLOR_WINDOW_S = 1.0       # BLUE -> gray while moving outward within this window = a vote

# --- Push detector (log-odds) ---
PUSH_WINDOW_S = 0.25            # camera vs odometry displacement window
PUSH_RESID_SPEED = 0.18         # ring units/s of unexplained motion = 1 "unit" of evidence
PUSH_ACCEL_MG = 220             # horizontal jolt (milli-g) beyond what our own driving explains
PUSH_YAW_DPS = 60               # gyro turn rate the wheels don't explain
PUSH_CONTACT_GAP = 0.06         # opponent closer than radii sum + this = touching
PUSH_STALL_CMD = 40             # commanding at least this...
PUSH_STALL_SPEED = 0.08         # ...while moving slower than this (ring units/s) = stalled
PUSH_WEIGHTS = {"resid": 2.2, "accel": 1.0, "yaw": 1.2, "contact": 1.3, "stall": 1.0}
PUSH_BIAS = -3.2                # log-odds with no evidence
PUSH_ON = 0.6
PUSH_OFF = 0.4
PUSH_NEAR_EDGE_R = 0.7

# Pushing the opponent out: while our front is on it and it is further out than we are, the
# tape rules relax so we finish the push instead of stopping short at the tape.
PUSH_OUT_NOSE_R = 1.10          # our front may go this far (ring units; the tape is at 1.0, ~4 cm past it)
PUSH_OUT_MAX_R = 0.92           # ...and our center no further than this
PUSH_OUT_HOLD_S = 0.3           # stays on this long after the contact flickers off (camera noise)
PUSH_OUT_MARGIN = 0.03          # opponent's center must be at least this much further out than ours
PUSH_FAR_GAP = 0.15             # opponent seen further than this (ring units, ~6 cm) past touching...
PUSH_FAR_WEIGHT = 4.0           # ...argues strongly against "being pushed" (it can't push from there)

# --- Behavior ---
# The robot turns ~8 deg/s per % and the camera + Bluetooth delay is ~0.25 s, so a strong
# steering gain overshoots every correction (it drove side to side at 90 / 70).
TURN_KP = 28.0                  # tank speed difference per radian of heading error
TURN_MAX = 22

# Speed: slow by default. The cap only ramps up while the opponent resists us (we're touching
# it and our wheels are driving but we're barely moving, or it is pushing us).
CRUISE_SPEED = 30               # fastest command with nothing to push against (%)
RESIST_START_SPEED = 30         # the moment resistance is detected the cap jumps to this...
RAMP_UP_PCT_S = 70              # ...then rises this fast while still resisted...
RAMP_DOWN_PCT_S = 150           # ...and falls back this fast when free
RESIST_MIN_CMD = 12             # need at least this forward/backward command to judge resistance
RESIST_SPEED_RATIO = 0.45       # moving slower than this fraction of the commanded speed = resisted
RESIST_HOLD_S = 0.2             # for this long
RESIST_TOUCH_GAP = 0.12         # opponent within this (ring units) past touching counts as "on us"
ALIGN_TOL = math.radians(15)    # ATTACK: charge once this well lined up
PUSHED_TURN_MAX = 40            # pushed: turn faster to square up (contact damps overshoot)
PUSHED_MIN_FWD = 0.5            # ...and keep driving into the opponent while turning...
PUSHED_PIVOT_ANGLE = math.radians(60)   # ...unless (defense) it's further off than this: pivot first
DEFENSE_HOME_R = 0.15           # defense: further from the center than this -> creep back...
DEFENSE_SETTLE_R = 0.06         # ...until this close, then stay still
DEFENSE_SPEED = 20
DEFENSE_FACE_TOL = math.radians(30)   # turn (slowly) to face the opponent only when this far off
DEFENSE_FACE_DONE = math.radians(12)
DEFENSE_TURN_MAX = 15
ATTACK_BASE_SPEED = 18
ATTACK_GAIN_SPEED = 12          # 18..30 %: approach speeds; pushing harder comes from the resist ramp
ATTACK_LEAD_S = 0.3             # aim where the opponent will be this far ahead
SEARCH_SPIN_SPEED = 16
BLIND_BASE_SPEED = 15
BLIND_GAIN_SPEED = 7
PUSH_INTENT_SPEED = 100         # pushing back / shoving: asks for full power, the resist ramp decides how much
COUNTDOWN_S = 3.0

# --- Drive model (defaults; `d` in sumo.py measures these into drive_calibration.json) ---
DEFAULT_DRIVE = {
    "reverse": False,               # True if the motor's "forward" drives toward the BACK tag
    "swap_sides": False,            # True if commanding a left turn turns the robot right (motor's
                                    # left/right ports swapped, or the phone image is mirrored)
    "enc_left_sign": -1,            # sign that makes each wheel's degrees count up when driving forward
    "enc_right_sign": 1,
    "ring_per_wheel_deg": 0.0013,   # ring units travelled per wheel degree (56 mm wheel, 77 cm ring)
    "track_ring": 0.28,             # wheel separation in ring units
    "wheel_dps_per_pct": 6.0,       # wheel deg/s per % speed (used only if encoders are missing)
    "gyro_axis": 2,                 # IMU gyro axis that measures yaw (0=x, 1=y, 2=z)
    "gyro_sign": -1,                # hub yaw is + clockwise from above; ring frame is + counterclockwise
    "gyro_scale": 1.0,              # deg/s per raw unit
}
