"""Q-learning to make the LEGO Double Motor bar-walker drive straight.

State  = the Double Motor's IMU yaw, binned into 7 states.
Action = one of 5 tank (left, right) speed pairs.
Update = Q_new(s,a) = (1 - alpha) * Q_old(s,a) + alpha * (r + gamma * max_a' Q(s',a'))

Training is fully automatic: put the robot down facing the direction you want,
start the script, and the heading at startup becomes 0 deg. After every episode
the robot spins in place back to 0 deg and starts the next one. Every few
episodes it runs a greedy tesqt episode and reports PASS/fail as a progress
check. Training keeps going until you press Ctrl+C.

Usage:
    python q_learning_straight.py check   # print live yaw; turn robot by hand to confirm sign
    python q_learning_straight.py train   # learn until Ctrl+C (Q-table saved after every episode)
    python q_learning_straight.py run     # drive using the learned table (no exploration)
    python q_learning_straight.py show    # print the saved Q-table
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import legoeducation as le

# Update these to match the Connection Card printed on/with your Double Motor.
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "5164"

Q_TABLE_PATH = Path(__file__).with_name("q_table.json")

YAW_RAW_PER_DEGREE = 10.0  # imu_device.yaw is in tenths of a degree

# --- States: yaw bins (degrees). Positive yaw = drifted right (clockwise). ---
# Upper edges of S0..S5; anything above the last edge is S6.
YAW_BIN_EDGES = [-15, -7, -2, 2, 7, 15]
STATE_NAMES = ["far L", "left", "sl. L", "STRAIGHT", "sl. R", "right", "far R"]

# --- Actions: (left speed, right speed) ---
BASE_SPEED = 50
ACTIONS = [
    (BASE_SPEED - 20, BASE_SPEED + 20),  # A0 hard left
    (BASE_SPEED - 8, BASE_SPEED + 8),    # A1 soft left
    (BASE_SPEED, BASE_SPEED),            # A2 straight
    (BASE_SPEED + 8, BASE_SPEED - 8),    # A3 soft right
    (BASE_SPEED + 20, BASE_SPEED - 20),  # A4 hard right
]
ACTION_NAMES = ["hard L", "soft L", "straight", "soft R", "hard R"]

# --- Learning parameters ---
ALPHA = 0.3          # learning rate
GAMMA = 0.9          # discount
EPSILON_START = 1.0
EPSILON_DECAY = 0.95  # multiplied in after every episode
EPSILON_MIN = 0.05

# --- Episode parameters ---
STEP_TIME = 0.3      # seconds each action runs before reading yaw again
MAX_STEPS = 40       # ~12 s per episode
FAIL_YAW = 30.0      # |yaw| beyond this ends the episode
START_DELAY = 3      # seconds after connecting before the heading is zeroed

# --- Recovery: spin in place back to 0 deg between episodes ---
RECOVER_TOLERANCE = 2.0   # degrees
RECOVER_GAIN = 1.5        # spin speed per degree of yaw
RECOVER_MIN_SPEED = 20
RECOVER_MAX_SPEED = 40
RECOVER_TIMEOUT = 10.0    # seconds

# --- Progress check: greedy test episodes (reported only; training never stops) ---
EVAL_EVERY = 5            # run a test episode after every N training episodes
GOOD_MEAN_YAW = 3.0       # test passes if it never fails and mean |yaw| <= this

# --- Reward parameters ---
STRAIGHT_YAW = 2.0
STRAIGHT_REWARD = 2.0
YAW_PENALTY_SCALE = 10.0
IMPROVEMENT_WEIGHT = 0.5
FAIL_REWARD = -10.0


def yaw_to_state(yaw):
    for state, edge in enumerate(YAW_BIN_EDGES):
        if yaw < edge:
            return state
    return len(YAW_BIN_EDGES)


def compute_reward(yaw_old, yaw_new):
    if abs(yaw_new) > FAIL_YAW:
        return FAIL_REWARD, True
    if abs(yaw_new) <= STRAIGHT_YAW:
        reward = STRAIGHT_REWARD
    else:
        reward = -abs(yaw_new) / YAW_PENALTY_SCALE
    reward += IMPROVEMENT_WEIGHT * (abs(yaw_old) - abs(yaw_new))
    return reward, False


def q_update(q, state, action, reward, next_state, done):
    """Q_new(s,a) = (1 - alpha) * Q_old(s,a) + alpha * (r + gamma * Q_hat(s'))"""
    q_hat_next = 0.0 if done else max(q[next_state])
    q[state][action] = (1 - ALPHA) * q[state][action] + ALPHA * (reward + GAMMA * q_hat_next)


def choose_action(q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    best = max(q[state])
    # Break ties randomly so an all-zero row doesn't always pick A0.
    return random.choice([a for a, value in enumerate(q[state]) if value == best])


def load_q_table():
    if Q_TABLE_PATH.exists():
        data = json.loads(Q_TABLE_PATH.read_text())
        print(f"Loaded Q-table from {Q_TABLE_PATH.name} (episode {data['episode']}, epsilon {data['epsilon']:.3f})")
        return data["q"], data["epsilon"], data["episode"]
    q = [[0.0] * len(ACTIONS) for _ in STATE_NAMES]
    return q, EPSILON_START, 0


def save_q_table(q, epsilon, episode):
    Q_TABLE_PATH.write_text(json.dumps({"q": q, "epsilon": epsilon, "episode": episode}, indent=2))


def print_q_table(q):
    header = f"{'':>10}" + "".join(f"{name:>10}" for name in ACTION_NAMES) + "   best"
    print(header)
    for state, row in enumerate(q):
        best = ACTION_NAMES[max(range(len(row)), key=row.__getitem__)]
        print(f"{STATE_NAMES[state]:>10}" + "".join(f"{value:>10.2f}" for value in row) + f"   {best}")


def connect():
    motor = le.DoubleMotor()
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not motor.connected:
        print("Error connecting to Double Motor.")
        exit(1)
    motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
    motor.movement_stop()
    return motor


def read_yaw(motor):
    # Wait out the NaN placeholder the library uses before the first IMU notification.
    while math.isnan(motor.imu_device.yaw):
        time.sleep(0.02)
    # The hub reports yaw in tenths of a degree (-1800..1800).
    return motor.imu_device.yaw / YAW_RAW_PER_DEGREE


def zero_heading(motor):
    print(f"Hold still: zeroing heading in {START_DELAY} s...")
    time.sleep(START_DELAY)
    motor.imu_reset_yaw_axis(0)
    time.sleep(0.2)
    print("Heading set to 0 deg.")


def recover_heading(motor):
    """Spin in place until the robot faces 0 deg again."""
    start = time.time()
    yaw = read_yaw(motor)
    while abs(yaw) > RECOVER_TOLERANCE and time.time() - start < RECOVER_TIMEOUT:
        speed = min(RECOVER_MAX_SPEED, max(RECOVER_MIN_SPEED, RECOVER_GAIN * abs(yaw)))
        # Positive yaw = facing right, so spin left (left back, right forward).
        spin = speed if yaw > 0 else -speed
        motor.movement_move_tank(int(-spin), int(spin), blocking=False)
        time.sleep(0.05)
        yaw = read_yaw(motor)
    motor.movement_stop()
    time.sleep(0.3)
    print(f"  recovered heading: yaw {read_yaw(motor):.1f}")


def run_episode(motor, q, epsilon, learn):
    """Drive one episode starting from the current heading (0 deg is set once at startup)."""
    yaw = read_yaw(motor)
    state = yaw_to_state(yaw)
    total_reward = 0.0
    total_abs_yaw = 0.0
    failed = False
    steps = 0

    for steps in range(1, MAX_STEPS + 1):
        action = choose_action(q, state, epsilon)
        motor.movement_move_tank(*ACTIONS[action], blocking=False)
        time.sleep(STEP_TIME)

        new_yaw = read_yaw(motor)
        new_state = yaw_to_state(new_yaw)
        reward, done = compute_reward(yaw, new_yaw)
        total_reward += reward
        total_abs_yaw += abs(new_yaw)
        if learn:
            q_update(q, state, action, reward, new_state, done)

        print(f"  step {steps:2d}  yaw {yaw:6.1f} -> {new_yaw:6.1f}  "
              f"{STATE_NAMES[state]:>8} {ACTION_NAMES[action]:>8}  r {reward:6.2f}")

        yaw, state = new_yaw, new_state
        if done:
            failed = True
            break

    motor.movement_stop()
    return total_reward, steps, failed, total_abs_yaw / steps


def train(motor):
    q, epsilon, episode = load_q_table()
    zero_heading(motor)
    good_tests = 0
    try:
        while True:
            recover_heading(motor)
            print(f"\nEpisode {episode + 1} (epsilon {epsilon:.3f})")
            total_reward, steps, failed, mean_yaw = run_episode(motor, q, epsilon, learn=True)
            episode += 1
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            save_q_table(q, epsilon, episode)
            print(f"Episode {episode}: {steps} steps, total reward {total_reward:.2f}, "
                  f"mean |yaw| {mean_yaw:.1f}{'  FAILED' if failed else ''}")
            print_q_table(q)

            if episode % EVAL_EVERY == 0:
                recover_heading(motor)
                print("\nTest episode (greedy, no learning)")
                _, steps, failed, mean_yaw = run_episode(motor, q, epsilon=0.0, learn=False)
                passed = not failed and mean_yaw <= GOOD_MEAN_YAW
                good_tests = good_tests + 1 if passed else 0
                print(f"Test: {steps} steps, mean |yaw| {mean_yaw:.1f} -> "
                      f"{'PASS' if passed else 'fail'} ({good_tests} passed in a row)")
    except KeyboardInterrupt:
        print("\nStopping training; Q-table saved.")
    motor.movement_stop()
    print_q_table(q)


def run(motor):
    q, _, _ = load_q_table()
    print_q_table(q)
    zero_heading(motor)
    try:
        while True:
            recover_heading(motor)
            total_reward, steps, failed, mean_yaw = run_episode(motor, q, epsilon=0.0, learn=False)
            print(f"{steps} steps, total reward {total_reward:.2f}, mean |yaw| {mean_yaw:.1f}"
                  f"{'  FAILED' if failed else ''}")
    except KeyboardInterrupt:
        pass



def check(motor):
    motor.imu_reset_yaw_axis(0)
    print("Turn the robot by hand. Turning RIGHT (clockwise) should give POSITIVE yaw. Ctrl+C to quit.")
    try:
        while True:
            yaw = read_yaw(motor)
            print(f"yaw {yaw:7.1f}  state {STATE_NAMES[yaw_to_state(yaw)]}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "run", "check", "show"], nargs="?", default="train")
    args = parser.parse_args()

    if args.mode == "show":
        print_q_table(load_q_table()[0])
        return

    motor = connect()
    try:
        {"train": train, "run": run, "check": check}[args.mode](motor)
    finally:
        motor.movement_stop()
        motor.disconnect()


if __name__ == "__main__":
    main()
