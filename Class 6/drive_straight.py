"""Drive the bar-walker forward using the learned Q-table (no learning, no exploration).

Put the robot down facing the direction you want; the heading at startup
becomes 0 deg. Every step it reads the yaw, looks up that state's row in
q_table.json, and runs the action with the highest Q-value. It keeps driving
until you press Ctrl+C.

Usage:
    python drive_straight.py
"""

import time

from q_learning_straight import (
    ACTION_NAMES,
    ACTIONS,
    Q_TABLE_PATH,
    STATE_NAMES,
    STEP_TIME,
    choose_action,
    connect,
    load_q_table,
    print_q_table,
    read_yaw,
    yaw_to_state,
    zero_heading,
)


def main():
    if not Q_TABLE_PATH.exists():
        print(f"No {Q_TABLE_PATH.name} found. Run q_learning_straight.py train first.")
        return

    q, _, _ = load_q_table()
    print_q_table(q)

    motor = connect()
    try:
        zero_heading(motor)
        print("Driving forward. Ctrl+C to stop.")
        while True:
            yaw = read_yaw(motor)
            state = yaw_to_state(yaw)
            action = choose_action(q, state, epsilon=0.0)
            motor.movement_move_tank(*ACTIONS[action], blocking=False)
            print(f"yaw {yaw:6.1f}  {STATE_NAMES[state]:>8} -> {ACTION_NAMES[action]}")
            time.sleep(STEP_TIME)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        motor.movement_stop()
        motor.disconnect()


if __name__ == "__main__":
    main()
