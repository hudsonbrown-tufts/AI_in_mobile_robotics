"""
Hardware smoke test for lelib.

Connects to any single motor, double motor, controller, and color sensor
(no specific card required — pass a card serial with e.g. --color-serial if
you need to disambiguate multiple devices of the same type), reads the
color sensor and controller joysticks, then runs the double motor for
1 second (reporting its yaw before and after) and spins the single motor
2 rotations.

Install first:
    pip install legoeducation
Then copy lelib.py from the SimpleLE repo into this project's folder.
"""

from lelib import singleMotor, doubleMotor, controller, colorSensor


def test_devices():
    """Connect to one of each device, read sensors, run the motors, disconnect."""
    color = colorSensor()
    ctl = controller()
    dm = doubleMotor()
    sm = singleMotor()

    print("Connecting to color sensor...")
    color.connect(card_serial=None)
    print("Connecting to controller...")
    ctl.connect(card_serial=None)
    print("Connecting to double motor...")
    dm.connect(card_serial=None)
    print("Connecting to single motor...")
    sm.connect(card_serial=None)

    try:
        detected_color = color.detect_color()
        left_position = ctl.left_position()
        right_position = ctl.right_position()
        print(f"Detected color: {detected_color}")
        print(f"Joystick position: left={left_position}, right={right_position}")

        dm.reset_heading()
        print("Running double motor for 1 second...")
        dm.run_time(1000)
        yaw = dm.yaw()
        print(f"Yaw after run: {yaw} degrees")

        print("Spinning single motor 2 rotations...")
        sm.spin(2)
    finally:
        dm.stop()
        sm.stop()
        color.disconnect()
        ctl.disconnect()
        dm.disconnect()
        sm.disconnect()

    return {
        "color": detected_color,
        "left_position": left_position,
        "right_position": right_position,
        "yaw": yaw,
    }


if __name__ == "__main__":
    result = test_devices()
    print(result)
