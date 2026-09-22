import time

import legoeducation as le

# Update these to match the Connection Card printed on/with your Single Motor.
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "7589"

motor = le.SingleMotor()
motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

if not motor.connected:
    print("Error connecting to Single Motor.")
    exit(1)

motor.motor_reset_relative_position()
motor.motor_run(speed=20)

for _ in range(50):
    print(f"Current position: {motor.motor.position}")
    if motor.motor.position > 360:
        motor.motor_run(speed=80)
    time.sleep(0.1)

motor.motor_stop()
motor.disconnect()
