# lelib — SimpleLE wrapper reference

`lelib.py` wraps the LEGO Education Bluetooth library with friendlier class names, automatic retry-on-connect, and extra convenience methods.  Every example folder imports from this single file.

```python
from lelib import singleMotor, doubleMotor, controller, colorSensor
```

---

## Card reading (all devices)

Every device — `singleMotor`, `doubleMotor`, `controller`, and `colorSensor` — can read LEGO cards placed on the hub's NFC sensor.

### card_serial

Return the serial number of the card currently resting on the hub.
Returns `0` when no card is present.

```python
dm = doubleMotor()
dm.connect(2279)
print(dm.card_serial())   # e.g. 1042  (or 0 if no card)
```

---

### card_tapped

Return the card's serial number the moment a new card is placed, and `None` on every subsequent call until the card changes.  Use this in a polling loop — no need to track the previous value yourself.

```python
while True:
    serial = dm.card_tapped()
    if serial:
        print(f"Card tapped — serial: {serial}")
    time.sleep(0.1)
```

Works identically on any device:

```python
sm.card_tapped()    # singleMotor
dm.card_tapped()    # doubleMotor
ctrl.card_tapped()  # controller
cs.card_tapped()    # colorSensor
```

Mapping cards to actions:

```python
CARD_ACTIONS = {
    1042: "forward",
    1055: "left",
    1071: "right",
    1088: "stop",
}

while True:
    serial = dm.card_tapped()
    if serial:
        action = CARD_ACTIONS.get(serial, "unknown")
        print(f"Card {serial} → {action}")
    time.sleep(0.1)
```

---

## singleMotor

Controls one LEGO motor hub over Bluetooth.

### connect

```python
sm = singleMotor()
sm.connect(2279)          # connect by serial number
sm.connect(2279, card_color="white")  # optionally filter by card colour
```

Retries up to 5 times with 1-second pauses before raising `ConnectionError`.

---

### spin

Rotate the motor shaft a given number of full turns.

```python
sm.spin()        # one full rotation (default)
sm.spin(3)       # three full rotations
sm.spin(0.5)     # half a rotation
```

---

### run

Start the motor spinning continuously.  Pass a speed from −100 to 100.
Positive = clockwise, negative = counter-clockwise, default = 50.

```python
sm.run()       # clockwise at 50%
sm.run(80)     # clockwise at 80%
sm.run(-80)    # counter-clockwise at 80%
time.sleep(2)
sm.stop()
```

---

### set_speed

Set the default speed used by other methods that don't take a speed argument.

```python
sm.set_speed(75)
sm.spin(2)     # spins at 75%
```

---

### stop

Stop the motor immediately.

```python
sm.stop()
```

---

## doubleMotor

Controls the LEGO double-motor hub (two drive motors + built-in IMU).

### connect

```python
dm = doubleMotor()
dm.connect(2279)
```

---

### move_steps

Move the robot forward one "step" — defined as 180° of wheel rotation.  Useful for grid-based movement.

```python
dm.move_steps()     # one step forward
dm.move_steps(3)    # three steps forward
```

---

### run

Start both motors moving continuously.  Pass a speed from −100 to 100.
Positive = forward, negative = backward, default = 50.

```python
dm.run()       # forward at 50%
dm.run(80)     # forward at 80%
dm.run(-60)    # backward at 60%
time.sleep(2)
dm.stop()
```

---

### run_time

Run both motors for a fixed duration in milliseconds.

```python
dm.run_time(2000)   # run for 2 seconds
dm.run_time(500)    # run for 0.5 seconds
```

---

### run_left / run_right

Run only the left or right motor.  Pass a number of degrees to run for a fixed amount, or nothing to run continuously.

```python
dm.run_left()          # left motor runs continuously
dm.run_left(180)       # left motor turns 180°

dm.run_right()         # right motor runs continuously
dm.run_right(360)      # right motor turns one full revolution
```

---

### turn_left / turn_right

Spin the whole robot in place by a given number of degrees.

```python
dm.turn_left()         # 90° left (default)
dm.turn_left(45)       # 45° left
dm.turn_right(180)     # 180° right (U-turn)
```

---

### set_speed / set_speed_left / set_speed_right

Set motor speeds as percentages (0–100).

```python
dm.set_speed(60)          # both motors at 60%
dm.set_speed_left(80)     # left motor only
dm.set_speed_right(40)    # right motor only
```

---

### stop

Stop both motors immediately.

```python
dm.stop()
```

---

### Tank drive (inherited from DoubleMotor)

Drive each side at an independent speed (−100 to 100).  Positive = forward, negative = backward.  This is the main method for joystick or AI-driven control.

```python
dm.movement_move_tank(left_speed, right_speed)

# examples:
dm.movement_move_tank(50, 50)    # straight ahead at 50%
dm.movement_move_tank(50, -50)   # spin left in place
dm.movement_move_tank(0, 0)      # stop
```

---

### Position control (inherited from DoubleMotor)

Move a single motor to a position relative to where it currently is.

```python
import legoeducation as le

dm.motor_run_to_relative_position(90,  motor=le.MOTOR_LEFT)   # left  +90°
dm.motor_run_to_relative_position(-45, motor=le.MOTOR_RIGHT)  # right -45°
dm.motor_reset_relative_position(motor=le.MOTOR_LEFT)         # zero encoder
```

---

### IMU — reset_heading

Zero the yaw angle at the robot's current orientation.  Call this once before any straight-line driving or heading-based control.

```python
dm.connect(2279)
dm.reset_heading()   # "this direction is 0°"
```

---

### IMU — yaw

Return the current yaw angle in degrees relative to the last `reset_heading()` call.  Positive = clockwise drift, negative = counter-clockwise drift.

```python
dm.reset_heading()
dm.run_time(3000)
print(dm.yaw())   # e.g. 7.0  means the robot drifted 7° right
```

---

### IMU — gyro_z

Return the Z-axis angular velocity as a raw integer (degrees per second, approximately).  Useful for detecting active rotation rather than accumulated drift.

```python
while True:
    print(f"yaw={dm.yaw():.1f}°   spin={dm.gyro_z():.0f}°/s")
    time.sleep(0.1)
```

---

## controller

Reads the LEGO joystick/controller over Bluetooth.

### connect

```python
ctrl = controller()
ctrl.connect(2279)
```

---

### left_position / right_position

Read each joystick as a percentage: −100 (full down/back) to +100 (full up/forward).  These are the main values for tank drive.

```python
left  = ctrl.left_position()   # e.g. 75
right = ctrl.right_position()  # e.g. -30
dm.movement_move_tank(left, right)
```

---

### left_up / left_down / left_released

Boolean checks for quick directional decisions.

```python
if ctrl.left_up():
    print("left stick pushed forward")

if ctrl.left_down():
    dm.turn_left()

if ctrl.left_released():
    dm.stop()
```

---

### right_up / right_down / right_released

Same as above for the right joystick.

```python
if ctrl.right_up():
    sm.spin()

if ctrl.right_released():
    sm.stop()
```

---

### drive

Run a simple tank-drive loop for `t` iterations (each 100 ms), reading both sticks and driving the double motor.

```python
ctrl.drive(dm)           # 100 iterations ≈ 10 seconds
ctrl.drive(dm, t=200)    # 200 iterations ≈ 20 seconds
```

---

## colorSensor

Reads the LEGO color sensor over Bluetooth.

### connect

```python
cs = colorSensor()
cs.connect(2279)
```

---

### detect_color

Return the LEGO built-in color name as a string.

```python
name = cs.detect_color()
print(name)   # e.g. "Red", "Blue", "Green", "White", "No color"
```

Possible values: `No color`, `Red`, `Yellow`, `Blue`, `Teal`, `Green`, `Purple`, `White`, `Magenta`, `Orange`, `Azure`.

---

### reflection

Return the raw reflection value (0–255).  Higher = more light reflected (lighter surface).

```python
r = cs.reflection()
if r > 200:
    print("white surface")
elif r < 30:
    print("black surface")
```

---

### raw_rgb

Return the raw red, green, and blue channel values as a tuple of 16-bit integers (0–65535).  More precise than the built-in colour detection.

```python
r, g, b = cs.raw_rgb()
print(f"R={r}  G={g}  B={b}")
```

---

### raw_reading

Return all sensor channels in a single dictionary, each normalised to the range 0–1.  Use this as the feature vector for ML classifiers (KNN, neural net, etc.).

```python
reading = cs.raw_reading()
print(reading)
# {
#   'rawRed':     0.412,
#   'rawGreen':   0.198,
#   'rawBlue':    0.083,
#   'reflection': 0.631,
#   'hue':        0.057,
#   'saturation': 0.712,
#   'value':      0.784,
# }
```

Keys:

| Key | Source | Range |
|-----|--------|-------|
| `rawRed` | 16-bit red channel | 0–1 |
| `rawGreen` | 16-bit green channel | 0–1 |
| `rawBlue` | 16-bit blue channel | 0–1 |
| `reflection` | reflected light intensity | 0–1 |
| `hue` | hue from HSV | 0–1 |
| `saturation` | saturation from HSV | 0–1 |
| `value` | brightness from HSV | 0–1 |

---

## Full examples

### Tank drive with joystick

```python
import time
from lelib import doubleMotor, controller

SERIAL = 2279
dm   = doubleMotor()
ctrl = controller()
dm.connect(SERIAL)
ctrl.connect(SERIAL)

try:
    while True:
        dm.movement_move_tank(ctrl.left_position(), ctrl.right_position())
        time.sleep(0.05)
finally:
    dm.stop()
```

---

### Drive straight using the IMU

```python
import time
from lelib import doubleMotor

SERIAL    = 2279
BASE      = 50     # % speed
KP        = 2.0    # proportional gain

dm = doubleMotor()
dm.connect(SERIAL)
dm.reset_heading()

try:
    while True:
        error      = dm.yaw()               # degrees off straight
        correction = KP * error
        dm.movement_move_tank(BASE - correction, BASE + correction)
        time.sleep(0.05)
finally:
    dm.stop()
```

---

### Read colour sensor and sort

```python
import time
from lelib import doubleMotor, colorSensor

SERIAL = 2279
dm = doubleMotor()
cs = colorSensor()
dm.connect(SERIAL)
cs.connect(SERIAL)

while True:
    color = cs.detect_color()
    if color == "Red":
        dm.turn_right(90)
    elif color == "Blue":
        dm.turn_left(90)
    time.sleep(0.2)
```

---

### Collect raw readings for ML training

```python
import time
from lelib import colorSensor

cs = colorSensor()
cs.connect(2279)

samples = []
label   = "red"

for _ in range(20):
    samples.append(cs.raw_reading())
    time.sleep(0.1)

print(f"Collected {len(samples)} samples for '{label}'")
print(samples[0])
```
