# test_devices.py

A quick hardware smoke test for `lelib.py`. Run it with everything powered on
and in BLE range:

```bash
python3 test_devices.py
```

What it does, in order:

1. Connects to a color sensor, controller, double motor, and single motor —
   each with `card_serial=None`, so it grabs whichever device of that type
   is nearby instead of requiring a specific card.
2. Prints the color sensor's detected color and both controller joystick
   positions.
3. Zeroes the double motor's heading, drives it for 1 second, then prints
   the yaw it picked up during that run.
4. Spins the single motor 2 full rotations.
5. Stops and disconnects all four devices, even if something above raised
   an error.

Returns a dict with `color`, `left_position`, `right_position`, and `yaw` for
scripting/logging.
