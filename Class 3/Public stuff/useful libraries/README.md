# Useful libraries

This folder will contain all the libraries we use in class to talk to the
LEGO Education hardware (motors, controller, color sensor) over Bluetooth.

## Setup

Use a virtual environment so your installs stay isolated from the rest of
your system Python:

```bash
python3 -m venv .venv

# activate it:
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

pip install legoeducation
```

Then copy `lelib.py` from this folder into your own project folder — it's a
thin wrapper around `legoeducation` that everything else in class builds on.

## Writing your own code

Once you have `lelib.py` next to your script, here's a recommended prompt
for an AI assistant to help you write code against the LEGO Education
hardware:

> I'm controlling LEGO Education hardware (motors, controller, color
> sensor) over Bluetooth using a wrapper called `lelib.py` — here's its
> contents: [paste `lelib.py`]. I want a script that connects to
> `<which devices, e.g. "a double motor and a color sensor">` using
> `lelib.py`'s `singleMotor` / `doubleMotor` / `controller` / `colorSensor`
> classes (not the raw `legoeducation` API directly), and does the
> following: `<describe the behavior you want>`. Use the connection-card
> pattern from `main.py` for the card color/serial.

The more specific you are about the behavior, the better the result —
describe what should happen for each color/button/joystick state, not just
"make it work."

## main.py

`main.py` is a template to build on. It's set up to try and control
**Spinning Fred** (the LEGO construction) — fill in the `Do...()` handler
functions with whatever behavior you want for each detected color and
controller state.
