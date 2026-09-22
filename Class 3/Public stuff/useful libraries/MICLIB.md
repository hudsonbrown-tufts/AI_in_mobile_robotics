# miclib — Microphone selection reference

`miclib.py` handles audio input device selection for ME193 AI examples.  It lists every microphone the system can see, asks you to pick one, and returns the device index ready to pass to `sounddevice`.

```python
from miclib import pick_mic
device_index = pick_mic()
```

---

## pick_mic

List all audio input devices, prompt the user to choose one, and return the device index.

```python
device_index = pick_mic()
```

**Parameters** — none.

**Returns**

| Value | Type | Description |
|-------|------|-------------|
| `device_index` | `int` or `None` | Index to pass to `sd.InputStream(device=...)`. `None` means use the system default. |

**Raises** — never raises.  If no input devices are found, or the user presses Enter without typing a number, it returns `None` (system default) and prints a short message.

---

## What happens when you call it

The available input devices are printed with their index numbers:

```
Available microphone inputs:
  [0] MacBook Air Microphone ← default
  [1] Camo Microphone
  [2] USB Audio Device

Enter device number (or press Enter for default):
```

- Type a number and press Enter to select that device.
- Press Enter with nothing typed to accept the system default (shown with `← default`).
- If you type something that is not a valid input-device number, the system default is used.

---

## Basic usage — open a stream on the chosen device

```python
import sounddevice as sd
from miclib import pick_mic

SAMPLE_RATE = 44100
CHUNK_SIZE  = 2048

device_index = pick_mic()

def callback(indata, frames, time, status):
    print(f"rms = {(indata**2).mean()**0.5:.4f}")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                    blocksize=CHUNK_SIZE, dtype="float32",
                    device=device_index,
                    callback=callback):
    input("Recording — press Enter to stop.")
```

---

## Usage in whistle-commands

Both `calibrate.py` and `whistle.py` use `pick_mic()` to let you choose which microphone feeds the FFT pitch detector:

```python
from miclib import pick_mic
import sounddevice as sd

mic_device = pick_mic()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                    blocksize=CHUNK_SIZE, dtype="float32",
                    device=mic_device,
                    callback=_audio_callback):
    # ... pitch detection loop ...
```

---

## Using the system default

If you always want the system default (no prompt), pass `device=None` directly to `sd.InputStream` — `pick_mic()` returns `None` in that case anyway:

```python
device_index = pick_mic()   # user presses Enter → returns None
# equivalent:
sd.InputStream(device=None, ...)   # sounddevice uses system default
```

---

## Listing devices without prompting

To inspect devices in your own code without a prompt, use `sounddevice` directly:

```python
import sounddevice as sd

for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"[{i}] {d['name']}  ({d['default_samplerate']:.0f} Hz)")
```

---

## Notes

- **Indices are stable** within a session but can change between reboots or when devices are plugged/unplugged.  Always call `pick_mic()` at startup rather than hard-coding an index.
- **Channels** — `pick_mic()` only lists devices that have at least one input channel.  Output-only devices (speakers) are never shown.
- **Sample rate** — the device index returned has no effect on sample rate; you still set `samplerate=` in `sd.InputStream` yourself.  If the device doesn't support your chosen rate, `sounddevice` will raise a `PortAudioError` with a clear message.
- **macOS microphone permission** — if the terminal has not been granted microphone access, `sd.InputStream` will open but deliver silence.  Grant access in System Settings → Privacy & Security → Microphone, then restart the terminal.
