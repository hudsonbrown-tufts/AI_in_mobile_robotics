# camlib — Camera selection reference

`camlib.py` handles everything needed to open a webcam in any ME193 example: it discovers connected cameras, prints them by name, asks you to choose one, opens it, and returns it ready to use.

```python
from camlib import pick_camera
cap, start_ms = pick_camera()
```

---

## pick_camera

Scan for cameras, show a numbered list with names, ask which one to use, then open and return it.

```python
cap, start_ms = pick_camera()
```

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width`   | `1280`  | Requested frame width in pixels |
| `height`  | `720`   | Requested frame height in pixels |

**Returns**

| Value | Type | Description |
|-------|------|-------------|
| `cap`      | `cv2.VideoCapture` | Opened camera, warmed up and ready to read |
| `start_ms` | `int`              | Time in milliseconds when the camera opened — use this as a timestamp base for MediaPipe VIDEO mode |

**Raises** `RuntimeError` if no cameras are found, or if the chosen camera cannot be opened.

---

## What happens when you call it

A window opens showing a live preview thumbnail of every connected camera, labelled with its index and name:

```
┌─────────────────────┐  ┌─────────────────────┐
│ [0] FaceTime Camera │  │ [1] Camo             │
│                     │  │                     │
│   (live preview)    │  │   (live preview)    │
│                     │  │                     │
│  press  0  to select│  │  press  1  to select│
└─────────────────────┘  └─────────────────────┘
```

Press the **number key** for the camera you want. Press **Q** or **Esc** to accept the default (first camera). The window closes and the selected camera opens automatically.

If only one camera is found the window is skipped entirely.

> **Why thumbnails instead of just names?** macOS lists camera names in a different order than OpenCV opens them, so text labels alone can be misleading. Seeing the actual image from each camera is unambiguous.

---

## Basic usage — show webcam feed

```python
import cv2
from camlib import pick_camera

cap, _ = pick_camera()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Usage with MediaPipe (VIDEO mode)

MediaPipe's VIDEO running mode requires a monotonically increasing timestamp in milliseconds for each frame. `start_ms` gives you the reference point to compute that offset.

```python
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from camlib import pick_camera

cap, start_ms = pick_camera()

# … set up your MediaPipe landmarker …

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts_ms    = int(time.time() * 1000) - start_ms        # timestamp for this frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect_for_video(mp_image, ts_ms)
```

---

## Custom resolution

```python
cap, start_ms = pick_camera(width=640, height=480)   # lower res for faster inference
cap, start_ms = pick_camera(width=1920, height=1080) # full HD if your camera supports it
```

---

## Notes

- **macOS only** — camera names come from `system_profiler SPCameraDataType`. On other platforms names fall back to `"Camera 0"`, `"Camera 1"`, etc., and everything else still works.
- **AVFoundation backend** — the camera is opened with `cv2.CAP_AVFOUNDATION` first, then falls back to the default OpenCV backend if that fails. This avoids a common macOS issue where `cap.read()` returns empty frames.
- **Warm-up** — a 0.5 s pause after opening lets the camera sensor initialise before the first frame is read, preventing black or garbled first frames.
- **`start_ms`** — if you are not using MediaPipe you can ignore this value with `cap, _ = pick_camera()`.
