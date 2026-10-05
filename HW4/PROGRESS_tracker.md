# HW4 Progress Log — Minifig model + LED matrix tracker

A YOLOv8 model finds the LEGO minifig in the webcam. The computer sends the minifig's position over MQTT to an Arduino UNO Q, and the UNO Q lights the matching pixel on its built-in 13×8 LED matrix. The car program ([PROGRESS_car.md](PROGRESS_car.md)) uses the same model and MQTT setup.

## Files
| File | Role |
|---|---|
| [train_minifig.py](train_minifig.py) | Fine-tunes `yolov8n.pt` on the Roboflow dataset |
| [Minifig.v1i.yolov8/](Minifig.v1i.yolov8/) | Roboflow YOLOv8 export, 1 class `Minifig`: 70 train / 20 valid / 10 test (2 test images are backgrounds) |
| `runs/detect/minifig/weights/best.pt` | Trained model (committed; `last.pt` is gitignored) |
| [minifig_tracker.py](minifig_tracker.py) | Computer side: webcam → YOLO → matrix cell → MQTT |
| [mqttlib.py](mqttlib.py) | Copy of HW3's paho-mqtt wrapper |
| [uno_q_app/](uno_q_app/) | UNO Q App Lab app (`app.yaml`, `python/main.py`, `python/requirements.txt`, `sketch/sketch.ino`, `sketch/sketch.yaml`) |
| `minifig_tracker_app.zip` | Zip of `uno_q_app/` for App Lab import (gitignored; rebuild it, see below) |

## Environment
- Windows 11 laptop with an **RTX 3060 Laptop GPU (6 GB)** and an AMD integrated GPU. WMI lists only the AMD/Parsec adapters, and `nvidia-smi` fails with a permissions error, but CUDA works.
- In the project `.venv` (Python 3.12): `torch 2.14.1+cu126` and `torchvision` from `https://download.pytorch.org/whl/cu126`, plus `ultralytics 8.4.169`. These are **not** in `requirements.txt`, because the CUDA build needs that index URL.
- **Gotcha (2026-10-01):** `torch.cuda.is_available()` was `False` for a while; probably the laptop was on battery or the GPU was asleep. All scripts check this and fall back to CPU. Never hardcode `device=0`.
- `cv2.VideoCapture(0)` opens the laptop webcam fine.

## Training (`train_minifig.py`)
- Roboflow's `data.yaml` uses `../train/images`-style paths, which don't resolve from the dataset folder. The script writes `data_resolved.yaml` (gitignored) with absolute `path:` + `train/images` etc.
- Uses `if __name__ == "__main__":` (needed on Windows for dataloader workers), `project=HW4/runs/detect`, `name="minifig"`, `exist_ok=True`. The test-split `val()` also passes `project`, so its output stays under HW4 (it first landed in the repo-root `runs/`).
- 100 epochs, imgsz 640, about 2.3 minutes on the 3060.
- Results: val P 0.997 / R 1.0 / mAP50 0.995 / mAP50-95 0.932; **test** P 0.994 / R 1.0 / mAP50 0.995 / mAP50-95 0.970.
- Ultralytics also downloaded `HW4/yolov8n.pt` and the repo-root `weights/yolo26n.pt` (AMP check). Both are gitignored.

## v2 model (2026-10-05)
- The user added 44 phone photos (`PXL_20261005_*`) of the minifig **on the car** (side and top views, busy classroom background; 2 have no label, as negatives) and exported Roboflow **version 2** → `Minifig.v2i.yolov8/` (101 train / 29 valid / 14 test; same 512×512 stretch, no augmentation). The v1 folder was removed from disk (still in git history).
- First attempt: the user re-downloaded v1 by mistake (identical files). Check `version:` in `data.yaml` and the image counts before training.
- `train_minifig.py` now uses `DATASET_DIR = Minifig.v2i.yolov8`, `RUN_NAME = "minifig_v2"` → `runs/detect/minifig_v2/weights/best.pt`. The old `runs/detect/minifig/` model is kept for comparison. Training took ~2.6 minutes on the 3060.
- Results: val P 0.998 / R 1.0 / mAP50 0.995 / mAP50-95 0.912; test P 0.996 / R 1.0 / mAP50 0.995 / mAP50-95 0.946.
- **On the 13 held-out car photos** (v2 valid+test): the old model found the minifig in 3/13 at conf ≥ 0.25 and 0/13 at ≥ 0.5 (max 0.36). The new model found it in **13/13 at ≥ 0.5 (0.89–0.99)**. On the phone-camera frame where the old model found nothing, the new model scored 0.97 at imgsz 640 and 0.96 at 960.
- `WEIGHTS` in `minifig_tracker.py` (also imported by `minifig_car.py`) now points to `minifig_v2`.

## Tracker protocol
- Broker `test.mosquitto.org:1883`, topic **`ME193/hudson`** (the same topic as HW3's ball/goalie game).
- `minifig:<col>,<row>` (col 0–12 left→right, row 0–7 top→bottom) or `minifig:none`. The prefix is there so HW3's `Goal` / `I lost` messages are ignored.
- Computer: takes the highest-confidence box (conf ≥ 0.5) and maps the box center to a cell. It sends on change plus a resend every 2 s, and sends `none` after 10 missed frames and on exit. `MIRROR` flips columns. `--dry-run` prints instead of publishing. The preview window draws the grid, the box and the current cell; `q` quits.

## UNO Q app design
- UNO Q = STM32U585 MCU (Zephyr, runs the sketch, owns the LED matrix and GPIO) + QRB2210 MPU (Debian, runs Python, owns Wi-Fi). They talk only via `Arduino_RouterBridge` RPC.
- `python/main.py`: paho-mqtt subscribes in `on_connect` (so it re-subscribes after a reconnect). The `on_message` callback (paho's thread) only stores the latest parsed position under a lock. `loop()`, run by `App.run(user_loop=loop)`, makes the `Bridge.call("set_pixel", col, row)` / `Bridge.call("clear_matrix")` calls. `try/finally` around `App.run` does the cleanup, because `App.run` exits via `SystemExit`, so code after it never runs.
- `sketch.ino`: the Bridge handlers only set volatile state, and `loop()` draws. `matrix.begin(); matrix.setGrayscaleBits(3); matrix.clear();`, then `matrix.draw(uint8_t[104])` with index `row*13+col`, brightness **7** (0–7 scale; 1 was too dim). The matrix clears itself after 5 s with no update.
- The user chose option A: the built-in matrix only lights in its own color (not green). Green would need external RGB hardware.
- `app.yaml`: name, description, icon, `version`, `ports: []`, `bricks: []`. No Bricks needed.

## Problems hit and fixes (keep these in mind)
1. **Zip import refused.** The user's zip had everything inside a `uno_q_app/` folder plus a `__pycache__`. App Lab zips need `app.yaml` at the **zip root**. Build zips with Python `zipfile` using posix paths relative to the app folder, skipping `__pycache__` (don't use PS 5.1 `Compress-Archive`, which writes backslashes).
2. **`unexpected status code 405: Method Not Allowed` on import**, then **`fatal error: ArxTypeTraits.h: No such file or directory`** when building. App Lab auto-adds `MsgPack` (a RouterBridge → RPClite dependency) but didn't fetch the rest of the chain. Final `sketch.yaml`:
   ```yaml
   profiles:
     default:
       fqbn: arduino:zephyr:unoq
       platforms:
         - platform: arduino:zephyr
       libraries:
         - MsgPack (0.4.2)
         - DebugLog (0.8.4)
         - ArxContainer (0.7.0)
         - ArxTypeTraits (0.3.1)
   default_profile: default
   ```
   **Root cause of both errors: the UNO Q's system date was wrong** (TLS failures). Once the user fixed the date, the app worked. Check the board's clock first for any App Lab download/import error.
3. **2026-10-05: link errors after the board's platform update.** The board moved to `arduino:zephyr` **1.0.0** (toolchain `arm-zephyr-eabi 1.0.1`; App Lab now fetches `Arduino_RouterBridge 0.4.3` / `Arduino_RPClite 0.3.1` "from sketch project"). The build failed at link time: `cannot find entry symbol main` and undefined `arduino::String::*`, `Serial2`, `operator new(unsigned, void*)`, `_exit`. The core itself wasn't being linked. The board ID (`unoq`) is unchanged in 1.0.0, and both libraries are at their latest versions. No public issue found. Fix attempted (not yet confirmed): the tracker's `sketch.yaml` was reset to match the current official examples exactly (no `fqbn`, no `libraries`); the clock fix makes the old library pins unnecessary. If it still fails: run App Lab's board update / restart, and test an official Bridge example (e.g. LED Matrix Frame). If that fails too, the platform install is broken, not our app. The car app's `sketch.yaml` still has the old pins; change it the same way if it hits this.
4. There is no alternative to Bridge for MPU↔MCU communication. Don't open `Serial1` / `/dev/ttyHS1` directly.

## Rebuilding the App Lab zip
```python
import zipfile; from pathlib import Path
src = Path("uno_q_app")
with zipfile.ZipFile("minifig_tracker_app.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(src).as_posix())
```

## Status
- Verified working end to end on the board (2026-09-30), after the date fix.
- Commits: `e0d16b5` (model + tracker + app), `2491125` (sketch.yaml libs), `83856e7` (App.run loop / manifests).
