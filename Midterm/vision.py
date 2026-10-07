"""Camera thread: MediaPipe Pose for the paddle (right wrist) and AprilTag
detection for the player card.

Pose and AprilTag both run on the UN-mirrored camera frame. The tag can't
be decoded mirrored, and MediaPipe's left/right labels are anatomical, so
landmark 16 is my real right wrist. Only the picture and the coordinates
handed to the GUI are mirrored, so the screen works like a mirror.

The wrist is mapped to a paddle position over my end of the table
*relative to my body* (shoulder midpoint and torso size), so it works
whether I stand close to the camera or far from it:
    paddle x: wrist distance right/left of the shoulder midpoint, in shoulder widths
    paddle y: wrist height between hip (table height) and shoulder
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.aruco as aruco
from cv2_enumerate_cameras import enumerate_cameras
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from game import TABLE_HALF_W, clamp

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "pose_landmarker_lite.task"

# Camera selection, same as HW4: the Android phone shared through Windows
# Phone Link shows up as "<phone name> (Windows Virtual Camera)". If no
# camera matches, CAMERA_INDEX is used instead (0 = laptop webcam).
CAMERA_NAME_HINT = "Windows Virtual Camera"
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (1280, 720)
DISPLAY_SIZE = (512, 288)      # camera panel size in the GUI

# True if the camera itself already sends a mirrored (selfie) image. Toggle
# live with the M key: the paddle following your LEFT hand means it's wrong.
CAMERA_MIRRORED = False

APRILTAG_DICTIONARY = aruco.DICT_APRILTAG_36h11

# --- Pose landmarks (MediaPipe numbering) ---
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
PADDLE_WRIST = R_WRIST         # right-handed
SKELETON = [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP),
            (R_SHOULDER, R_HIP), (L_HIP, R_HIP)]

# --- Wrist -> paddle mapping ---
SHOULDER_WIDTHS_TO_EDGE = 2.0  # wrist this many shoulder widths to the side = table edge
PADDLE_Y_AT_SHOULDER = 0.42    # m above the table when the wrist is at shoulder height
TORSO_PER_SHOULDER_W = 1.4     # torso length guess if the hips are out of frame
MIN_VISIBILITY = 0.5
SMOOTHING = 0.55               # EMA weight of the newest wrist sample (higher = snappier)


@dataclass
class VisionResult:
    frame_rgb: "object" = None      # mirrored RGB frame at DISPLAY_SIZE
    landmarks: dict = None          # {index: (u, v)} mirrored, normalized 0..1
    wrist_uv: tuple = None          # mirrored, normalized
    paddle: tuple = None            # (x, y) in table meters, or None if no wrist
    tag_id: int = None
    tag_corners: list = None        # mirrored, normalized [(u, v)] * 4
    fps: float = 0.0
    t: float = 0.0


def list_cameras():
    for cam in enumerate_cameras(cv2.CAP_MSMF):
        print(f"  [{cam.index}] {cam.name}")


def open_camera():
    """Open the camera matching CAMERA_NAME_HINT, falling back to CAMERA_INDEX."""
    cap = None
    if CAMERA_NAME_HINT:
        for cam in enumerate_cameras(cv2.CAP_MSMF):
            if CAMERA_NAME_HINT.lower() in cam.name.lower():
                cap = cv2.VideoCapture(cam.index, cam.backend)
                if cap.isOpened():
                    print(f"[vision] Using camera [{cam.index}] {cam.name}")
                    break
                print(f"[vision] Found '{cam.name}' but couldn't open it.")
                cap = None
        if cap is None:
            print(f"[vision] No camera matching '{CAMERA_NAME_HINT}' -- using camera {CAMERA_INDEX}.")
    if cap is None:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX}.")
    if CAMERA_RESOLUTION:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    return cap


class Vision:
    def __init__(self):
        self.result = VisionResult()   # replaced whole each frame (atomic swap, no lock needed)
        self.status = "starting..."
        self.camera_mirrored = CAMERA_MIRRORED
        self._stop = threading.Event()
        self._smoothed = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def close(self):
        self._stop.set()

    def _run(self):
        try:
            cap = open_camera()
            landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ))
        except Exception as exc:
            self.status = f"error: {exc}"
            print(f"[vision] {self.status}")
            return
        tag_detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(APRILTAG_DICTIONARY),
                                           aruco.DetectorParameters())
        self.status = "running"
        last_ts = 0
        fps_t, fps_n, fps = time.monotonic(), 0, 0.0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status = "camera read failed"
                    time.sleep(0.05)
                    continue
                self.status = "running"
                now = time.monotonic()
                if self.camera_mirrored:
                    frame = cv2.flip(frame, 1)   # undo the camera's mirroring first
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(last_ts + 1, int(now * 1000))
                last_ts = ts
                pose = landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ts)

                landmarks, wrist_uv, paddle = None, None, None
                if pose.pose_landmarks:
                    lm = pose.pose_landmarks[0]
                    landmarks = {i: (1 - lm[i].x, lm[i].y) for i in range(len(lm))
                                 if (lm[i].visibility or 0) > MIN_VISIBILITY}
                    paddle = self._paddle_from_pose(lm, w, h)
                    if paddle is not None:
                        wrist_uv = (1 - lm[PADDLE_WRIST].x, lm[PADDLE_WRIST].y)

                tag_id, tag_corners = None, None
                corners, ids, _ = tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if ids is not None and len(ids) > 0:
                    tag_id = int(ids.ravel()[0])
                    tag_corners = [(1 - x / w, y / h) for x, y in corners[0][0]]

                fps_n += 1
                if now - fps_t >= 1.0:
                    fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

                display = cv2.resize(cv2.flip(rgb, 1), DISPLAY_SIZE)
                self.result = VisionResult(display, landmarks, wrist_uv, paddle,
                                           tag_id, tag_corners, fps, now)
        finally:
            cap.release()
            landmarker.close()

    def _paddle_from_pose(self, lm, w, h):
        """Map the right wrist to a paddle (x, y) in table meters, smoothed."""
        def px(i):
            return lm[i].x * w, lm[i].y * h

        def visible(i):
            return (lm[i].visibility or 0) > MIN_VISIBILITY

        if not (visible(L_SHOULDER) and visible(R_SHOULDER) and visible(PADDLE_WRIST)):
            self._smoothed = None
            return None
        (lsx, lsy), (rsx, rsy) = px(L_SHOULDER), px(R_SHOULDER)
        wx, wy = px(PADDLE_WRIST)
        mid_x, sh_y = (lsx + rsx) / 2, (lsy + rsy) / 2
        shoulder_w = max(20.0, abs(lsx - rsx))
        if visible(L_HIP) and visible(R_HIP):
            hip_y = (px(L_HIP)[1] + px(R_HIP)[1]) / 2
        else:
            hip_y = sh_y + TORSO_PER_SHOULDER_W * shoulder_w
        torso = max(20.0, hip_y - sh_y)

        # Un-mirrored image: my right side is the image's left, so right of me = smaller x.
        side = (mid_x - wx) / shoulder_w
        x = clamp(side / SHOULDER_WIDTHS_TO_EDGE * TABLE_HALF_W, -1.0, 1.0)
        y = clamp((hip_y - wy) / torso, -0.3, 1.6) * PADDLE_Y_AT_SHOULDER

        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            sx, sy = self._smoothed
            self._smoothed = (sx + SMOOTHING * (x - sx), sy + SMOOTHING * (y - sy))
        return self._smoothed
