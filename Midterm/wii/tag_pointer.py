"""Wii pointer from an AprilTag taped to the FRONT of the Double Motor.

The camera (by the screen) sees the tag whenever the remote is aimed at the
screen, like the real Wii Remote's camera sees the sensor bar. The pointer
follows WHERE THE TAG IS in the camera frame: move the remote and the
pointer moves with it.

Optionally (AIM_WEIGHT > 0) it also follows WHICH WAY THE TAG FACES, from
its 3D pose (cv2.solvePnP with the square-marker solver): turn the nose
right/up and the pointer goes right/up. That's off by default; see AIM_WEIGHT.

The pointer is measured from a "center" you can set: hold the remote where
you want the middle of the screen to be and press P in the LEGO window.

Tag corners come from vision.py (detected on the un-mirrored frame, handed
over mirrored and normalized). They are un-mirrored again here so the pose
math sees a real camera image.
"""

import math
import time

import cv2
import numpy as np

POINTER_TAG_ID = None        # None = any tag; or the ID of the tag on the motor
POSITION_WEIGHT = 1.0
# The tag's facing direction is OFF by default. Tested on synthetic tags with realistic
# noise, a small tag's pose sometimes flips to its mirror image (errors up to ~40 deg, and
# at 2 m upward vs. flat is barely distinguishable), so it made the pointer jump. Its
# position is reliable. Try e.g. 0.5 with a big tag held close to the camera.
AIM_WEIGHT = 0.0
POSITION_RANGE = 0.22        # tag this far left/right (fraction of frame WIDTH) from center = side edge
POSITION_RANGE_Y = 0.25      # tag this far up/down (fraction of frame HEIGHT) from center = top/bottom edge.
                             # Was effectively 0.39 (it shared the width scale), so the bottom needed the
                             # tag almost at the frame's edge, where a lowered hand drops out of view.
# Flip a pointer axis. Both were backwards on the real setup (2026-10-09), so both are on;
# set either back to False if that axis ever moves the wrong way.
INVERT_X = True
INVERT_Y = True
AIM_RANGE_DEG = 25.0         # nose turned this far from center = pointer at the edge
SMOOTHING = 0.45             # EMA weight of the newest reading
HOLD_LOST_S = 0.4            # keep the last pointer this long after losing the tag
CAMERA_HFOV_DEG = 65.0       # rough phone-camera field of view (only affects the aim estimate)
FRAME_ASPECT = 16 / 9
AMBIGUOUS_ERROR_RATIO = 3.0  # two pose fits within this error ratio = can't tell them apart

# Marker model in its own frame (x right, y up, z out of the tag toward the viewer),
# in ArUco corner order: top-left, top-right, bottom-right, bottom-left.
_OBJ = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float64)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class TagPointer:
    def __init__(self):
        self.pointer = None          # (x, y) in -1..1, +y up
        self.raw = None              # unsmoothed, uncentered (x, y)
        self.aim_deg = None          # (yaw, pitch) of the remote's nose, + = right / up
        self.tag_id = None
        self.center = (0.0, 0.0)     # raw reading that maps to the screen center (P sets it)
        self._smoothed = None
        self._last_seen = -1e9

    def recenter(self):
        """Treat the current reading as 'aimed at the middle of the screen'."""
        if self.raw is not None:
            self.center = self.raw
            self._smoothed = None

    @staticmethod
    def _aim_angles(img, K):
        """(yaw, pitch) of the remote's nose in degrees (+ = my right / up), or None.

        A small, nearly head-on square has two poses that fit the image almost
        equally well (mirror images about the line of sight). Take the better fit
        when it's clearly better; otherwise the one closer to facing the camera,
        which is the smaller (safer) aim.
        """
        try:
            n, rvecs, _, errs = cv2.solvePnPGeneric(_OBJ, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if not n:
            return None
        options = []
        for rvec, err in zip(rvecs, np.ravel(errs) if errs is not None else [0.0] * n):
            R, _ = cv2.Rodrigues(rvec)
            nx, ny, nz = R[:, 2]               # tag normal in camera coords (x right, y down, z ahead)
            if nz > 0:                          # a visible tag faces the camera
                nx, ny, nz = -nx, -ny, -nz
            # Nose turned to MY right = toward the camera's left (-x); nose up = normal up (-y).
            yaw = math.degrees(math.atan2(-nx, -nz))
            pitch = math.degrees(math.atan2(-ny, -nz))
            options.append((float(err), yaw, pitch))
        options.sort()
        if len(options) > 1 and options[1][0] < AMBIGUOUS_ERROR_RATIO * max(options[0][0], 1e-9):
            options.sort(key=lambda o: math.hypot(o[1], o[2]))
        return options[0][1], options[0][2]

    @property
    def visible(self):
        return self.tag_id is not None

    def update(self, vis, now=None):
        now = time.monotonic() if now is None else now
        corners = None
        if vis is not None and vis.tag_corners and (POINTER_TAG_ID is None or vis.tag_id == POINTER_TAG_ID):
            corners = vis.tag_corners
            self.tag_id = vis.tag_id
        else:
            self.tag_id = None

        if corners is None:
            if now - self._last_seen > HOLD_LOST_S:
                self.pointer, self.raw, self.aim_deg, self._smoothed = None, None, None, None
            return
        self._last_seen = now

        # Un-mirror into a real camera image, in pixels of a 1000-wide frame.
        w, h = 1000.0, 1000.0 / FRAME_ASPECT
        img = np.array([[(1 - u) * w, v * h] for u, v in corners], dtype=np.float64)

        # Position: tag center, mirrored back so + = toward my right on screen.
        cu = sum(u for u, _ in corners) / 4      # mirrored u: my right is +
        cv_ = sum(v for _, v in corners) / 4
        pos = ((cu - 0.5) / POSITION_RANGE, -(cv_ - 0.5) / POSITION_RANGE_Y)

        # Aim: the tag's facing direction from its pose.
        aim = (0.0, 0.0)
        f = (w / 2) / math.tan(math.radians(CAMERA_HFOV_DEG) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        angles = self._aim_angles(img, K)
        if angles is not None:
            self.aim_deg = angles
            aim = (angles[0] / AIM_RANGE_DEG, angles[1] / AIM_RANGE_DEG)

        self.raw = ((-1 if INVERT_X else 1) * (POSITION_WEIGHT * pos[0] + AIM_WEIGHT * aim[0]),
                    (-1 if INVERT_Y else 1) * (POSITION_WEIGHT * pos[1] + AIM_WEIGHT * aim[1]))
        x = clamp(self.raw[0] - self.center[0], -1, 1)
        y = clamp(self.raw[1] - self.center[1], -1, 1)
        if self._smoothed is None:
            self._smoothed = (x, y)
        else:
            sx, sy = self._smoothed
            self._smoothed = (sx + SMOOTHING * (x - sx), sy + SMOOTHING * (y - sy))
        self.pointer = self._smoothed
