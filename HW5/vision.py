"""Overhead camera: finds the ring, our robot (FRONT + BACK AprilTags) and the opponent.

Everything is reported in ring units (center (0,0), blue tape at radius 1.0,
x = image right, y = image up), so the rest of the code never sees pixels.

Ring (`c` in sumo.py): the most ring-shaped bluish thing in the frame is the tape;
an ellipse fitted to it gives a first floor map. The empty-ring frame from the
same moment is kept as the background for opponent detection. (`m` lets you
click points on the tape instead if the automatic fit fails.)

Perspective (`p`): the camera sits to the side (on a chair), so the floor is seen
at an angle. With the robot in the ring, its tags are watched for a couple of
seconds, the floor map is refined to a full homography, and the camera's
position is solved (perspective.py). Then the tags (on top of the robot) and the
opponent's body are corrected for their height above the floor.

Our robot: robot_tags.json (from make_tags.py) says which tag ID is the front
and which is the back. Both seen: heading = back -> front, center between them.
One seen: that tag's own orientation and its known offset from the center.

Opponent: the biggest blob that differs from the empty-ring background, inside
the ring, after cutting out our own robot.
"""

import json
import math
import threading
import time
from dataclasses import dataclass

import cv2
import cv2.aruco as aruco
import numpy as np

import config
from opponent import OpponentDetector
from perspective import solve_camera, solve_floor_homography



# ---------------- camera ----------------

def list_cameras():
    from cv2_enumerate_cameras import enumerate_cameras
    for cam in enumerate_cameras(cv2.CAP_MSMF):
        print(f"  [{cam.index}] {cam.name}")


def open_camera():
    """Open the camera matching CAMERA_NAME_HINT (the phone), falling back to CAMERA_INDEX. From HW4."""
    from cv2_enumerate_cameras import enumerate_cameras
    if config.CAMERA_NAME_HINT:
        for cam in enumerate_cameras(cv2.CAP_MSMF):
            if config.CAMERA_NAME_HINT.lower() in cam.name.lower():
                cap = cv2.VideoCapture(cam.index, cam.backend)
                if cap.isOpened():
                    print(f"[vision] using camera [{cam.index}] {cam.name}")
                    return cap
                print(f"[vision] found '{cam.name}' but couldn't open it.")
        print(f"[vision] no camera matching '{config.CAMERA_NAME_HINT}', falling back to camera {config.CAMERA_INDEX}.")
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}.")
    return cap


# ---------------- ring geometry ----------------

class Ring:
    """Map between image pixels and ring units on the FLOOR (a homography H: ring -> pixels).

    `c` gives a first guess from the tape ellipse alone (affine: right for a camera
    straight overhead). `p` refines it to full perspective with the robot's tags
    (perspective.py) and also finds where the camera is, which lets
    `true_from_apparent` correct things that stand above the floor.
    """

    tape_hsv = None        # median OpenCV HSV of the tape found by auto() (for the record)

    def __init__(self, H, ellipse=None, image_size=None):
        self.H = np.asarray(H, float) / H[2][2]
        self.Hinv = np.linalg.inv(self.H)
        self.ellipse = ellipse
        self.image_size = image_size     # (w, h) of the frames it was calibrated on
        self.tape_points = None          # Nx2 pixels on the tape (for the perspective refinement)
        self.camera = None               # {"f", "C": [x, y, height]} in ring units, once `p` has run
        self.perspective = None          # fit info from `p`
        self.center = self.to_image([[0.0, 0.0]])[0]
        a = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        rim = self.to_image(np.stack([np.cos(a), np.sin(a)], 1))
        self.radius_px = float(np.mean(np.linalg.norm(rim - self.center, axis=1)))

    @staticmethod
    def from_ellipse(ellipse, image_size=None):
        """Affine first guess: the ellipse -> the unit circle."""
        (cx, cy), (w, h), ang = ellipse
        a, b = w / 2.0, h / 2.0
        t = math.radians(ang)
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        S = np.diag([1.0 / a, 1.0 / b])
        flip = np.diag([1.0, -1.0])                     # image y is down, ring y is up
        Minv = np.linalg.inv(flip @ R @ S @ R.T)         # ring units -> pixels (relative to center)
        H = np.array([[Minv[0, 0], Minv[0, 1], cx], [Minv[1, 0], Minv[1, 1], cy], [0, 0, 1.0]])
        ell = ((float(cx), float(cy)), (float(w), float(h)), float(ang))
        return Ring(H, ell, image_size)

    def to_ring(self, pts):
        """Pixels -> ring units, for points ON THE FLOOR."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        q = np.column_stack([pts, np.ones(len(pts))]) @ self.Hinv.T
        return q[:, :2] / q[:, 2:3]

    def to_image(self, pts):
        """Ring units (on the floor) -> pixels."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        q = np.column_stack([pts, np.ones(len(pts))]) @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def _height_factor(self, height):
        if self.camera is None or height <= 0:
            return None
        hc = self.camera["C"][2]
        return (hc - height) / hc if hc > height * 1.05 else None

    def true_from_apparent(self, pts, height):
        """A point `height` (ring units) above the floor, seen through to_ring(), is really here."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        k = self._height_factor(height)
        if k is None:
            return pts
        F = np.array(self.camera["C"][:2])
        return F + (pts - F) * k

    def apparent_from_true(self, pts, height):
        """Where a point `height` above the floor at ring position `pts` shows up in floor coordinates."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        k = self._height_factor(height)
        if k is None:
            return pts
        F = np.array(self.camera["C"][:2])
        return F + (pts - F) / k

    def to_dict(self):
        return {"H": self.H.tolist(), "ellipse": self.ellipse, "image_size": self.image_size,
                "tape_hsv": self.tape_hsv, "camera": self.camera, "perspective": self.perspective,
                "tape_points": None if self.tape_points is None else np.round(self.tape_points, 1).tolist()}

    @staticmethod
    def from_dict(d):
        if "H" in d:
            ring = Ring(np.array(d["H"]), d.get("ellipse"), d.get("image_size"))
        else:   # older file: ellipse only
            e = d["ellipse"]
            ring = Ring.from_ellipse(((e[0][0], e[0][1]), (e[1][0], e[1][1]), e[2]))
        ring.tape_hsv = d.get("tape_hsv")
        ring.camera = d.get("camera")
        ring.perspective = d.get("perspective")
        if d.get("tape_points"):
            ring.tape_points = np.array(d["tape_points"], float)
        return ring

    def describe(self):
        if self.camera is None:
            return "perspective not corrected (put the robot in the ring and press p)"
        cx, cy, hz = self.camera["C"]
        return f"camera {hz:.2f} ring-radii up, {math.hypot(cx, cy):.2f} out, f {self.camera['f']:.0f}px"

    @staticmethod
    def from_points(points, image_size=None):
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        if len(pts) < 5:
            return None
        ring = Ring.from_ellipse(cv2.fitEllipse(pts), image_size)
        ring.tape_points = np.asarray(points, float).reshape(-1, 2)
        return ring

    @staticmethod
    def auto(frame):
        """Find the most ring-shaped bluish thing in `frame` and fit the ring to it.

        The tape can be anything from light (sky) blue to deep blue painter's tape,
        and lighting changes how it looks, so no single color threshold is used.
        Every combination of a hue window and a saturation cutoff (measured above
        the gray floor's own saturation) makes a candidate mask; each candidate's
        blobs get an ellipse fit, and the winner is the fit that best looks like a
        ring of tape: blue pixels found all the way round the ellipse, and few
        blue pixels off it. Ties go to the bluer candidate. Returns (Ring or None, message).
        """
        h0, w0 = frame.shape[:2]
        s = min(1.0, 640 / w0)                          # search on a smaller copy
        small = cv2.resize(frame, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
        floor_s = float(np.median(S))                   # the gray floor dominates the frame
        h, w = small.shape[:2]

        best = None                                     # (score, ring, mask, label)
        for lo, hi in config.TAPE_HUE_WINDOWS:
            hue_ok = (H >= lo) & (H <= hi) & (V >= config.TAPE_MIN_VALUE)
            for ds in config.TAPE_SAT_STEPS:
                mask = (hue_ok & (S >= floor_s + ds)).astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                # Blobs = connected tape pixels. Every pixel is used (not just outlines), so the
                # fit lands on the middle of the tape, not its outer edge.
                n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                order = [i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
                         if stats[i, cv2.CC_STAT_AREA] >= 40][:4]
                if not order:
                    continue
                blobs = {i: np.column_stack(np.nonzero(labels == i)[::-1]) for i in order}
                # Try all the big blobs together (a ring broken into pieces) and each of the top ones alone.
                biggest = stats[order[0], cv2.CC_STAT_AREA]
                groups = [[blobs[i] for i in order if stats[i, cv2.CC_STAT_AREA] > 0.1 * biggest]] +                          [[blobs[i]] for i in order[:3]]
                for group in groups:
                    ring = Ring._score_fit(group, mask, w, h)
                    if ring is None:
                        continue
                    score, ring = ring
                    blueness = 1 - abs((lo + hi) / 2 - 110) / 60      # deep blue (H~110) slightly preferred
                    score += 0.02 * blueness
                    if best is None or score > best[0]:
                        best = (score, ring, mask, f"hue {lo}-{hi}, sat > floor+{ds}")
        if best is None or best[0] < config.TAPE_MIN_SCORE:
            got = f" (best fit {best[0]:.2f})" if best else ""
            return None, f"no ring of blue tape found{got}; press m to click points on the tape"

        score, ring, mask, label = best
        # Scale the ellipse back to full resolution.
        (cx, cy), (ew, eh), ang = ring.ellipse
        full = Ring.from_ellipse(((cx / s, cy / s), (ew / s, eh / s), ang), (w0, h0))
        # Record what the tape looks like (median HSV near the ellipse) and keep some tape
        # pixels for the perspective refinement (`p`).
        ys, xs = np.nonzero(mask)
        r = np.linalg.norm(ring.to_ring(np.stack([xs, ys], 1)), axis=1)
        near = np.abs(r - 1) < 0.1
        if near.any():
            idx = np.nonzero(near)[0]
            idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 600)).astype(int)]
            full.tape_points = np.stack([xs[idx], ys[idx]], 1) / s
            full.tape_hsv = [int(np.median(H[ys[near], xs[near]])), int(np.median(S[ys[near], xs[near]])),
                             int(np.median(V[ys[near], xs[near]]))]
        tape = f", tape HSV {full.tape_hsv}" if full.tape_hsv else ""
        print(f"[vision] ring fit score {score:.2f} using {label}{tape}")
        return full, f"ring found ({ew / s:.0f} x {eh / s:.0f} px, fit {score:.2f}{tape})"

    @staticmethod
    def _score_fit(blobs, mask, w, h):
        """Fit an ellipse to the blobs' pixels; score 0..1 for how much it looks like a tape ring."""
        pts = np.vstack(blobs).astype(np.float32)
        if len(pts) < 40:
            return None
        if len(pts) > 5000:
            pts = pts[np.linspace(0, len(pts) - 1, 5000).astype(int)]
        ring = Ring.from_ellipse(cv2.fitEllipse(pts.reshape(-1, 1, 2)))
        (cx, cy), (ew, eh), _ = ring.ellipse
        if not (0.15 * min(w, h) < min(ew, eh) and max(ew, eh) < 1.5 * max(w, h)) or min(ew, eh) < 0.4 * max(ew, eh):
            return None
        # Coverage: how much of the ellipse has tape on it.
        a = np.linspace(0, 2 * math.pi, 180, endpoint=False)
        around = ring.to_image(np.stack([np.cos(a), np.sin(a)], 1)).round().astype(int)
        inside = (around[:, 0] >= 0) & (around[:, 0] < w) & (around[:, 1] >= 0) & (around[:, 1] < h)
        if inside.mean() < 0.5:
            return None
        band = max(3, int(0.04 * ring.radius_px))
        grown = cv2.dilate(mask, np.ones((band, band), np.uint8))
        hits = grown[around[inside, 1], around[inside, 0]] > 0
        coverage = hits.sum() / len(a)          # parts of the ring off-frame count as missing
        # Precision: how much of the blobs lies on the ellipse.
        r = np.linalg.norm(ring.to_ring(pts), axis=1)
        precision = float(np.mean(np.abs(r - 1) < 0.08))
        return coverage * precision, ring


def footprint(x, y, theta, length, width, scale=1.0):
    """Corners of the robot's rectangle in ring units (front-left, front-right, back-right, back-left)."""
    c, s = math.cos(theta), math.sin(theta)
    hl, hw = length * scale / 2, width * scale / 2
    out = []
    for fx, fy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        out.append((x + fx * c - fy * s, y + fx * s + fy * c))
    return np.array(out)


def fraction_outside(x, y, theta, length, width, radius=1.0):
    """How much of the robot's footprint lies past `radius` (sampled on a 7 x 5 grid)."""
    c, s = math.cos(theta), math.sin(theta)
    n = 0
    total = 0
    for i in range(7):
        fx = (i / 6 - 0.5) * length
        for j in range(5):
            fy = (j / 4 - 0.5) * width
            px, py = x + fx * c - fy * s, y + fx * s + fy * c
            n += (px * px + py * py) > radius * radius
            total += 1
    return n / total


# ---------------- robot geometry ----------------

class Geometry:
    """Robot and tag dimensions, in ring units (converted from robot_tags.json cm)."""

    def __init__(self):
        try:
            tags = json.loads(config.ROBOT_TAGS_PATH.read_text())
        except (OSError, ValueError):
            raise SystemExit(f"{config.ROBOT_TAGS_PATH.name} not found. Run: python HW5/make_tags.py")
        self.tags = tags
        self.front_id = int(tags["front_id"])
        self.back_id = int(tags["back_id"])
        self.ring_diameter_cm = config.RING_DIAMETER_CM or config.RING_DIAMETER_FALLBACK_CM
        self._estimated = config.RING_DIAMETER_CM is not None
        self._n_est = 0

    def cm(self, value_cm):
        return value_cm / (self.ring_diameter_cm / 2.0)

    @property
    def length(self): return self.cm(self.tags["robot_length_cm"])
    @property
    def width(self): return self.cm(self.tags["robot_width_cm"])
    @property
    def radius(self): return 0.5 * math.hypot(self.length, self.width) * 0.85
    def tag_height_cm_of(self, tag_id):
        """Height of a tag above the floor (cm): per tag if robot_tags.json has it."""
        t = self.tags
        common = t.get("tag_height_cm", config.DEFAULT_TAG_HEIGHT_CM)
        if tag_id == self.front_id:
            return t.get("front_tag_height_cm", common)
        if tag_id == self.back_id:
            return t.get("back_tag_height_cm", common)
        return common

    def tag_height_of(self, tag_id):
        return self.cm(self.tag_height_cm_of(tag_id))

    @property
    def tag_height(self):
        """The higher of the two tags (the robot's top, for masking its box)."""
        return max(self.tag_height_of(self.front_id), self.tag_height_of(self.back_id))
    @property
    def front_offset(self): return self.cm(self.tags["front_offset_cm"])
    @property
    def back_offset(self): return self.cm(self.tags["back_offset_cm"])

    def observe_tag_side(self, side_ring):
        """Running estimate of the ring size from a tag's measured side (ring units)."""
        if config.RING_DIAMETER_CM is not None or side_ring <= 1e-4:
            return
        est = 2.0 * self.tags["tag_size_cm"] / side_ring
        if not 20 < est < 400:
            return
        self._n_est = min(self._n_est + 1, 200)
        self.ring_diameter_cm += (est - self.ring_diameter_cm) / self._n_est
        self._estimated = True


# ---------------- measurements ----------------

@dataclass
class OurPose:
    x: float
    y: float
    theta: float
    n_tags: int                 # 2 = both tags (best), 1 = one tag


@dataclass
class Measurement:
    t: float                    # when the scene actually happened (capture time - latency)
    our: OurPose = None
    our_out_fraction: float = None
    opp: tuple = None           # (x, y, radius) in ring units
    tag_corners: dict = None    # id -> 4x2 image points (for the overlay)


class Vision:
    def __init__(self, geometry):
        self.geom = geometry
        self.status = "starting"
        self.ring = None
        self.opponent = None        # OpponentDetector (needs the ring + an empty-ring background)
        self.opp_hint = None        # (x, y) where fusion expects the opponent (set by sumo.py)
        self._bg_frames = None      # collecting empty-ring frames for the background
        self.events = []            # (kind, ok, message) for each finished calibration step (setup walkthrough)
        self.fps = 0.0
        self._lock = threading.Lock()
        self._frame = None
        self._measurement = None
        self._seq = 0
        self._stop = threading.Event()
        self._calibrate_request = None   # "auto", "background", "perspective", or clicked image points
        self._collect_until = None       # collecting tag observations for `p` until then
        self._tag_obs = []               # (tag id, 4x2 corners) views of our tags (for `p`)
        self._last_our = None            # (t, OurPose) for masking when the tags blink out
        self._load()

    # ---------- persistence ----------

    def _load(self):
        try:
            data = json.loads(config.RING_CALIBRATION_PATH.read_text())
            self.ring = Ring.from_dict(data)
            print(f"[vision] loaded ring calibration ({self.ring.describe()}).")
            self.opponent = OpponentDetector(self.ring, self.geom)
            if config.RING_BACKGROUND_PATH.exists():
                self.opponent.load(config.RING_BACKGROUND_PATH)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save(self):
        config.RING_CALIBRATION_PATH.write_text(json.dumps(self.ring.to_dict(), indent=2))
        if self.opponent is not None and self.opponent.full is not None:
            self.opponent.save(config.RING_BACKGROUND_PATH)

    # ---------- thread API ----------

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="vision").start()
        return self

    def stop(self):
        self._stop.set()

    def reset_calibration(self):
        """Forget the ring, perspective and background (the setup walkthrough's "start fresh")."""
        self.ring = None
        self.opponent = None
        self._tag_obs = []
        self._bg_frames = None
        self._collect_until = None
        self._last_our = None
        self.status = "ring not calibrated (press c)"

    def request_calibration(self, what="auto"):
        """'auto' (ring + background), 'background', 'perspective' (watch the tags, then
        correct for the camera's angle), or a list of clicked image points."""
        self._calibrate_request = what

    def latest(self):
        """(sequence number, Measurement, frame). The sequence number changes with each new frame."""
        with self._lock:
            return self._seq, self._measurement, self._frame

    # ---------- main loop ----------

    def _run(self):
        try:
            cap = open_camera()
        except Exception as exc:
            self.status = f"camera error: {exc}"
            print(f"[vision] {self.status}")
            return
        detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11),
                                       aruco.DetectorParameters())
        self.status = "running" if self.ring else "ring not calibrated (press c)"
        fps_t, fps_n = time.monotonic(), 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            t_cap = time.monotonic()
            if not ok:
                self.status = "camera read failed"
                time.sleep(0.05)
                continue
            fps_n += 1
            if t_cap - fps_t >= 1.0:
                self.fps, fps_n, fps_t = fps_n / (t_cap - fps_t), 0, t_cap

            if self._calibrate_request is not None:
                self._do_calibration(frame, self._calibrate_request)
                self._calibrate_request = None
            if self._bg_frames is not None:
                self._collect_background(frame)

            m = Measurement(t=t_cap - config.CAMERA_LATENCY_S)
            if self.ring is not None:
                self._measure(frame, detector, m)
                if self._collect_until is not None:
                    self._collect_tags(m, frame, t_cap)
            with self._lock:
                self._frame, self._measurement = frame, m
                self._seq += 1
        cap.release()

    def _do_calibration(self, frame, what):
        if what == "perspective":
            if self.ring is None or self.ring.tape_points is None:
                self.status = "calibrate the ring first (c)"
                self.events.append(("perspective", False, self.status))
                return
            self._collect_until = time.monotonic() + config.PERSPECTIVE_COLLECT_S
            self.status = "perspective: watching the robot's tags..."
            return
        if what == "background":
            if self.ring is None:
                self.status = "calibrate the ring first (c)"
                return
            self._bg_frames = []
            self.status = "recording the empty ring (keep it empty)..."
            return
        h, w = frame.shape[:2]
        if what == "auto":
            ring, msg = Ring.auto(frame)
        else:
            ring = Ring.from_points(what, (w, h))
            msg = "ring set from clicked points" if ring else "need at least 5 clicked points"
        print(f"[vision] {msg}")
        self.events.append(("ring", ring is not None, msg))
        if ring is None:
            self.status = msg
            return
        self.ring = ring
        self._tag_obs = []               # new ring = start the perspective fit over
        self.opponent = OpponentDetector(ring, self.geom)
        self._bg_frames = []             # record the empty ring over the next frames
        self._save()
        self.status = msg + "; recording the empty ring..."

    def _collect_background(self, frame):
        self._bg_frames.append(frame.copy())
        if len(self._bg_frames) < config.BG_FRAMES:
            return
        frames, self._bg_frames = self._bg_frames, None
        if self.opponent is None:
            self.opponent = OpponentDetector(self.ring, self.geom)
        self.opponent.calibrate(frames)
        self._save()
        self.events.append(("background", True, "empty ring recorded"))
        self.status = ("empty ring recorded. Now put the robot in the ring and press p"
                       if self.ring.camera is None else "empty ring recorded")

    def _collect_tags(self, m, frame, now):
        g = self.geom
        for tid in (g.front_id, g.back_id):
            if m.tag_corners and tid in m.tag_corners:
                self._tag_obs.append((tid, m.tag_corners[tid].copy()))
        self._tag_obs = self._tag_obs[-config.PERSPECTIVE_MAX_OBS:]
        if now < self._collect_until:
            return
        self._collect_until = None
        if len(self._tag_obs) < 4:
            self.status = "perspective: the robot's tags weren't seen; put it in the ring and press p"
            self.events.append(("perspective", False, "the robot's tags weren't seen. Both tags must face "
                                                      "the camera; ENTER = try again"))
            return
        h, w = frame.shape[:2]
        self.status = self.solve_perspective((w, h))
        self.events.append(("perspective", self.status.startswith("perspective ok"), self.status))

    def solve_perspective(self, image_size):
        """Fit the full floor homography + camera position from the tape and the collected tags."""
        ring = self.ring
        # Always start from the tape-only (affine) guess, so repeated `p` presses don't drift.
        start = Ring.from_ellipse(ring.ellipse, image_size) if ring.ellipse else ring
        # Stage 1: free homography (tape on the circle, tags square). Stage 2: a real camera.
        ids = [tid for tid, _ in self._tag_obs]
        views = [c for _, c in self._tag_obs]
        H8, info = solve_floor_homography(start.H, ring.tape_points, views, ids)
        H, cam, info2 = solve_camera(H8, ring.tape_points, views, image_size, ids)
        info.update(info2)
        info["n_tags"] = len(self._tag_obs)
        if cam is None:
            print(f"[vision] perspective fit didn't give a sensible camera: {info}")
            return ("perspective: couldn't solve the camera. Move the robot to 2-3 spots "
                    "(press p at each), keep both tags in view")
        new = Ring(H, ring.ellipse, list(image_size))
        new.tape_hsv, new.tape_points = ring.tape_hsv, ring.tape_points
        new.camera, new.perspective = cam, info
        self.ring = new
        if self.opponent is not None:
            self.opponent.set_ring(new)
        self._save()
        hz_cm = cam["C"][2] * self.geom.ring_diameter_cm / 2
        print(f"[vision] perspective solved: {new.describe()}, camera ~{hz_cm:.0f} cm high, "
              f"tape rms {info['tape_rms']:.3f}, tag shape rms {info['tag_shape_rms']:.3f}, "
              f"{info['n_tags']} tag views")
        return (f"perspective ok: camera ~{hz_cm:.0f} cm up, tape fit {info['tape_rms']:.3f}. "
                f"Press p again at other spots to refine")

    # ---------- measuring ----------

    def _measure(self, frame, detector, m):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        tags = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                tags[int(i)] = c.reshape(4, 2)
        m.tag_corners = tags
        m.our = self._our_pose(tags)
        g = self.geom
        if m.our is not None:
            self._last_our = (m.t, m.our)
            m.our_out_fraction = fraction_outside(m.our.x, m.our.y, m.our.theta, g.length, g.width)
        # An opponent wearing any other AprilTag is found from its tag.
        others = [tid for tid in tags if tid not in (g.front_id, g.back_id)]
        if others:
            c = self.ring.true_from_apparent(self.ring.to_ring(tags[others[0]]), g.cm(config.OPP_TAG_HEIGHT_CM))
            ctr = c.mean(axis=0)
            m.opp = (float(ctr[0]), float(ctr[1]), g.cm(config.OPP_SIZE_CM) * 0.55)
        elif self.opponent is not None and self.opponent.bg is not None:
            recent = self._last_our is not None and m.t - self._last_our[0] < 0.5
            o = self._last_our[1] if recent else None
            m.opp = self.opponent.detect(frame, (o.x, o.y, o.theta) if o else None, self.opp_hint)

    def _tag_in_ring(self, corners_px, tag_id):
        """Center and forward direction (tag's top edge) of one tag, in ring units."""
        # TL, TR, BR, BL in the tag's own frame; the tag is on top of the robot, so undo its height.
        c = self.ring.true_from_apparent(self.ring.to_ring(corners_px), self.geom.tag_height_of(tag_id))
        center = c.mean(axis=0)
        fwd = (c[0] + c[1]) / 2 - (c[3] + c[2]) / 2
        side = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])
        self.geom.observe_tag_side(side)
        return center, math.atan2(fwd[1], fwd[0])

    def _our_pose(self, tags):
        g = self.geom
        front = self._tag_in_ring(tags[g.front_id], g.front_id) if g.front_id in tags else None
        back = self._tag_in_ring(tags[g.back_id], g.back_id) if g.back_id in tags else None
        if front and back:
            d = front[0] - back[0]
            theta = math.atan2(d[1], d[0])
            span = g.front_offset + g.back_offset
            k = g.back_offset / span if span > 1e-6 else 0.5
            center = back[0] + d * k
            return OurPose(float(center[0]), float(center[1]), theta, 2)
        if front:
            th = front[1]
            c = front[0] - g.front_offset * np.array([math.cos(th), math.sin(th)])
            return OurPose(float(c[0]), float(c[1]), th, 1)
        if back:
            th = back[1]
            c = back[0] + g.back_offset * np.array([math.cos(th), math.sin(th)])
            return OurPose(float(c[0]), float(c[1]), th, 1)
        return None
