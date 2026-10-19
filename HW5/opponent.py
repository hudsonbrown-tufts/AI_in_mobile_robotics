"""Finds the opponent: whatever in the ring doesn't look like the empty ring.

Fixes over the first version (which "wasn't doing a good job"):
  - Only the ring's part of the frame is processed, at up to PROC_MAX_W pixels wide,
    instead of the whole frame shrunk to 480 px (from a chair the ring is a small,
    flattened part of the picture, so the opponent was only a few pixels tall).
  - The empty-ring background is the average of several frames, with each pixel's
    own noise level, so the threshold adapts per pixel.
  - The phone's auto-exposure and white balance shift the whole picture; that shift
    is measured every frame and removed before comparing.
  - Shadows (darker, same color) are discounted.
  - A dark robot on a dark floor barely changes the color, but its edges (LEGO studs,
    beams) change the image's texture, so edge changes count too.
  - The background no longer slowly absorbs a still opponent (it only follows
    lighting through the per-frame exposure correction, and very slowly elsewhere).
  - Blobs are scored on size (how big an opponent should be on the floor) and on
    being where the tracker expects it, instead of just "the biggest blob".
  - Seen from the side, a robot's blob stretches away from the camera; its position
    is taken from the near edge of the blob (where it touches the floor) instead of
    the middle of the blob.
"""

import math

import cv2
import numpy as np

import config

PROC_MAX_W = 640


class OpponentDetector:
    """The empty-ring background is kept for the whole frame (Lab mean + per-pixel noise, and
    the mean picture for its edges), so it can be re-cut whenever the ring's floor map
    changes (e.g. after the perspective step `p`)."""

    def __init__(self, ring, geom):
        self.ring = ring
        self.geom = geom
        self.full = None               # whole-frame background: {"lab": HxWx3, "std": HxWx3, "bgr": HxWx3}
        self.bg = None                 # dict of float32 arrays in the ROI: L, a, b, sL, sa, sb, edge
        self.roi = None                # (x0, y0, x1, y1) in full-frame pixels
        self.scale = 1.0               # ROI pixels -> processing pixels
        self._mask_ring = None
        self._px_floor = None          # processing pixel -> ring coords (floor)
        self._hits = []                # recent frames' candidate positions (for confirmation)
        self.debug = None              # (fg mask, roi, scale, candidates) for the `o` overlay
        self._setup_roi()

    # ---------- geometry ----------

    def _setup_roi(self, frame_shape=None):
        ring = self.ring
        if frame_shape is not None:
            h, w = frame_shape[:2]
        elif ring.image_size:
            w, h = ring.image_size
        else:
            w, h = 1280, 720
        a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        R = config.OPP_SEARCH_RADIUS
        pts = ring.to_image(np.stack([R * np.cos(a), R * np.sin(a)], 1))
        # Tall things lean away from a side camera; leave room above the far side.
        tall = ring.to_image(ring.apparent_from_true(np.stack([R * np.cos(a), R * np.sin(a)], 1),
                                                     self.geom.cm(config.OPP_MAX_HEIGHT_CM)))
        allp = np.vstack([pts, tall])
        x0, y0 = np.floor(allp.min(0)).astype(int) - 4
        x1, y1 = np.ceil(allp.max(0)).astype(int) + 4
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        self.roi = (int(x0), int(y0), int(x1), int(y1))
        self.scale = min(1.0, PROC_MAX_W / max(1, x1 - x0))
        pw, ph = int(round((x1 - x0) * self.scale)), int(round((y1 - y0) * self.scale))
        ys, xs = np.mgrid[0:ph, 0:pw]
        full = np.stack([xs.ravel() / self.scale + x0, ys.ravel() / self.scale + y0], 1)
        floor = ring.to_ring(full)
        r = np.linalg.norm(floor, axis=1).reshape(ph, pw)
        # Search inside R, but skip the tape itself: exposure changes make the bright tape
        # "change" and it would join the opponent into one ring-shaped blob.
        self._mask_ring = ((r < R) & (np.abs(r - 1.0) > config.OPP_TAPE_BAND)).astype(np.uint8)
        self._px_floor = floor.reshape(ph, pw, 2)
        self._proc_size = (pw, ph)

    def _crop(self, frame):
        x0, y0, x1, y1 = self.roi
        crop = frame[y0:y1, x0:x1]
        if self.scale < 1.0:
            crop = cv2.resize(crop, self._proc_size, interpolation=cv2.INTER_AREA)
        elif crop.shape[1::-1] != self._proc_size:
            crop = cv2.resize(crop, self._proc_size)
        return crop

    @staticmethod
    def _lab_edges(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
        return lab, edge

    # ---------- background ----------

    def calibrate(self, frames):
        """Build the empty-ring background from a few frames (the ring must be empty)."""
        smooth = [cv2.GaussianBlur(f, (5, 5), 0) for f in frames]       # per-pixel camera noise down
        labs = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in smooth])
        std = labs.std(0) if len(frames) > 1 else np.zeros(labs.shape[1:], np.float32)
        self.full = {"lab": labs.mean(0), "std": std,
                     "bgr": np.mean(np.stack(smooth).astype(np.float32), 0).astype(np.uint8)}
        self.set_ring(self.ring, frames[0].shape)

    def set_ring(self, ring, frame_shape=None):
        """(Re)cut the background for this ring's floor map."""
        self.ring = ring
        if frame_shape is None and self.full is not None:
            frame_shape = self.full["bgr"].shape
        self._setup_roi(frame_shape)
        self._hits = []
        if self.full is None:
            self.bg = None
            return
        lab = self._crop(self.full["lab"])
        std = self._crop(self.full["std"])
        bgr = self._crop(self.full["bgr"])
        _, edge = self._lab_edges(bgr)
        self.bg = {"L": lab[..., 0].copy(), "a": lab[..., 1].copy(), "b": lab[..., 2].copy(),
                   "sL": std[..., 0], "sa": std[..., 1], "sb": std[..., 2], "edge": edge,
                   "bgr": bgr.astype(np.float32)}

    def save(self, path):
        np.savez_compressed(path, lab=self.full["lab"].astype(np.float16),
                            std=self.full["std"].astype(np.float16), bgr=self.full["bgr"])

    def load(self, path):
        d = np.load(path)
        self.full = {"lab": d["lab"].astype(np.float32), "std": d["std"].astype(np.float32), "bgr": d["bgr"]}
        self.set_ring(self.ring, self.full["bgr"].shape)
        return True

    # ---------- detection ----------

    def detect(self, frame, our_pose=None, predicted=None):
        """(x, y, radius) in ring units, or None. our_pose: (x, y, theta) to cut ourselves out."""
        if self.bg is None:
            return None
        img = cv2.GaussianBlur(self._crop(frame), (5, 5), 0)
        bg = self.bg
        inside = self._mask_ring > 0

        # Our robot, as a box from the floor to its top (the camera sees its sides).
        ours = np.zeros_like(self._mask_ring)
        if our_pose is not None:
            g = self.geom
            from vision import footprint
            poly = footprint(our_pose[0], our_pose[1], our_pose[2], g.length, g.width, config.OUR_MASK_SCALE)
            top = self.ring.apparent_from_true(poly, g.tag_height * 1.15)
            box = np.vstack([self.ring.to_image(poly), self.ring.to_image(top)])
            box = (box - np.array(self.roi[:2])) * self.scale
            cv2.fillPoly(ours, [cv2.convexHull(box.astype(np.int32))], 1)
            ours = cv2.dilate(ours, np.ones((5, 5), np.uint8))

        # Exposure / white balance: the phone rescales each color channel; undo that by matching
        # each channel's median over the ring (minus our robot) to the empty ring's.
        ref = inside & (ours == 0)
        if ref.sum() < 50:
            return None
        imgf = img.astype(np.float32)
        gains = np.array([np.median(imgf[..., c][ref]) / max(1.0, np.median(bg["bgr"][..., c][ref]))
                          for c in range(3)], np.float32)
        norm = np.clip(imgf / np.maximum(gains, 0.05), 0, 255).astype(np.uint8)
        lab, edge = self._lab_edges(norm)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        dL = L - bg["L"]
        dA = A - bg["a"]
        dB = B - bg["b"]
        zL = dL / (bg["sL"] + config.OPP_NOISE_L)
        zc = np.sqrt(dA ** 2 + dB ** 2) / (np.sqrt(bg["sa"] ** 2 + bg["sb"] ** 2) + config.OPP_NOISE_C)
        tex = (edge - bg["edge"]) / (bg["edge"] * 0.5 + config.OPP_NOISE_EDGE)
        # Soft shadows: a bit darker, same color, no new texture. Only mildly discounted, because a
        # dark gray robot on a dark gray floor looks the same way (its edges tell it apart).
        ratio = (L + 1.0) / (bg["L"] + 1.0)
        shadow = (ratio > 0.5) & (ratio < 0.92) & (np.abs(dA) < 4) & (np.abs(dB) < 4) & (tex < 1.0)
        zL = np.where(shadow, zL * config.OPP_SHADOW_DISCOUNT, zL)
        score = np.sqrt(zL ** 2 + zc ** 2)
        fg = (score > config.OPP_Z) | ((tex > config.OPP_EDGE_Z) & (score > config.OPP_Z * 0.4))
        fg = (fg & inside & (ours == 0)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Follow slow lighting drift only where nothing is going on.
        quiet = (score < config.OPP_Z * 0.3) & (cv2.dilate(fg, np.ones((15, 15), np.uint8)) == 0) & (ours == 0)
        k = config.BG_ADAPT_RATE
        for ch, val in (("L", L), ("a", A), ("b", B)):
            bg[ch][quiet] += k * (val[quiet] - bg[ch][quiet])

        best, cands = self._pick(fg, predicted)
        self.debug = (fg, self.roi, self.scale, cands)
        # Confirm a new opponent over a few frames; a tracked one is accepted at once.
        self._hits = (self._hits + [best])[-3:]
        if best is None:
            return None
        if predicted is None and sum(h is not None for h in self._hits) < 2:
            return None
        return best

    def _pick(self, fg, predicted):
        """Score blobs (and groups of nearby blobs) as the opponent; return the best one."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        expect = self.geom.cm(config.OPP_SIZE_CM)
        blobs = []                                            # (stats, floor points, floor center)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 12:
                continue
            ys, xs = np.nonzero(labels == i)
            pts = self._px_floor[ys, xs]                      # blob pixels on the floor (apparent)
            blobs.append((stats[i], pts, pts.mean(0)))
        # A robot on the tape is split in two by the skipped tape band: also try each blob
        # together with its near neighbours on the OTHER side of the tape.
        groups = [[k] for k in range(len(blobs))]
        for k in range(len(blobs)):
            rk = np.linalg.norm(blobs[k][2])
            near = [k] + [j for j in range(len(blobs)) if j != k
                          and (np.linalg.norm(blobs[j][2]) > 1.0) != (rk > 1.0)
                          and np.linalg.norm(blobs[j][2] - blobs[k][2]) < 1.2 * expect]
            if len(near) > 1:
                groups.append(near)
        best, best_score, cands = None, -1e9, []
        for group in groups:
            pts = np.vstack([blobs[j][1] for j in group])
            est = self._footprint(pts)
            if est is None:
                continue
            x, y, size = est
            ok = 0.35 * expect < size < 3.0 * expect
            if len(group) == 1:
                cands.append((blobs[group[0]][0], ok))
            if not ok:
                continue
            sc = -abs(math.log(size / expect))                # about the right size
            if predicted is not None:
                sc -= 2.0 * math.hypot(x - predicted[0], y - predicted[1])
            sc -= 0.5 * max(0.0, math.hypot(x, y) - 1.0)      # mostly inside the ring
            sc += 0.1 * (len(group) > 1)                      # both halves beat one half
            if sc > best_score:
                best, best_score = (x, y, 0.55 * size), sc
        return best, cands

    def _footprint(self, pts):
        """Ground position and size of a blob from its floor-mapped pixels."""
        if len(pts) < 8:
            return None
        cam = self.ring.camera
        if cam is None:
            c = pts.mean(0)
            c = self.ring.true_from_apparent(c, self.geom.cm(config.OPP_CENTROID_HEIGHT_CM))[0]
            span = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
            return float(c[0]), float(c[1]), float(min(span) * 1.0)
        # Seen from the side, the blob = the robot's footprint plus its body smeared away from
        # the camera. Its near edge (toward the camera's foot point F) is where it meets the floor.
        F = np.array(cam["C"][:2])
        u = pts.mean(0) - F
        u /= max(np.linalg.norm(u), 1e-9)
        perp = np.array([-u[1], u[0]])
        s = (pts - F) @ u
        l = (pts - F) @ perp
        width = float(np.percentile(l, 95) - np.percentile(l, 5))
        near = float(np.percentile(s, 3))
        depth = width                                          # robots are roughly as deep as wide
        c = F + u * (near + depth / 2) + perp * float((np.percentile(l, 95) + np.percentile(l, 5)) / 2)
        return float(c[0]), float(c[1]), width
