"""Perspective correction for a camera that looks at the ring from the side (e.g. on a chair).

From an angle, the tape circle still looks like an ellipse, but:
  - the near half of the ring is magnified and the far half shrunk, so a simple
    "squash the ellipse back into a circle" map puts things in the wrong place,
  - the ellipse's center is NOT the ring's center,
  - things above the floor (our AprilTags, the opponent's body) appear shifted
    away from the camera.

One ellipse can't tell how the floor is tilted (it fixes only 2 of the 4 numbers
needed), so we add the robot's FRONT/BACK tags: each is a square, and every view
of the same tag is the same size. (The two tags may sit at different heights,
which makes them look slightly different in size, so sizes are only compared
within one tag.) `solve_floor_homography` finds the floor -> image homography H
(ring units -> pixels) that makes
  - every tape pixel land on the unit circle, and
  - every observed tag a square, every view of one tag the same size.
Tags sit on a plane parallel to the floor, which the camera sees as the floor
plane scaled about the camera's foot point; that keeps squares square, so the
tags' height doesn't bias this fit.

`camera_from_homography` then recovers the camera itself (focal length, and
its position above the ring in ring units), assuming square pixels and the
principal point at the image center (true for phones). With the camera's
height and foot point, raised points are corrected exactly: a point at height
h appears at  F + (P - F) * Hc / (Hc - h)  on the floor, where F is the foot
point and Hc the camera height.
"""

import math

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

# A square's corners in the tag's own frame: TL, TR, BR, BL (AprilTag corner order).
SQUARE = np.array([-1 + 1j, 1 + 1j, 1 - 1j, -1 - 1j])


def apply_h(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H).T
    return q[:, :2] / q[:, 2:3]


def _square_fit(z, canon):
    """Best similarity  canon -> z  (complex). Returns (residual array, scale)."""
    m = z.mean()
    d = z - m
    a = np.sum(np.conj(canon) * d) / np.sum(np.abs(canon) ** 2)
    return d - a * canon, abs(a)


def solve_floor_homography(H0, tape_px, tag_obs, tag_groups=None, w_tape=100.0, w_tag=60.0, w_scale=30.0):
    """Refine H0 (ring units -> pixels).

    tape_px: Nx2 pixel points on the tape (should map to radius 1).
    tag_obs: list of 4x2 pixel corner arrays (TL, TR, BR, BL).
    tag_groups: one label per view (the tag ID); views with the same label must be the
        same size. None = all one group.
    Returns (H, info dict).
    """
    H0 = np.asarray(H0, float) / H0[2, 2]
    x0 = H0.ravel()[:8]
    tape_px = np.asarray(tape_px, float)
    tags = [np.asarray(t, float).reshape(4, 2) for t in tag_obs]

    # The image -> floor map may mirror the tag's corner order; pick the orientation once.
    Hinv0 = np.linalg.inv(H0)
    canons = []
    for t in tags:
        q = apply_h(Hinv0, t)
        z = q[:, 0] + 1j * q[:, 1]
        r1, _ = _square_fit(z, SQUARE)
        r2, _ = _square_fit(z, np.conj(SQUARE))
        canons.append(SQUARE if np.sum(np.abs(r1) ** 2) <= np.sum(np.abs(r2) ** 2) else np.conj(SQUARE))

    res_for = _residual_fn(tape_px, tags, canons, tag_groups, w_tape, w_tag, w_scale)

    def residuals(x):
        H = np.append(x, 1.0).reshape(3, 3)
        reg = 1e-3 * (x - x0) / (np.abs(x0) + 1.0)          # pins down the free rotation about the center
        return np.concatenate([res_for(H), reg])

    sol = least_squares(residuals, x0, method="trf", loss="soft_l1", f_scale=1.0, max_nfev=400)
    H = np.append(sol.x, 1.0).reshape(3, 3)
    Hinv = np.linalg.inv(H)
    q = apply_h(Hinv, tape_px)
    tape_rms = float(np.sqrt(np.mean((np.hypot(q[:, 0], q[:, 1]) - 1) ** 2)))
    shape = []
    for t, canon in zip(tags, canons):
        c = apply_h(Hinv, t)
        z = c[:, 0] + 1j * c[:, 1]
        r, a = _square_fit(z, canon)
        shape.append(np.sqrt(np.mean(np.abs(r) ** 2)) / a)
    return H, {"tape_rms": tape_rms, "tag_shape_rms": float(np.mean(shape)) if shape else None,
               "n_tags": len(tags), "ok": sol.success}


def _residual_fn(tape_px, tags, canons, tag_groups=None, w_tape=100.0, w_tag=60.0, w_scale=30.0):
    groups = np.array(tag_groups if tag_groups is not None else [0] * len(tags))
    labels = sorted(set(groups.tolist()))

    def residuals_for(H):
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.full(len(tape_px) + 9 * len(tags), 1e3)
        q = apply_h(Hinv, tape_px)
        res = [w_tape * (np.hypot(q[:, 0], q[:, 1]) - 1.0)]
        logs = []
        for t, canon in zip(tags, canons):
            c = apply_h(Hinv, t)
            z = c[:, 0] + 1j * c[:, 1]
            r, a = _square_fit(z, canon)
            r = r / max(a, 1e-9)
            res.append(w_tag * np.concatenate([r.real, r.imag]))
            logs.append(math.log(max(a, 1e-9)))
        if logs:
            logs = np.array(logs)
            for g in labels:                        # same size within each tag (same height)
                sel = groups == g
                res.append(w_scale * (logs[sel] - logs[sel].mean()))
        return np.concatenate(res)
    return residuals_for


def _rodrigues(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _rotvec(R):
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def solve_camera(H_init, tape_px, tag_obs, image_size, tag_groups=None):
    """Fit a real camera (focal length, rotation, position; square pixels, centered lens).

    Starts from the free 8-number homography `H_init` (solve_floor_homography), whose
    decomposition seeds f, R, t; then refines those 7 numbers on the same evidence.
    Returns (H, camera dict, info) or (None, None, info).
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    seed = camera_from_homography(H_init, image_size, strict=False)
    if seed is None:
        return None, None, {"reason": "no camera seed"}
    tape_px = np.asarray(tape_px, float)
    tags = [np.asarray(t, float).reshape(4, 2) for t in tag_obs]
    Hinv0 = np.linalg.inv(H_init)
    canons = []
    for t in tags:
        q = apply_h(Hinv0, t)
        z = q[:, 0] + 1j * q[:, 1]
        r1, _ = _square_fit(z, SQUARE)
        r2, _ = _square_fit(z, np.conj(SQUARE))
        canons.append(SQUARE if np.sum(np.abs(r1) ** 2) <= np.sum(np.abs(r2) ** 2) else np.conj(SQUARE))
    res_for = _residual_fn(tape_px, tags, canons, tag_groups)

    def H_of(x):
        f = math.exp(x[0])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        R = _rodrigues(x[1:4])
        return K @ np.column_stack([R[:, 0], R[:, 1], x[4:7]])

    x0 = np.concatenate([[math.log(seed["f"])], _rotvec(seed["R"]), seed["t"]])
    sol = least_squares(lambda x: res_for(H_of(x)), x0, method="trf", loss="soft_l1", max_nfev=600)
    H = H_of(sol.x)
    H = H / H[2, 2]
    f = math.exp(sol.x[0])
    R = _rodrigues(sol.x[1:4])
    C = -R.T @ sol.x[4:7]
    info = {"ok": bool(sol.success), "cost": float(sol.cost)}
    if not (0.25 * w < f < 5 * w) or abs(C[2]) < 0.2:
        info["reason"] = f"implausible camera (f {f:.0f}px, height {abs(C[2]):.2f})"
        return None, None, info
    cam = {"f": f, "C": [float(C[0]), float(C[1]), float(abs(C[2]))]}
    return H, cam, info


def camera_from_homography(H, image_size, strict=True):
    """Focal length (px) and camera center (ring units, z = height) from H = K [r1 r2 t].

    strict: return None if H doesn't look like a real camera (e.g. the purely affine
    first guess). With strict=False it returns the closest camera anyway (as a seed).
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    def cols(f):
        Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])
        M = Kinv @ H
        return M[:, 0], M[:, 1], M[:, 2]

    def cost(logf):
        m1, m2, _ = cols(math.exp(logf))
        n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
        return (m1 @ m2 / (n1 * n2)) ** 2 + ((n1 - n2) / (n1 + n2)) ** 2

    grid = np.linspace(math.log(0.2 * w), math.log(6 * w), 300)
    costs = [cost(g) for g in grid]
    i = int(np.argmin(costs))
    lo, hi = grid[max(0, i - 2)], grid[min(len(grid) - 1, i + 2)]
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded")
    f = math.exp(res.x)
    if strict and (res.fun > 2e-3 or not (0.25 * w < f < 5 * w)):
        return None
    m1, m2, m3 = cols(f)
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    if m3[2] * lam < 0:
        lam = -lam            # the ring must be in front of the camera
    r1, r2, t = lam * m1, lam * m2, lam * m3
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    if strict and abs(C[2]) < 1e-3:
        return None
    return {"f": f, "C": [float(C[0]), float(C[1]), float(abs(C[2]))], "orth_err": float(res.fun),
            "R": R, "t": t}
