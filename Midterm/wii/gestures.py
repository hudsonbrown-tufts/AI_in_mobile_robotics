"""Pose -> Wii pointer and button gestures.

The right hand holds the LEGO "remote", so the RIGHT WRIST is the pointer
(sent as the right stick, which the Dolphin profile binds to the Wii IR
pointer). It's measured relative to your body -- shoulder midpoint and
torso size -- so it works at any distance from the camera:
    x: wrist left/right of the shoulder midpoint, in shoulder widths
    y: wrist height around the middle of your chest

The LEFT hand is free, so it makes buttons (each must be held a moment so a
passing arm doesn't trigger it):
    left hand above your head        -> A
    both hands above your head (1 s) -> HOME
    left hand out wide to your left  -> D-pad LEFT
    left hand across to your right   -> D-pad RIGHT

Landmarks come from vision.py: mirrored, normalized (u, v), so your right
side is on the right of the image, like a mirror.
"""

import time

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 11, 12, 15, 16, 23, 24

POINTER_X_RANGE = 1.3       # wrist this many shoulder widths from center = pointer at the screen edge
POINTER_Y_CENTER = 0.35     # pointer centered with the wrist this far down the torso (0 = shoulders)
POINTER_Y_RANGE = 0.55      # torso lengths from center to the top/bottom edge
POINTER_X_OFFSET = 0.6      # shoulder widths: a right hand naturally rests right of center
TORSO_PER_SHOULDER_W = 1.4  # torso length guess if the hips are out of frame
SMOOTHING = 0.5

GESTURE_HOLD_S = 0.20       # a gesture must be held this long to press
HOME_HOLD_S = 1.0
DPAD_OUT_WIDTHS = 1.4       # left wrist this far out to the left = D-pad left
DPAD_ACROSS_WIDTHS = 0.3    # left wrist this far past center to the right = D-pad right


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Gestures:
    def __init__(self, aspect=16 / 9):
        self.aspect = aspect            # width / height of the normalized frame
        self.pointer = None             # (x, y) in -1..1, +y up
        self.buttons = set()
        self.enabled = True
        self._since = {}                # raw gesture name -> time first seen
        self._smoothed = None

    def update(self, landmarks, now=None):
        now = time.monotonic() if now is None else now
        lm = landmarks or {}
        if L_SHOULDER not in lm or R_SHOULDER not in lm:
            self.pointer, self.buttons, self._smoothed = None, set(), None
            self._since.clear()
            return

        def px(i):     # into a space where 1 unit is the same distance horizontally and vertically
            u, v = lm[i]
            return u * self.aspect, v

        (lsx, lsy), (rsx, rsy) = px(L_SHOULDER), px(R_SHOULDER)
        mid_x, sh_y = (lsx + rsx) / 2, (lsy + rsy) / 2
        sw = max(0.02, abs(rsx - lsx))
        if L_HIP in lm and R_HIP in lm:
            torso = max(0.02, (px(L_HIP)[1] + px(R_HIP)[1]) / 2 - sh_y)
        else:
            torso = TORSO_PER_SHOULDER_W * sw

        # ----- pointer: right wrist -----
        if R_WRIST in lm:
            wx, wy = px(R_WRIST)
            x = ((wx - mid_x) / sw - POINTER_X_OFFSET) / POINTER_X_RANGE
            y = -((wy - sh_y) / torso - POINTER_Y_CENTER) / POINTER_Y_RANGE
            x, y = clamp(x, -1, 1), clamp(y, -1, 1)
            if self._smoothed is None:
                self._smoothed = (x, y)
            else:
                sx, sy = self._smoothed
                self._smoothed = (sx + SMOOTHING * (x - sx), sy + SMOOTHING * (y - sy))
            self.pointer = self._smoothed
        else:
            self.pointer, self._smoothed = None, None

        # ----- buttons: left hand (and both hands for HOME) -----
        raw = set()
        if self.enabled and L_WRIST in lm:
            lx, ly = px(L_WRIST)
            head_y = px(NOSE)[1] if NOSE in lm else sh_y - 0.5 * sw
            left_up = ly < head_y
            right_up = R_WRIST in lm and px(R_WRIST)[1] < head_y
            if left_up and right_up:
                raw.add("HOME")
            elif left_up:
                raw.add("A")
            elif (mid_x - lx) / sw > DPAD_OUT_WIDTHS and ly < sh_y + torso:
                raw.add("LEFT")
            elif (lx - mid_x) / sw > DPAD_ACROSS_WIDTHS and ly < sh_y + torso:
                raw.add("RIGHT")

        for name in list(self._since):
            if name not in raw:
                del self._since[name]
        pressed = set()
        for name in raw:
            start = self._since.setdefault(name, now)
            if now - start >= (HOME_HOLD_S if name == "HOME" else GESTURE_HOLD_S):
                pressed.add(name)
        self.buttons = pressed
