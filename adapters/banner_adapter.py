"""
adapters/banner_adapter.py
--------------------------
Finds a banner on the downward Pi camera and centres the drone over it. Sends BANNER_FOUND
once it has been centred for BANNER_ALIGNED_FRAMES frames in a row.

Two uses, same code:
  ALIGN_BANNER       green entry banner  (config.BANNER_HSV)
  ALIGN_EXIT_BANNER  blue exit banner    (config.EXIT_BANNER_HSV)

Commands (tuples), executed by the runner:
  ("velocity_body", forward_ms, right_ms)   body-frame horizontal velocity while aligning
  ("hold",)                                  banner not seen, or aligned: zero velocity
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from fsm.mission_fsm import Event, State


@dataclass
class BannerOutput:
    events: List[Event] = field(default_factory=list)
    commands: List[tuple] = field(default_factory=list)


def find_banner(frame_bgr, hsv=None, min_area=None) -> Optional[Tuple[float, float]]:
    """Normalised (ex, ey) of the largest blob of this exact HSV colour, relative to the image
    centre, in -1..+1. None if no blob is large enough."""
    hue, sat, val = hsv or config.BANNER_HSV
    min_area = config.BANNER_MIN_AREA_PX if min_area is None else min_area
    h, w = frame_bgr.shape[:2]
    img_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,
                       np.array([hue - config.BANNER_HUE_TOL, config.BANNER_S_MIN, config.BANNER_V_MIN]),
                       np.array([hue + config.BANNER_HUE_TOL, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    return ((x + bw / 2 - w / 2) / (w / 2), (y + bh / 2 - h / 2) / (h / 2))


def centring_velocity(ex: float, ey: float) -> Tuple[float, float, bool]:
    """Forward and right velocities (m/s) that move the banner toward the image centre."""
    tol = config.BANNER_ALIGN_TOL
    right = config.BANNER_RIGHT_SIGN * config.BANNER_KP * ex if abs(ex) > tol else 0.0
    forward = config.BANNER_FORWARD_SIGN * config.BANNER_KP * ey if abs(ey) > tol else 0.0
    m = config.BANNER_MAX_SPEED_MS
    right, forward = max(-m, min(m, right)), max(-m, min(m, forward))
    centred = abs(ex) <= tol and abs(ey) <= tol
    return forward, right, centred


class BannerAdapter:
    def __init__(self, state: State = State.ALIGN_BANNER,
                 hsv: Optional[tuple] = None, min_area: Optional[int] = None):
        self.state = state
        self.hsv = hsv or config.BANNER_HSV
        self.min_area = config.BANNER_MIN_AREA_PX if min_area is None else min_area
        self._centred_frames = 0

    def poll(self, state: State, frame) -> BannerOutput:
        out = BannerOutput()
        if state is not self.state or frame is None:
            self._centred_frames = 0
            return out

        pos = find_banner(frame, self.hsv, self.min_area)
        if pos is None:
            self._centred_frames = 0
            out.commands.append(("hold",))
            return out

        forward, right, centred = centring_velocity(*pos)
        self._centred_frames = self._centred_frames + 1 if centred else 0
        if self._centred_frames >= config.BANNER_ALIGNED_FRAMES:
            out.events.append(Event.BANNER_FOUND)
            out.commands.append(("hold",))
        else:
            out.commands.append(("velocity_body", forward, right))
        return out