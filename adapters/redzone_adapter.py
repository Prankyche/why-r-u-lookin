"""
adapters/redzone_adapter.py
---------------------------
Looks for the red zone in the downward Pi camera frame. One rule, no timers:
red blob centred in view -> True (the sweep goes fast), otherwise False.
"""
from typing import Optional

import cv2
import numpy as np

import config


def red_centred(frame_bgr) -> bool:
    if frame_bgr is None:
        return False
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.RED_HSV_LOW1), np.array(config.RED_HSV_HIGH1))
    mask |= cv2.inRange(hsv, np.array(config.RED_HSV_LOW2), np.array(config.RED_HSV_HIGH2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < config.RED_ZONE_MIN_AREA_PX:
        return False
    m = cv2.moments(c)
    if m["m00"] == 0:
        return False
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    r = config.RED_ZONE_TRIGGER_RADIUS
    return abs(cx - w / 2) <= r * (w / 2) and abs(cy - h / 2) <= r * (w / 2)


class RedZoneAdapter:
    def poll(self, frame) -> bool:
        return red_centred(frame)