#!/usr/bin/env python3
"""
aerothon_qr.py - QR vision for the AeroTHON delivery mission.  (vision only: no MAVLink here)

Camera : Arducam B0332 = OV9281 mono global shutter, 1280x800, 3 um pixels, 70 deg (H) lens,
         f = (1280/2)/tan(35 deg) = 914 px.  Calibrate it (calibrate_camera.py) before flying.
Target : black-on-white QR, 1.00 m x 1.00 m, plus a 3 inch (0.0762 m) white border on every side
         -> white box = 1.1524 m.   1 m QR = 91 px at 10 m, 183 px at 5 m  (box 105 px / 211 px).

Pipeline (white-box logic, as in qr_whiteboxcheck.py)
  1. find white squares (the border) on a 1/4 scale image, filtered by the size they MUST have at
     the current altitude, squareness, fill, and "dark/light pattern inside"
  2. refine the box at full resolution, decode inside it (pyzbar; CLAHE/Otsu rescue)
  3. boxes that do not decode are kept as undecoded candidates (nothing is lost)
  4. whole-frame decode every few frames as a safety net

Matching the target (the part that does not trust decoded text)
  The start QR may not decode to the same text as the target, or the target may be too small to
  decode at all, so every candidate is also compared by PATTERN: the QR is rectified to a canonical
  square, locally contrast-normalised, the three finder-pattern corners are masked out (every QR
  shares those, they would inflate decoy scores), and the correlation is taken over 4 rotations and
  a small shift range.  Measured on synthetic frames: same code 0.92-0.99 (10 m ... 5 m, blur,
  noise, tilt), different codes <= 0.59 even when the text differs by one character.

  accept if   text equal                              (strong)
         or   undecoded      and pattern >= 0.80
         or   decoded, text differs, pattern >= 0.85  (text is not trusted; pattern must be convincing)
  and, for pattern-only matches, the best must beat every other candidate by 0.15.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode


# ============================================================================ config
@dataclass
class QRConfig:
    # ---- camera: Arducam B0332 (OV9281) ----
    width: int = 1280
    height: int = 800
    fps: int = 30
    exposure_abs: Optional[int] = None      # manual exposure, V4L2 units (100 us); None = auto
    hfov_deg: float = 70.0                  # datasheet, nominal
    sensor_w_mm: float = 3.84               # 1280 px x 3 um
    calib: Optional[Tuple[float, float, float, float]] = None   # (fx, fy, cx, cy) from calibrate_camera.py
    latency_s: float = 0.05                 # exposure + USB + MJPG decode; measure it

    # ---- the printed QR ----
    qr_size_m: float = 1.0                  # QR modules only
    border_m: float = 0.0762                # 3 inch white border on each side

    # ---- white-box finder ----
    box_scale: float = 0.25
    white_min: int = 40                     # absolute floor for any first-pass threshold
    white_levels: Tuple[float, ...] = (0.85, 0.70, 0.55, 0.42, 0.30)   # first-pass thresholds, x brightest 0.1 % of the frame
    max_fits_per_level: int = 12            # cap on local fits per threshold level (CPU)
    box_squareness: float = 0.78            # strict tests, applied to the locally re-fitted box
    box_fill_min: float = 0.80
    box_local_contrast: float = 35.0        # white level minus ground level needed for a local fit
    box_size_tol: float = 0.40              # +-40 % of the size expected at this altitude
    box_min_px: int = 30                    # used only when the altitude is unknown
    box_max_px: int = 900
    inner_contrast_min: float = 70.0
    inner_dark_range: Tuple[float, float] = (0.15, 0.75)

    # ---- decoding ----
    fullframe_every: int = 4
    crop_margin: float = 0.25

    # ---- pattern matching ----
    patch: int = 128                        # canonical square, px
    mask_frac: float = 0.34                 # corner squares masked out (finder patterns)
    pattern_blur: float = 2.0               # px at canonical size
    shift_margin: int = 10                  # +- px of misregistration tolerated
    match_threshold: float = 0.80           # undecoded candidate vs reference
    conflict_threshold: float = 0.85        # decoded candidate with DIFFERENT text vs reference
    match_margin: float = 0.15              # best must beat the runner-up by this (pattern-only matches)
    same_threshold: float = 0.85            # "same code" test used while building the start reference

    # ---- voting ----
    start_window: int = 5
    start_needed: int = 3
    confirm_frames: int = 3                 # target seen in this many frames ...
    confirm_window_s: float = 1.0           # ... within this time

    # ---- derived ----
    @property
    def border_frac(self) -> float:
        return self.border_m / self.qr_size_m

    @property
    def box_m(self) -> float:
        return self.qr_size_m + 2.0 * self.border_m

    @property
    def fx(self) -> float:
        return self.calib[0] if self.calib else (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

    def expected_box_px(self, altitude_m: float) -> float:
        return self.box_m * self.fx / altitude_m

    def expected_qr_px(self, altitude_m: float) -> float:
        return self.qr_size_m * self.fx / altitude_m


# ============================================================================ data
@dataclass(eq=False)
class Candidate:
    text: Optional[str]                     # decoded text, None = undecoded white box
    quad: np.ndarray                        # inner QR corners, full-frame px, float32 (4,2), clockwise
    method: str                             # crop | rescue | direct | box
    frame_wh: Tuple[int, int]
    patch: Optional[np.ndarray] = None      # canonical normalised patch
    score: float = 0.0                      # pattern score against the reference
    verdict: str = ""                       # text | pattern | conflict | "" (not accepted)
    t: float = field(default_factory=time.time)

    @property
    def center(self) -> np.ndarray:
        return self.quad.mean(axis=0)

    @property
    def size_px(self) -> float:
        q = self.quad
        return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


@dataclass
class Reference:
    """The start QR as the matcher knows it."""
    patch: np.ndarray
    text: Optional[str] = None
    size_px: float = 0.0

    def save(self, path: str) -> None:
        np.savez(path, patch=self.patch, text=np.array(self.text or ""), has_text=np.array(self.text is not None),
                 size_px=np.array(self.size_px))

    @staticmethod
    def load(path: str) -> "Reference":
        d = np.load(path, allow_pickle=False)
        text = str(d["text"]) if bool(d["has_text"]) else None
        return Reference(d["patch"].astype(np.float32), text, float(d["size_px"]))


Detection = Tuple[float, float, float]      # (u, v, capture_time)


# ============================================================================ geometry helpers
def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def order_quad(pts) -> np.ndarray:
    """4 corners, clockwise in image coordinates (start corner arbitrary: matching tries 4 rotations)."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p))
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)].astype(np.float32)


def expand_quad(q: np.ndarray, f: float) -> np.ndarray:
    c = q.mean(axis=0)
    return (c + (q - c) * f).astype(np.float32)


def quad_side(q: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


# ============================================================================ pattern matching
def canonical_patch(gray: np.ndarray, quad: np.ndarray, cfg: QRConfig) -> np.ndarray:
    """Rectify the QR to a cfg.patch square and normalise local contrast (illumination independent)."""
    K = cfg.patch
    q = order_quad(quad)
    side = quad_side(q)
    h, w = gray.shape[:2]
    x0, y0 = np.maximum(q.min(axis=0).astype(int) - 4, 0)
    x1, y1 = np.minimum(q.max(axis=0).astype(int) + 5, [w, h])
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((K, K), np.float32)
    if side > 1.3 * K:                       # anti-alias before shrinking
        roi = cv2.GaussianBlur(roi, (0, 0), 0.45 * side / K)
    dst = np.array([[0, 0], [K - 1, 0], [K - 1, K - 1], [0, K - 1]], np.float32)
    m = cv2.getPerspectiveTransform(q - np.array([x0, y0], np.float32), dst)
    p = cv2.warpPerspective(roi, m, (K, K), flags=cv2.INTER_LINEAR).astype(np.float32)
    bg = cv2.GaussianBlur(p, (0, 0), K / 6.0)
    p = p - bg
    sd = np.sqrt(cv2.GaussianBlur(p * p, (0, 0), K / 6.0)) + 1e-3
    return np.clip(p / sd, -2.5, 2.5)


class PatternMatcher:
    """Scores any patch against one reference. Templates are prepared once."""

    def __init__(self, reference_patch: np.ndarray, cfg: QRConfig):
        self.cfg = cfg
        K, m = cfg.patch, cfg.shift_margin
        rb = cv2.GaussianBlur(reference_patch, (0, 0), cfg.pattern_blur) if cfg.pattern_blur else reference_patch
        self.templates = [np.ascontiguousarray(np.rot90(rb, k)[m:K - m, m:K - m]) for k in range(4)]
        n = K - 2 * m
        mask = np.ones((n, n), np.float32)
        c = max(0, int(round(cfg.mask_frac * K)) - m)
        if c > 0:
            mask[:c, :c] = 0
            mask[:c, -c:] = 0
            mask[-c:, :c] = 0
            mask[-c:, -c:] = 0
        self.mask = mask

    def score(self, patch: np.ndarray) -> float:
        cb = cv2.GaussianBlur(patch, (0, 0), self.cfg.pattern_blur) if self.cfg.pattern_blur else patch
        best = -1.0
        for t in self.templates:
            r = cv2.matchTemplate(cb, t, cv2.TM_CCORR_NORMED, mask=self.mask)
            v = float(np.nanmax(r)) if r.size else -1.0
            if np.isfinite(v) and v > best:
                best = v
        return best


def pattern_score(a: np.ndarray, b: np.ndarray, cfg: QRConfig) -> float:
    return PatternMatcher(a, cfg).score(b)


# ============================================================================ detector
class QRDetector:
    """White boxes -> decode inside -> undecoded boxes kept as candidates (+ whole-frame safety net)."""

    def __init__(self, cfg: Optional[QRConfig] = None):
        self.cfg = cfg or QRConfig()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._n = 0

    # ---- low level ----
    @staticmethod
    def _zbar(img: np.ndarray):
        return zbar_decode(img, symbols=[ZBarSymbol.QRCODE])

    def _make(self, results, method: str, gray: np.ndarray, scale=1.0, ox=0.0, oy=0.0) -> List[Candidate]:
        h, w = gray.shape[:2]
        out = []
        for r in results:
            poly = np.array([[p.x, p.y] for p in r.polygon], np.float32)
            if len(poly) < 3:
                continue
            quad = (order_quad(poly) / scale + np.array([ox, oy], np.float32)).astype(np.float32)
            c = Candidate(r.data.decode("utf-8", errors="replace"), quad, method, (w, h))
            c.patch = canonical_patch(gray, quad, self.cfg)
            out.append(c)
        return out

    @staticmethod
    def _duplicate(c: Candidate, others: List[Candidate]) -> bool:
        return any(np.linalg.norm(c.center - o.center) < 0.5 * max(o.size_px, c.size_px, 1.0) for o in others)

    # ---- white box ----
    def _inner(self, box: np.ndarray) -> np.ndarray:
        return expand_quad(box, 1.0 / (1.0 + 2.0 * self.cfg.border_frac))

    def _looks_like_qr(self, gray: np.ndarray, box: np.ndarray) -> bool:
        c = self.cfg
        inner = self._inner(box)
        h, w = gray.shape[:2]
        x0, y0 = np.maximum(inner.min(axis=0).astype(int), 0)
        x1, y1 = np.minimum(inner.max(axis=0).astype(int), [w, h])
        crop = gray[y0:y1:2, x0:x1:2]
        if crop.size < 100:
            return False
        lo, hi = np.percentile(crop, (5, 95))
        if hi - lo < c.inner_contrast_min:
            return False
        dark = float((crop < (lo + hi) / 2.0).mean())
        return c.inner_dark_range[0] <= dark <= c.inner_dark_range[1]

    def _fit_box(self, gray: np.ndarray, coarse: np.ndarray, exp: Optional[float]) -> Optional[np.ndarray]:
        """
        Stage 2: re-fit the white box at full resolution with a LOCAL threshold halfway between this
        blob's own white level and the ground level around it. Immune to shade (both levels drop) and to
        bright ground merging into the blob. Returns the 4 corners, or None if it is not a clean square.
        """
        c = self.cfg
        h, w = gray.shape[:2]
        side0 = quad_side(coarse)
        pad = 0.15 * side0
        x0, y0 = np.maximum(coarse.min(axis=0) - pad, 0).astype(int)
        x1, y1 = np.minimum(coarse.max(axis=0) + pad, [w, h]).astype(int)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        inside = np.zeros(crop.shape, np.uint8)
        hull = cv2.convexHull((expand_quad(coarse, 1.12) - np.array([x0, y0], np.float32)).astype(np.int32))
        cv2.fillConvexPoly(inside, hull, 255)
        ring = crop[inside == 0]
        ground = float(np.median(ring)) if ring.size > 50 else float(np.percentile(crop, 20))
        white = float(np.percentile(crop[inside > 0], 98)) if inside.any() else float(np.percentile(crop, 98))
        if white - ground < c.box_local_contrast:
            return None
        m = (crop > 0.5 * (white + ground)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c0 = coarse.mean(axis=0) - np.array([x0, y0], np.float32)
        best, best_d = None, 1e9
        for cnt in cnts:
            if cv2.contourArea(cnt) < 0.3 * side0 * side0:
                continue
            rect = cv2.minAreaRect(cnt)
            bw, bh = rect[1]
            if min(bw, bh) < 2 or min(bw, bh) / max(bw, bh) < c.box_squareness:
                continue
            if cv2.contourArea(cnt) / (bw * bh) < c.box_fill_min:
                continue
            side = 0.5 * (bw + bh)
            if exp:
                if abs(side - exp) > c.box_size_tol * exp:
                    continue
            elif not (c.box_min_px <= side <= c.box_max_px):
                continue
            d = float(np.linalg.norm(np.array(rect[0]) - c0))
            if d < best_d and d < 0.25 * side0:
                best, best_d = rect, d
        if best is None:
            return None
        return order_quad(cv2.boxPoints(best)) + np.array([x0, y0], np.float32)

    def _white_boxes(self, gray: np.ndarray, altitude_m: Optional[float]) -> List[np.ndarray]:
        """
        Stage 1 (coarse, 1/4 scale): ladder of thresholds relative to the brightest paper in view; blobs of
        plausible size and squareness go on. Stage 2 (_fit_box, full resolution): strict local fit.
        A QR in sun is found on the top rungs, one in shade on a lower rung, and a blob merged with bright
        ground on one rung is clean on another.
        """
        c = self.cfg
        s = c.box_scale
        # max-pool (not area-average): a 9 px wide border averaged into 4x4 blocks loses a third of its
        # brightness and breaks up in dim light; max-pooling keeps thin white features at full level.
        # (the blob grows by <= 3 px; stage 2 recomputes the exact geometry at full resolution)
        k = int(round(1.0 / s))
        small = cv2.dilate(gray, np.ones((k, k), np.uint8))[::k, ::k]
        s = 1.0 / k
        top = float(np.percentile(small, 99.9))
        exp = c.expected_box_px(altitude_m) if altitude_m and altitude_m > 1.0 else None
        boxes: List[np.ndarray] = []
        tried: List[np.ndarray] = []
        for rel in c.white_levels:
            thr = max(c.white_min, rel * top)
            mask = cv2.threshold(small, thr, 255, cv2.THRESH_BINARY)[1]
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = []
            for cnt in cnts:
                if len(cnt) < 4:
                    continue
                rect = cv2.minAreaRect(cnt)
                bw, bh = rect[1]
                if min(bw, bh) < 2:
                    continue
                side = 0.5 * (bw + bh) / s
                if exp:                                    # loose gate: a merged blob is bigger than a clean box
                    if not (0.5 * exp <= side <= 1.9 * exp):
                        continue
                elif not (c.box_min_px <= side <= c.box_max_px * 1.5):
                    continue
                sq = min(bw, bh) / max(bw, bh)
                fill = cv2.contourArea(cnt) / (bw * bh)
                if sq < 0.55 or fill < 0.55:
                    continue
                if any(np.linalg.norm(np.array(rect[0]) / s - t) < 0.3 * side for t in tried):
                    continue                               # this spot already has a box
                blobs.append((sq * fill, rect, side))
            for _, rect, side in sorted(blobs, key=lambda b: -b[0])[:c.max_fits_per_level]:
                box = self._fit_box(gray, order_quad(cv2.boxPoints(rect) / s), exp)
                if box is not None and self._looks_like_qr(gray, box):
                    boxes.append(box)
                    tried.append(box.mean(axis=0))
        return boxes

    # ---- decoding inside a box ----
        # ---- decoding inside a box ----
    def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[Candidate]:
        h, w = gray.shape[:2]
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        margin = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
        x0, y0 = int(max(0, x0 - margin)), int(max(0, y0 - margin))
        x1, y1 = int(min(w, x1 + margin)), int(min(h, y1 + margin))
        base = gray[y0:y1, x0:x1]
        if base.size == 0:
            return []

        # Try the native crop first, then a 2x crop for small targets. Keep the
        # scale for mapping pyzbar's crop coordinates back to full-frame pixels.
        initial_scale = 2.0 if max(base.shape) < 240 else 1.0
        scales = [initial_scale]
        if initial_scale != 3.0:
            scales.append(3.0)  # extra rescue for small/blurred QRs at altitude

        for scale in scales:
            crop = (base if scale == 1.0 else
                    cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

            # Original grayscale decode.
            res = self._zbar(crop)
            if res:
                return self._make(res, "crop", gray, scale, x0, y0)

            # Contrast/sharpen rescue, followed by Otsu thresholding.
            clahe = self._clahe.apply(crop)
            sharp = cv2.addWeighted(
                clahe, 1.6,
                cv2.GaussianBlur(clahe, (0, 0), 2.0),
                -0.6, 0,
            )
            variants = (
                sharp,
                cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
            )
            for variant in variants:
                res = self._zbar(variant)
                if res:
                    return self._make(res, "rescue", gray, scale, x0, y0)

            # OpenCV's QR decoder is a last resort. Its geometry is not used,
            # because the white-box corners are more stable for the pixel centre.
            try:
                text, points, _ = cv2.QRCodeDetector().detectAndDecode(crop)
            except cv2.error:
                text, points = "", None
            if text and points is not None:
                candidate = Candidate(text, quad.copy(), "rescue", (w, h))
                candidate.patch = canonical_patch(gray, candidate.quad, self.cfg)
                return [candidate]

        return []
    # def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[Candidate]:
    #     h, w = gray.shape[:2]
    #     x0, y0 = quad.min(axis=0)
    #     x1, y1 = quad.max(axis=0)
    #     m = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
    #     x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    #     x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
    #     crop = gray[y0:y1, x0:x1]
    #     if crop.size == 0:
    #         return []
    #     up = 2.0 if max(crop.shape) < 240 else 1.0
    #     if up != 1.0:
    #         crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    #     res = self._zbar(crop)
    #     if res:
    #         return self._make(res, "crop", gray, up, x0, y0)
    #     c = self._clahe.apply(crop)
    #     sharp = cv2.addWeighted(c, 1.6, cv2.GaussianBlur(c, (0, 0), 2.0), -0.6, 0)
    #     res = self._zbar(sharp) or self._zbar(cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    #     if res:
    #         return self._make(res, "rescue", gray, up, x0, y0)
    #     return []

    # ---- one frame ----
    def scan(self, gray: np.ndarray, altitude_m: Optional[float] = None,
             full: Optional[bool] = None) -> List[Candidate]:
        """All QR candidates in this frame (decoded or not). altitude_m = height above the ground."""
        self._n += 1
        h, w = gray.shape[:2]
        found: List[Candidate] = []
        for box in self._white_boxes(gray, altitude_m):
            inner = self._inner(box)
            probe = Candidate(None, inner, "box", (w, h))
            if self._duplicate(probe, found):
                continue
            decoded = self._decode_crop(gray, inner)
            if decoded:
                found += [d for d in decoded if not self._duplicate(d, found)]
            else:
                probe.patch = canonical_patch(gray, inner, self.cfg)
                found.append(probe)
        if full is None:
            full = self._n % self.cfg.fullframe_every == 0
        if full:
            found += [d for d in self._make(self._zbar(gray), "direct", gray) if not self._duplicate(d, found)]
        return found


# ============================================================================ start QR -> reference
class StartScanner:
    """
    Call once per frame while looking at the start QR (~5 m). Returns a Reference once the same code
    has been seen in `start_needed` of the last `start_window` frames. 'Same' = same decoded text OR
    same pattern, so it also works when the code never decodes.
    """

    def __init__(self, cfg: Optional[QRConfig] = None, detector: Optional[QRDetector] = None):
        self.cfg = cfg or QRConfig()
        self.det = detector or QRDetector(self.cfg)
        self.window: Deque[Optional[Candidate]] = deque(maxlen=self.cfg.start_window)

    def _same(self, a: Candidate, b: Candidate) -> bool:
        if a.text is not None and b.text is not None and a.text.strip() == b.text.strip():
            return True
        return pattern_score(a.patch, b.patch, self.cfg) >= self.cfg.same_threshold

    def update(self, gray: np.ndarray, altitude_m: Optional[float] = None) -> Optional[Reference]:
        cands = [c for c in self.det.scan(gray, altitude_m) if c.patch is not None]
        self.window.append(max(cands, key=lambda c: c.size_px) if cands else None)
        seen = [c for c in self.window if c is not None]
        if len(seen) < self.cfg.start_needed:
            return None
        best_group: List[Candidate] = []
        for a in seen:
            group = [b for b in seen if b is a or self._same(a, b)]
            if len(group) > len(best_group):
                best_group = group
        if len(best_group) < self.cfg.start_needed:
            return None
        texts = Counter(c.text.strip() for c in best_group if c.text is not None)
        text = texts.most_common(1)[0][0] if texts else None
        pick = max(best_group, key=lambda c: (c.text is not None, c.size_px))
        return Reference(pick.patch.copy(), text, pick.size_px)


# ============================================================================ target decision
class TargetMatcher:
    """Decides which candidate in a frame (if any) is the target. Text is evidence, pattern decides."""

    def __init__(self, reference: Reference, cfg: Optional[QRConfig] = None):
        self.cfg = cfg or QRConfig()
        self.ref = reference
        self.pm = PatternMatcher(reference.patch, self.cfg)
        self._ref_text = reference.text.strip() if reference.text is not None else None

    def match(self, cands: List[Candidate]) -> Optional[Candidate]:
        cfg = self.cfg

        for candidate in cands:
            candidate.verdict = ""
            candidate.score = (
                self.pm.score(candidate.patch) if candidate.patch is not None else -1.0
            )

            text_matches = (
                self._ref_text is not None
                and candidate.text is not None
                and candidate.text.strip() == self._ref_text
            )

            if text_matches:
                candidate.verdict = "text"
            elif candidate.text is None:
                if candidate.score >= cfg.match_threshold:
                    candidate.verdict = "pattern"
            elif candidate.score >= cfg.conflict_threshold:
                candidate.verdict = "conflict"

        accepted = [c for c in cands if c.verdict]
        if not accepted:
            return None

        best = max(accepted, key=lambda c: (c.verdict == "text", c.score))

        if best.verdict != "text":
            other_scores = [
                c.score for c in cands
                if c is not best and c.patch is not None
            ]
            if other_scores and best.score - max(other_scores) < cfg.match_margin:
                return None  # two similar candidates: don't guess

        return best


# ============================================================================ tracking
class TargetTracker:
    """Per frame: scan -> decide -> publish (u, v, capture time). Keeps recent detections for confirmation."""

    def __init__(self, detector: QRDetector, matcher: TargetMatcher,
                 altitude_fn: Callable[[], Optional[float]] = lambda: None,
                 publish: Optional[Callable[[float, float, float], None]] = None,
                 undistort: Optional[Callable[[float, float], Tuple[float, float]]] = None):
        self.det, self.matcher, self.altitude_fn = detector, matcher, altitude_fn
        self.publish, self.undistort = publish, undistort
        self.cfg = matcher.cfg
        self.recent: Deque[Detection] = deque(maxlen=20)
        self.last: Optional[Candidate] = None
        self.frames = self.matches = 0

    def process(self, gray: np.ndarray, t_capture: float) -> Optional[Detection]:
        self.frames += 1
        cands = self.det.scan(gray, self.altitude_fn())
        hit = self.matcher.match(cands)
        self.last = hit
        if hit is None:
            return None
        u, v = (float(x) for x in hit.center)
        if self.undistort:
            u, v = self.undistort(u, v)
        d = (u, v, t_capture - self.cfg.latency_s)
        self.matches += 1
        self.recent.append(d)
        if self.publish:
            self.publish(*d)
        return d

    def confirmed(self) -> Optional[List[Detection]]:
        now = time.time()
        fresh = [d for d in self.recent if now - d[2] < self.cfg.confirm_window_s]
        return fresh[-self.cfg.confirm_frames:] if len(fresh) >= self.cfg.confirm_frames else None


# ============================================================================ camera
class FrameGrabber:
    """
    Reads the camera on a thread and always hands out the NEWEST frame together with its capture time
    (stamped as soon as cap.read() returns). Forces MJPG 1280x800, buffer 1.
    """

    def __init__(self, src=0, cfg: Optional[QRConfig] = None):
        cfg = cfg or QRConfig()
        backend = cv2.CAP_V4L2 if (isinstance(src, int) and sys.platform.startswith("linux")) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # before the size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cfg.exposure_abs is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                         # 1 = manual (V4L2)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_abs)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.size != (cfg.width, cfg.height):
            print(f"[WARN] camera delivers {self.size[0]}x{self.size[1]}, wanted {cfg.width}x{cfg.height}: "
                  f"the camera model and the size filter assume {cfg.width}x{cfg.height}", file=sys.stderr)
        self._lock, self._new, self._stop = threading.Lock(), threading.Event(), threading.Event()
        self._frame, self._idx, self._t = None, 0, 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.time()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._idx, self._t = frame, self._idx + 1, t
            self._new.set()

    def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray, float]]:
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        with self._lock:
            return self._idx, self._frame, self._t

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DetectionWorker(threading.Thread):
    """grab -> track continuously (also while the flight code is busy flying)."""

    def __init__(self, grabber, tracker: TargetTracker):
        super().__init__(daemon=True)
        self.grabber, self.tracker = grabber, tracker
        self.errors = 0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            item = self.grabber.read(timeout=0.5)
            if item is None:
                continue
            _, frame, t = item
            try:
                self.tracker.process(to_gray(frame), t)
            except Exception as e:                 # the detector must never die mid-mission
                self.errors += 1
                print(f"[aerothon_qr] detector error #{self.errors}: {e!r}", file=sys.stderr)

    def stop(self) -> None:
        self._stop_evt.set()


def make_undistorter(K: np.ndarray, dist: np.ndarray):
    """(u, v) from the real lens -> (u, v) in the ideal pinhole image the flight code assumes."""
    def undistort(u: float, v: float) -> Tuple[float, float]:
        p = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist, P=K)
        return float(p[0, 0, 0]), float(p[0, 0, 1])
    return undistort


def load_calibration(path: str):
    """calib.npz from calibrate_camera.py -> ((fx, fy, cx, cy), K, dist)."""
    d = np.load(path)
    K, dist = d["K"], d["dist"]
    return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])), K, dist


def video_frames(path: str) -> Iterator[Tuple[int, np.ndarray, float]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame, idx / fps
    cap.release()


# ============================================================================ test harness
def _draw(frame: np.ndarray, cands: List[Candidate], target: Optional[Candidate]) -> np.ndarray:
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for k in cands:
        col = (0, 255, 0) if k is target else ((0, 200, 0) if k.text is not None else (0, 200, 255))
        cv2.polylines(img, [k.quad.astype(np.int32)], True, col, 2)
        label = f"{k.method} {k.text[:14] if k.text else '-'} {k.score:.2f} {k.verdict}"
        cv2.putText(img, label, tuple(k.quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def run(args) -> None:
    cfg = QRConfig(exposure_abs=args.exposure, qr_size_m=args.qr_size)
    if args.calib:
        cfg.calib = load_calibration(args.calib)[0]
    det = QRDetector(cfg)
    scanner = StartScanner(cfg, det) if args.mode == "start" else None
    tracker = None
    if args.mode == "target":
        if not args.reference:
            sys.exit("--mode target needs --reference FILE (made with --mode start --save-reference FILE)")
        tracker = TargetTracker(det, TargetMatcher(Reference.load(args.reference), cfg), lambda: args.altitude)

    print(f"f={cfg.fx:.0f}px  expected white box at {args.altitude:.0f} m = {cfg.expected_box_px(args.altitude):.0f}px, "
          f"QR = {cfg.expected_qr_px(args.altitude):.0f}px")
    grabber = FrameGrabber(args.camera, cfg) if args.camera is not None else None
    frames = iter(lambda: grabber.read(), None) if grabber else video_frames(args.video)
    times, n, with_hit = [], 0, 0
    t_begin = time.time()
    try:
        for item in frames:
            if item is None:
                break
            idx, frame, t_cap = item
            t0 = time.perf_counter()
            gray = to_gray(frame)
            target = None
            if args.mode == "raw":
                cands = det.scan(gray, args.altitude, full=True)
            elif args.mode == "start":
                ref = scanner.update(gray, args.altitude)
                cands = [c for c in scanner.window if c is not None][-1:]
                if ref is not None:
                    print(f"START QR CONFIRMED at frame {idx}: text={ref.text!r} size={ref.size_px:.0f}px")
                    if args.save_reference:
                        ref.save(args.save_reference)
                        print(f"reference saved to {args.save_reference}")
                    break
            else:
                tracker.process(gray, t_cap)
                target = tracker.last
                cands = [target] if target else []
                if target and tracker.confirmed():
                    u, v = target.center
                    print(f"TARGET CONFIRMED at frame {idx}: pixel=({u:.0f},{v:.0f}) {target.verdict} "
                          f"score={target.score:.2f} text={target.text!r}")
                    break
            times.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            with_hit += bool(cands)
            if args.verbose:
                for k in cands:
                    print(f"[{idx}] {k.method:6s} size={k.size_px:5.0f}px text={k.text!r} score={k.score:.2f} {k.verdict}")
            if args.show:
                cv2.imshow("qr", cv2.resize(_draw(frame, cands, target), None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_seconds and time.time() - t_begin > args.max_seconds:
                break
    finally:
        if grabber:
            grabber.close()
        if args.show:
            cv2.destroyAllWindows()
    if n:
        t = np.array(times)
        print(f"\nframes={n} frames_with_QR={with_hit} ({100 * with_hit / n:.0f}%)  "
              f"ms/frame p50={np.percentile(t, 50):.0f} p95={np.percentile(t, 95):.0f} max={t.max():.0f}  "
              f"~{1000.0 / max(t.mean(), 1e-6):.1f} FPS")


def main() -> None:
    ap = argparse.ArgumentParser(description="AeroTHON QR vision test harness")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--video", help="recording at full 1280x800 (no resizing)")
    ap.add_argument("--mode", choices=["raw", "start", "target"], default="raw")
    ap.add_argument("--reference", help="reference file for --mode target")
    ap.add_argument("--save-reference", help="where --mode start writes the reference")
    ap.add_argument("--altitude", type=float, default=10.0, help="m above ground (size filter)")
    ap.add_argument("--qr-size", type=float, default=1.0)
    ap.add_argument("--calib", help="calib.npz from calibrate_camera.py")
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (V4L2 units, 100 us)")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# aerothon_qr.py - QR vision for the AeroTHON delivery mission.  (vision only: no MAVLink here)

# Camera : Arducam B0332 = OV9281 mono global shutter, 1280x800, 3 um pixels, 70 deg (H) lens,
#          f = (1280/2)/tan(35 deg) = 914 px.  Calibrate it (calibrate_camera.py) before flying.
# Target : black-on-white QR, 1.00 m x 1.00 m, plus a 3 inch (0.0762 m) white border on every side
#          -> white box = 1.1524 m.   1 m QR = 91 px at 10 m, 183 px at 5 m  (box 105 px / 211 px).

# Pipeline (white-box logic, as in qr_whiteboxcheck.py)
#   1. find white squares (the border) on a 1/4 scale image, filtered by the size they MUST have at
#      the current altitude, squareness, fill, and "dark/light pattern inside"
#   2. refine the box at full resolution, decode inside it (pyzbar; CLAHE/Otsu rescue)
#   3. boxes that do not decode are kept as undecoded candidates (nothing is lost)
#   4. whole-frame decode every few frames as a safety net

# Matching the target (the part that does not trust decoded text)
#   The start QR may not decode to the same text as the target, or the target may be too small to
#   decode at all, so every candidate is also compared by PATTERN: the QR is rectified to a canonical
#   square, locally contrast-normalised, the three finder-pattern corners are masked out (every QR
#   shares those, they would inflate decoy scores), and the correlation is taken over 4 rotations and
#   a small shift range.  Measured on synthetic frames: same code 0.92-0.99 (10 m ... 5 m, blur,
#   noise, tilt), different codes <= 0.59 even when the text differs by one character.

#   accept if   text equal                              (strong)
#          or   undecoded      and pattern >= 0.80
#          or   decoded, text differs, pattern >= 0.85  (text is not trusted; pattern must be convincing)
#   and, for pattern-only matches, the best must beat the runner-up by 0.15.
# """
# from __future__ import annotations

# import argparse
# import math
# import sys
# import threading
# import time
# from collections import Counter, deque
# from dataclasses import dataclass, field
# from typing import Callable, Deque, Iterator, List, Optional, Tuple

# import cv2
# import numpy as np
# from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode


# # ============================================================================ config
# @dataclass
# class QRConfig:
#     # ---- camera: Arducam B0332 (OV9281) ----
#     width: int = 1280
#     height: int = 800
#     fps: int = 30
#     exposure_abs: Optional[int] = None      # manual exposure, V4L2 units (100 us); None = auto
#     hfov_deg: float = 70.0                  # datasheet, nominal
#     sensor_w_mm: float = 3.84               # 1280 px x 3 um
#     calib: Optional[Tuple[float, float, float, float]] = None   # (fx, fy, cx, cy) from calibrate_camera.py
#     latency_s: float = 0.05                 # exposure + USB + MJPG decode; measure it

#     # ---- the printed QR ----
#     qr_size_m: float = 1.0                  # QR modules only
#     border_m: float = 0.0762                # 3 inch white border on each side

#     # ---- white-box finder ----
#     box_scale: float = 0.25
#     white_min: int = 40                     # absolute floor for any first-pass threshold
#     white_levels: Tuple[float, ...] = (0.85, 0.70, 0.55, 0.42, 0.30)   # first-pass thresholds, x brightest 0.1 % of the frame
#     max_fits_per_level: int = 12            # cap on local fits per threshold level (CPU)
#     box_squareness: float = 0.78            # strict tests, applied to the locally re-fitted box
#     box_fill_min: float = 0.80
#     box_local_contrast: float = 35.0        # white level minus ground level needed for a local fit
#     box_size_tol: float = 0.40              # +-40 % of the size expected at this altitude
#     box_min_px: int = 30                    # used only when the altitude is unknown
#     box_max_px: int = 900
#     inner_contrast_min: float = 70.0
#     inner_dark_range: Tuple[float, float] = (0.15, 0.75)

#     # ---- decoding ----
#     fullframe_every: int = 4
#     crop_margin: float = 0.25

#     # ---- pattern matching ----
#     patch: int = 128                        # canonical square, px
#     mask_frac: float = 0.34                 # corner squares masked out (finder patterns)
#     pattern_blur: float = 2.0               # px at canonical size
#     shift_margin: int = 10                  # +- px of misregistration tolerated
#     match_threshold: float = 0.80           # undecoded candidate vs reference
#     conflict_threshold: float = 0.85        # decoded candidate with DIFFERENT text vs reference
#     match_margin: float = 0.15              # best must beat the runner-up by this (pattern-only matches)
#     same_threshold: float = 0.85            # "same code" test used while building the start reference

#     # ---- voting ----
#     start_window: int = 5
#     start_needed: int = 3
#     confirm_frames: int = 3                 # target seen in this many frames ...
#     confirm_window_s: float = 1.0           # ... within this time

#     # ---- derived ----
#     @property
#     def border_frac(self) -> float:
#         return self.border_m / self.qr_size_m

#     @property
#     def box_m(self) -> float:
#         return self.qr_size_m + 2.0 * self.border_m

#     @property
#     def fx(self) -> float:
#         return self.calib[0] if self.calib else (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

#     def expected_box_px(self, altitude_m: float) -> float:
#         return self.box_m * self.fx / altitude_m

#     def expected_qr_px(self, altitude_m: float) -> float:
#         return self.qr_size_m * self.fx / altitude_m


# # ============================================================================ data
# @dataclass(eq=False)
# class Candidate:
#     text: Optional[str]                     # decoded text, None = undecoded white box
#     quad: np.ndarray                        # inner QR corners, full-frame px, float32 (4,2), clockwise
#     method: str                             # crop | rescue | direct | box
#     frame_wh: Tuple[int, int]
#     patch: Optional[np.ndarray] = None      # canonical normalised patch
#     score: float = 0.0                      # pattern score against the reference
#     verdict: str = ""                       # text | pattern | conflict | "" (not accepted)
#     t: float = field(default_factory=time.time)

#     @property
#     def center(self) -> np.ndarray:
#         return self.quad.mean(axis=0)

#     @property
#     def size_px(self) -> float:
#         q = self.quad
#         return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


# @dataclass
# class Reference:
#     """The start QR as the matcher knows it."""
#     patch: np.ndarray
#     text: Optional[str] = None
#     size_px: float = 0.0

#     def save(self, path: str) -> None:
#         np.savez(path, patch=self.patch, text=np.array(self.text or ""), has_text=np.array(self.text is not None),
#                  size_px=np.array(self.size_px))

#     @staticmethod
#     def load(path: str) -> "Reference":
#         d = np.load(path, allow_pickle=False)
#         text = str(d["text"]) if bool(d["has_text"]) else None
#         return Reference(d["patch"].astype(np.float32), text, float(d["size_px"]))


# Detection = Tuple[float, float, float]      # (u, v, capture_time)


# # ============================================================================ geometry helpers
# def to_gray(frame: np.ndarray) -> np.ndarray:
#     return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# def order_quad(pts) -> np.ndarray:
#     """4 corners, clockwise in image coordinates (start corner arbitrary: matching tries 4 rotations)."""
#     p = np.asarray(pts, np.float32).reshape(-1, 2)
#     if len(p) != 4:
#         p = cv2.boxPoints(cv2.minAreaRect(p))
#     c = p.mean(axis=0)
#     ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
#     return p[np.argsort(ang)].astype(np.float32)


# def expand_quad(q: np.ndarray, f: float) -> np.ndarray:
#     c = q.mean(axis=0)
#     return (c + (q - c) * f).astype(np.float32)


# def quad_side(q: np.ndarray) -> float:
#     return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


# # ============================================================================ pattern matching
# def canonical_patch(gray: np.ndarray, quad: np.ndarray, cfg: QRConfig) -> np.ndarray:
#     """Rectify the QR to a cfg.patch square and normalise local contrast (illumination independent)."""
#     K = cfg.patch
#     q = order_quad(quad)
#     side = quad_side(q)
#     h, w = gray.shape[:2]
#     x0, y0 = np.maximum(q.min(axis=0).astype(int) - 4, 0)
#     x1, y1 = np.minimum(q.max(axis=0).astype(int) + 5, [w, h])
#     roi = gray[y0:y1, x0:x1]
#     if roi.size == 0:
#         return np.zeros((K, K), np.float32)
#     if side > 1.3 * K:                       # anti-alias before shrinking
#         roi = cv2.GaussianBlur(roi, (0, 0), 0.45 * side / K)
#     dst = np.array([[0, 0], [K - 1, 0], [K - 1, K - 1], [0, K - 1]], np.float32)
#     m = cv2.getPerspectiveTransform(q - np.array([x0, y0], np.float32), dst)
#     p = cv2.warpPerspective(roi, m, (K, K), flags=cv2.INTER_LINEAR).astype(np.float32)
#     bg = cv2.GaussianBlur(p, (0, 0), K / 6.0)
#     p = p - bg
#     sd = np.sqrt(cv2.GaussianBlur(p * p, (0, 0), K / 6.0)) + 1e-3
#     return np.clip(p / sd, -2.5, 2.5)


# class PatternMatcher:
#     """Scores any patch against one reference. Templates are prepared once."""

#     def __init__(self, reference_patch: np.ndarray, cfg: QRConfig):
#         self.cfg = cfg
#         K, m = cfg.patch, cfg.shift_margin
#         rb = cv2.GaussianBlur(reference_patch, (0, 0), cfg.pattern_blur) if cfg.pattern_blur else reference_patch
#         self.templates = [np.ascontiguousarray(np.rot90(rb, k)[m:K - m, m:K - m]) for k in range(4)]
#         n = K - 2 * m
#         mask = np.ones((n, n), np.float32)
#         c = max(0, int(round(cfg.mask_frac * K)) - m)
#         if c > 0:
#             mask[:c, :c] = 0
#             mask[:c, -c:] = 0
#             mask[-c:, :c] = 0
#             mask[-c:, -c:] = 0
#         self.mask = mask

#     def score(self, patch: np.ndarray) -> float:
#         cb = cv2.GaussianBlur(patch, (0, 0), self.cfg.pattern_blur) if self.cfg.pattern_blur else patch
#         best = -1.0
#         for t in self.templates:
#             r = cv2.matchTemplate(cb, t, cv2.TM_CCORR_NORMED, mask=self.mask)
#             v = float(np.nanmax(r)) if r.size else -1.0
#             if np.isfinite(v) and v > best:
#                 best = v
#         return best


# def pattern_score(a: np.ndarray, b: np.ndarray, cfg: QRConfig) -> float:
#     return PatternMatcher(a, cfg).score(b)


# # ============================================================================ detector
# class QRDetector:
#     """White boxes -> decode inside -> undecoded boxes kept as candidates (+ whole-frame safety net)."""

#     def __init__(self, cfg: Optional[QRConfig] = None):
#         self.cfg = cfg or QRConfig()
#         self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#         self._n = 0

#     # ---- low level ----
#     @staticmethod
#     def _zbar(img: np.ndarray):
#         return zbar_decode(img, symbols=[ZBarSymbol.QRCODE])

#     def _make(self, results, method: str, gray: np.ndarray, scale=1.0, ox=0.0, oy=0.0) -> List[Candidate]:
#         h, w = gray.shape[:2]
#         out = []
#         for r in results:
#             poly = np.array([[p.x, p.y] for p in r.polygon], np.float32)
#             if len(poly) < 3:
#                 continue
#             quad = (order_quad(poly) / scale + np.array([ox, oy], np.float32)).astype(np.float32)
#             c = Candidate(r.data.decode("utf-8", errors="replace"), quad, method, (w, h))
#             c.patch = canonical_patch(gray, quad, self.cfg)
#             out.append(c)
#         return out

#     @staticmethod
#     def _duplicate(c: Candidate, others: List[Candidate]) -> bool:
#         return any(np.linalg.norm(c.center - o.center) < 0.5 * max(o.size_px, c.size_px, 1.0) for o in others)

#     # ---- white box ----
#     def _inner(self, box: np.ndarray) -> np.ndarray:
#         return expand_quad(box, 1.0 / (1.0 + 2.0 * self.cfg.border_frac))

#     def _looks_like_qr(self, gray: np.ndarray, box: np.ndarray) -> bool:
#         c = self.cfg
#         inner = self._inner(box)
#         h, w = gray.shape[:2]
#         x0, y0 = np.maximum(inner.min(axis=0).astype(int), 0)
#         x1, y1 = np.minimum(inner.max(axis=0).astype(int), [w, h])
#         crop = gray[y0:y1:2, x0:x1:2]
#         if crop.size < 100:
#             return False
#         lo, hi = np.percentile(crop, (5, 95))
#         if hi - lo < c.inner_contrast_min:
#             return False
#         dark = float((crop < (lo + hi) / 2.0).mean())
#         return c.inner_dark_range[0] <= dark <= c.inner_dark_range[1]

#     def _fit_box(self, gray: np.ndarray, coarse: np.ndarray, exp: Optional[float]) -> Optional[np.ndarray]:
#         """
#         Stage 2: re-fit the white box at full resolution with a LOCAL threshold halfway between this
#         blob's own white level and the ground level around it. Immune to shade (both levels drop) and to
#         bright ground merging into the blob. Returns the 4 corners, or None if it is not a clean square.
#         """
#         c = self.cfg
#         h, w = gray.shape[:2]
#         side0 = quad_side(coarse)
#         pad = 0.15 * side0
#         x0, y0 = np.maximum(coarse.min(axis=0) - pad, 0).astype(int)
#         x1, y1 = np.minimum(coarse.max(axis=0) + pad, [w, h]).astype(int)
#         crop = gray[y0:y1, x0:x1]
#         if crop.size == 0:
#             return None
#         inside = np.zeros(crop.shape, np.uint8)
#         hull = cv2.convexHull((expand_quad(coarse, 1.12) - np.array([x0, y0], np.float32)).astype(np.int32))
#         cv2.fillConvexPoly(inside, hull, 255)
#         ring = crop[inside == 0]
#         ground = float(np.median(ring)) if ring.size > 50 else float(np.percentile(crop, 20))
#         white = float(np.percentile(crop[inside > 0], 98)) if inside.any() else float(np.percentile(crop, 98))
#         if white - ground < c.box_local_contrast:
#             return None
#         m = (crop > 0.5 * (white + ground)).astype(np.uint8) * 255
#         cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         c0 = coarse.mean(axis=0) - np.array([x0, y0], np.float32)
#         best, best_d = None, 1e9
#         for cnt in cnts:
#             if cv2.contourArea(cnt) < 0.3 * side0 * side0:
#                 continue
#             rect = cv2.minAreaRect(cnt)
#             bw, bh = rect[1]
#             if min(bw, bh) < 2 or min(bw, bh) / max(bw, bh) < c.box_squareness:
#                 continue
#             if cv2.contourArea(cnt) / (bw * bh) < c.box_fill_min:
#                 continue
#             side = 0.5 * (bw + bh)
#             if exp:
#                 if abs(side - exp) > c.box_size_tol * exp:
#                     continue
#             elif not (c.box_min_px <= side <= c.box_max_px):
#                 continue
#             d = float(np.linalg.norm(np.array(rect[0]) - c0))
#             if d < best_d and d < 0.25 * side0:
#                 best, best_d = rect, d
#         if best is None:
#             return None
#         return order_quad(cv2.boxPoints(best)) + np.array([x0, y0], np.float32)

#     def _white_boxes(self, gray: np.ndarray, altitude_m: Optional[float]) -> List[np.ndarray]:
#         """
#         Stage 1 (coarse, 1/4 scale): ladder of thresholds relative to the brightest paper in view; blobs of
#         plausible size and squareness go on. Stage 2 (_fit_box, full resolution): strict local fit.
#         A QR in sun is found on the top rungs, one in shade on a lower rung, and a blob merged with bright
#         ground on one rung is clean on another.
#         """
#         c = self.cfg
#         s = c.box_scale
#         # max-pool (not area-average): a 9 px wide border averaged into 4x4 blocks loses a third of its
#         # brightness and breaks up in dim light; max-pooling keeps thin white features at full level.
#         # (the blob grows by <= 3 px; stage 2 recomputes the exact geometry at full resolution)
#         k = int(round(1.0 / s))
#         small = cv2.dilate(gray, np.ones((k, k), np.uint8))[::k, ::k]
#         s = 1.0 / k
#         top = float(np.percentile(small, 99.9))
#         exp = c.expected_box_px(altitude_m) if altitude_m and altitude_m > 1.0 else None
#         boxes: List[np.ndarray] = []
#         tried: List[np.ndarray] = []
#         for rel in c.white_levels:
#             thr = max(c.white_min, rel * top)
#             mask = cv2.threshold(small, thr, 255, cv2.THRESH_BINARY)[1]
#             cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             blobs = []
#             for cnt in cnts:
#                 if len(cnt) < 4:
#                     continue
#                 rect = cv2.minAreaRect(cnt)
#                 bw, bh = rect[1]
#                 if min(bw, bh) < 2:
#                     continue
#                 side = 0.5 * (bw + bh) / s
#                 if exp:                                    # loose gate: a merged blob is bigger than a clean box
#                     if not (0.5 * exp <= side <= 1.9 * exp):
#                         continue
#                 elif not (c.box_min_px <= side <= c.box_max_px * 1.5):
#                     continue
#                 sq = min(bw, bh) / max(bw, bh)
#                 fill = cv2.contourArea(cnt) / (bw * bh)
#                 if sq < 0.55 or fill < 0.55:
#                     continue
#                 if any(np.linalg.norm(np.array(rect[0]) / s - t) < 0.3 * side for t in tried):
#                     continue                               # this spot already has a box
#                 blobs.append((sq * fill, rect, side))
#             for _, rect, side in sorted(blobs, key=lambda b: -b[0])[:c.max_fits_per_level]:
#                 box = self._fit_box(gray, order_quad(cv2.boxPoints(rect) / s), exp)
#                 if box is not None and self._looks_like_qr(gray, box):
#                     boxes.append(box)
#                     tried.append(box.mean(axis=0))
#         return boxes

#     # ---- decoding inside a box ----
#     def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[Candidate]:
#         h, w = gray.shape[:2]
#         x0, y0 = quad.min(axis=0)
#         x1, y1 = quad.max(axis=0)
#         margin = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
#         x0, y0 = int(max(0, x0 - margin)), int(max(0, y0 - margin))
#         x1, y1 = int(min(w, x1 + margin)), int(min(h, y1 + margin))
#         base = gray[y0:y1, x0:x1]
#         if base.size == 0:
#             return []

#         # Try the native crop first, then a 2x crop for small targets. Keep the
#         # scale for mapping pyzbar's crop coordinates back to full-frame pixels.
#         initial_scale = 2.0 if max(base.shape) < 240 else 1.0
#         scales = [initial_scale]
#         if initial_scale != 3.0:
#             scales.append(3.0)  # extra rescue for small/blurred QRs at altitude

#         for scale in scales:
#             crop = (base if scale == 1.0 else
#                     cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

#             # Original grayscale decode.
#             res = self._zbar(crop)
#             if res:
#                 return self._make(res, "crop", gray, scale, x0, y0)

#             # Contrast/sharpen rescue, followed by Otsu thresholding.
#             clahe = self._clahe.apply(crop)
#             sharp = cv2.addWeighted(
#                 clahe, 1.6,
#                 cv2.GaussianBlur(clahe, (0, 0), 2.0),
#                 -0.6, 0,
#             )
#             variants = (
#                 sharp,
#                 cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
#                 cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
#             )
#             for variant in variants:
#                 res = self._zbar(variant)
#                 if res:
#                     return self._make(res, "rescue", gray, scale, x0, y0)

#             # OpenCV's independent QR decoder is a final fallback; its detected
#             # geometry is deliberately not used because the white-box geometry
#             # is more stable for the navigation pixel centre.
#             try:
#                 text, points, _ = cv2.QRCodeDetector().detectAndDecode(crop)
#             except cv2.error:
#                 text, points = "", None
#             if text and points is not None:
#                 candidate = Candidate(text, quad.copy(), "rescue", (w, h))
#                 candidate.patch = canonical_patch(gray, candidate.quad, self.cfg)
#                 return [candidate]

#         return []

#     # ---- one frame ----
#     def scan(self, gray: np.ndarray, altitude_m: Optional[float] = None,
#              full: Optional[bool] = None) -> List[Candidate]:
#         """All QR candidates in this frame (decoded or not). altitude_m = height above the ground."""
#         self._n += 1
#         h, w = gray.shape[:2]
#         found: List[Candidate] = []
#         for box in self._white_boxes(gray, altitude_m):
#             inner = self._inner(box)
#             probe = Candidate(None, inner, "box", (w, h))
#             if self._duplicate(probe, found):
#                 continue
#             decoded = self._decode_crop(gray, inner)
#             if decoded:
#                 found += [d for d in decoded if not self._duplicate(d, found)]
#             else:
#                 probe.patch = canonical_patch(gray, inner, self.cfg)
#                 found.append(probe)
#         if full is None:
#             full = self._n % self.cfg.fullframe_every == 0
#         if full:
#             found += [d for d in self._make(self._zbar(gray), "direct", gray) if not self._duplicate(d, found)]
#         return found


# # ============================================================================ start QR -> reference
# class StartScanner:
#     """
#     Call once per frame while looking at the start QR (~5 m). Returns a Reference once the same code
#     has been seen in `start_needed` of the last `start_window` frames. 'Same' = same decoded text OR
#     same pattern, so it also works when the code never decodes.
#     """

#     def __init__(self, cfg: Optional[QRConfig] = None, detector: Optional[QRDetector] = None):
#         self.cfg = cfg or QRConfig()
#         self.det = detector or QRDetector(self.cfg)
#         self.window: Deque[Optional[Candidate]] = deque(maxlen=self.cfg.start_window)

#     def _same(self, a: Candidate, b: Candidate) -> bool:
#         if a.text is not None and b.text is not None and a.text.strip() == b.text.strip():
#             return True
#         return pattern_score(a.patch, b.patch, self.cfg) >= self.cfg.same_threshold

#     def update(self, gray: np.ndarray, altitude_m: Optional[float] = None) -> Optional[Reference]:
#         cands = [c for c in self.det.scan(gray, altitude_m) if c.patch is not None]
#         self.window.append(max(cands, key=lambda c: c.size_px) if cands else None)
#         seen = [c for c in self.window if c is not None]
#         if len(seen) < self.cfg.start_needed:
#             return None
#         best_group: List[Candidate] = []
#         for a in seen:
#             group = [b for b in seen if b is a or self._same(a, b)]
#             if len(group) > len(best_group):
#                 best_group = group
#         if len(best_group) < self.cfg.start_needed:
#             return None
#         texts = Counter(c.text.strip() for c in best_group if c.text is not None)
#         text = texts.most_common(1)[0][0] if texts else None
#         pick = max(best_group, key=lambda c: (c.text is not None, c.size_px))
#         return Reference(pick.patch.copy(), text, pick.size_px)


# # ============================================================================ target decision
# class TargetMatcher:
#     """Decides which candidate in a frame (if any) is the target. Text is evidence, pattern decides."""

#     def __init__(self, reference: Reference, cfg: Optional[QRConfig] = None):
#         self.cfg = cfg or QRConfig()
#         self.ref = reference
#         self.pm = PatternMatcher(reference.patch, self.cfg)
#         self._ref_text = reference.text.strip() if reference.text is not None else None

#     def match(self, cands: List[Candidate]) -> Optional[Candidate]:
#         c = self.cfg
#         for k in cands:
#             k.verdict = ""
#             k.score = self.pm.score(k.patch) if k.patch is not None else -1.0
#             if self._ref_text is not None and k.text is not None and k.text.strip() == self._ref_text:
#                 k.verdict = "text"
#             elif k.text is None and k.score >= c.match_threshold:
#                 k.verdict = "pattern"
#             elif k.text is not None and k.score >= c.conflict_threshold:
#                 k.verdict = "conflict"            # decoded to other text, but the pattern says it is the same code
#         accepted = [k for k in cands if k.verdict]
#         if not accepted:
#             return None

#         # An exact text match wins. Otherwise, compare only other accepted
#         # pattern-based candidates, not unrelated low-scoring decoys.
#         best = max(accepted, key=lambda k: (k.verdict == "text", k.score))
#         if best.verdict != "text":
#             runners_up = [
#                 k.score for k in accepted
#                 if k is not best and k.verdict in ("pattern", "conflict")
#             ]
#             if runners_up and best.score - max(runners_up) < c.match_margin:
#                 return None                    # ambiguous visual match: do not guess
#         return best


# # ============================================================================ tracking
# class TargetTracker:
#     """Per frame: scan -> decide -> publish (u, v, capture time). Keeps recent detections for confirmation."""

#     def __init__(self, detector: QRDetector, matcher: TargetMatcher,
#                  altitude_fn: Callable[[], Optional[float]] = lambda: None,
#                  publish: Optional[Callable[[float, float, float], None]] = None,
#                  undistort: Optional[Callable[[float, float], Tuple[float, float]]] = None):
#         self.det, self.matcher, self.altitude_fn = detector, matcher, altitude_fn
#         self.publish, self.undistort = publish, undistort
#         self.cfg = matcher.cfg
#         self.recent: Deque[Detection] = deque(maxlen=20)
#         self.last: Optional[Candidate] = None
#         self.frames = self.matches = 0

#     def process(self, gray: np.ndarray, t_capture: float) -> Optional[Detection]:
#         self.frames += 1
#         cands = self.det.scan(gray, self.altitude_fn())
#         hit = self.matcher.match(cands)
#         self.last = hit
#         if hit is None:
#             return None
#         u, v = (float(x) for x in hit.center)
#         if self.undistort:
#             u, v = self.undistort(u, v)
#         d = (u, v, t_capture - self.cfg.latency_s)
#         self.matches += 1
#         self.recent.append(d)
#         if self.publish:
#             self.publish(*d)
#         return d

#     def confirmed(self) -> Optional[List[Detection]]:
#         now = time.time()
#         fresh = [d for d in self.recent if now - d[2] < self.cfg.confirm_window_s]
#         return fresh[-self.cfg.confirm_frames:] if len(fresh) >= self.cfg.confirm_frames else None


# # ============================================================================ camera
# class FrameGrabber:
#     """
#     Reads the camera on a thread and always hands out the NEWEST frame together with its capture time
#     (stamped as soon as cap.read() returns). Forces MJPG 1280x800, buffer 1.
#     """

#     def __init__(self, src=0, cfg: Optional[QRConfig] = None):
#         cfg = cfg or QRConfig()
#         backend = cv2.CAP_V4L2 if (isinstance(src, int) and sys.platform.startswith("linux")) else cv2.CAP_ANY
#         self.cap = cv2.VideoCapture(src, backend)
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Cannot open camera {src}")
#         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # before the size
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
#         self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         if cfg.exposure_abs is not None:
#             self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                         # 1 = manual (V4L2)
#             self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_abs)
#         self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         if self.size != (cfg.width, cfg.height):
#             print(f"[WARN] camera delivers {self.size[0]}x{self.size[1]}, wanted {cfg.width}x{cfg.height}: "
#                   f"the camera model and the size filter assume {cfg.width}x{cfg.height}", file=sys.stderr)
#         self._lock, self._new, self._stop = threading.Lock(), threading.Event(), threading.Event()
#         self._frame, self._idx, self._t = None, 0, 0.0
#         self._thread = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()

#     def _loop(self) -> None:
#         while not self._stop.is_set():
#             ok, frame = self.cap.read()
#             t = time.time()
#             if not ok:
#                 time.sleep(0.01)
#                 continue
#             with self._lock:
#                 self._frame, self._idx, self._t = frame, self._idx + 1, t
#             self._new.set()

#     def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray, float]]:
#         if not self._new.wait(timeout):
#             return None
#         self._new.clear()
#         with self._lock:
#             return self._idx, self._frame, self._t

#     def close(self) -> None:
#         self._stop.set()
#         self._thread.join(timeout=1.0)
#         self.cap.release()

#     def __enter__(self):
#         return self

#     def __exit__(self, *exc):
#         self.close()


# class DetectionWorker(threading.Thread):
#     """grab -> track continuously (also while the flight code is busy flying)."""

#     def __init__(self, grabber, tracker: TargetTracker):
#         super().__init__(daemon=True)
#         self.grabber, self.tracker = grabber, tracker
#         self.errors = 0
#         self._stop_evt = threading.Event()

#     def run(self) -> None:
#         while not self._stop_evt.is_set():
#             item = self.grabber.read(timeout=0.5)
#             if item is None:
#                 continue
#             _, frame, t = item
#             try:
#                 self.tracker.process(to_gray(frame), t)
#             except Exception as e:                 # the detector must never die mid-mission
#                 self.errors += 1
#                 print(f"[aerothon_qr] detector error #{self.errors}: {e!r}", file=sys.stderr)

#     def stop(self) -> None:
#         self._stop_evt.set()


# def make_undistorter(K: np.ndarray, dist: np.ndarray):
#     """(u, v) from the real lens -> (u, v) in the ideal pinhole image the flight code assumes."""
#     def undistort(u: float, v: float) -> Tuple[float, float]:
#         p = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist, P=K)
#         return float(p[0, 0, 0]), float(p[0, 0, 1])
#     return undistort


# def load_calibration(path: str):
#     """calib.npz from calibrate_camera.py -> ((fx, fy, cx, cy), K, dist)."""
#     d = np.load(path)
#     K, dist = d["K"], d["dist"]
#     return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])), K, dist


# def video_frames(path: str) -> Iterator[Tuple[int, np.ndarray, float]]:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video {path}")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         idx += 1
#         yield idx, frame, idx / fps
#     cap.release()


# # ============================================================================ test harness
# def _draw(frame: np.ndarray, cands: List[Candidate], target: Optional[Candidate]) -> np.ndarray:
#     img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     for k in cands:
#         col = (0, 255, 0) if k is target else ((0, 200, 0) if k.text is not None else (0, 200, 255))
#         cv2.polylines(img, [k.quad.astype(np.int32)], True, col, 2)
#         label = f"{k.method} {k.text[:14] if k.text else '-'} {k.score:.2f} {k.verdict}"
#         cv2.putText(img, label, tuple(k.quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
#     return img


# def run(args) -> None:
#     cfg = QRConfig(exposure_abs=args.exposure, qr_size_m=args.qr_size)
#     if args.calib:
#         cfg.calib = load_calibration(args.calib)[0]
#     det = QRDetector(cfg)
#     scanner = StartScanner(cfg, det) if args.mode == "start" else None
#     tracker = None
#     if args.mode == "target":
#         if not args.reference:
#             sys.exit("--mode target needs --reference FILE (made with --mode start --save-reference FILE)")
#         tracker = TargetTracker(det, TargetMatcher(Reference.load(args.reference), cfg), lambda: args.altitude)

#     print(f"f={cfg.fx:.0f}px  expected white box at {args.altitude:.0f} m = {cfg.expected_box_px(args.altitude):.0f}px, "
#           f"QR = {cfg.expected_qr_px(args.altitude):.0f}px")
#     grabber = FrameGrabber(args.camera, cfg) if args.camera is not None else None
#     frames = iter(lambda: grabber.read(), None) if grabber else video_frames(args.video)
#     times, n, with_hit = [], 0, 0
#     t_begin = time.time()
#     try:
#         for item in frames:
#             if item is None:
#                 break
#             idx, frame, t_cap = item
#             t0 = time.perf_counter()
#             gray = to_gray(frame)
#             target = None
#             if args.mode == "raw":
#                 cands = det.scan(gray, args.altitude, full=True)
#             elif args.mode == "start":
#                 ref = scanner.update(gray, args.altitude)
#                 cands = [c for c in scanner.window if c is not None][-1:]
#                 if ref is not None:
#                     print(f"START QR CONFIRMED at frame {idx}: text={ref.text!r} size={ref.size_px:.0f}px")
#                     if args.save_reference:
#                         ref.save(args.save_reference)
#                         print(f"reference saved to {args.save_reference}")
#                     break
#             else:
#                 tracker.process(gray, t_cap)
#                 target = tracker.last
#                 cands = [target] if target else []
#                 if target and tracker.confirmed():
#                     u, v = target.center
#                     print(f"TARGET CONFIRMED at frame {idx}: pixel=({u:.0f},{v:.0f}) {target.verdict} "
#                           f"score={target.score:.2f} text={target.text!r}")
#                     break
#             times.append((time.perf_counter() - t0) * 1000.0)
#             n += 1
#             with_hit += bool(cands)
#             if args.verbose:
#                 for k in cands:
#                     print(f"[{idx}] {k.method:6s} size={k.size_px:5.0f}px text={k.text!r} score={k.score:.2f} {k.verdict}")
#             if args.show:
#                 cv2.imshow("qr", cv2.resize(_draw(frame, cands, target), None, fx=0.5, fy=0.5))
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break
#             if args.max_seconds and time.time() - t_begin > args.max_seconds:
#                 break
#     finally:
#         if grabber:
#             grabber.close()
#         if args.show:
#             cv2.destroyAllWindows()
#     if n:
#         t = np.array(times)
#         print(f"\nframes={n} frames_with_QR={with_hit} ({100 * with_hit / n:.0f}%)  "
#               f"ms/frame p50={np.percentile(t, 50):.0f} p95={np.percentile(t, 95):.0f} max={t.max():.0f}  "
#               f"~{1000.0 / max(t.mean(), 1e-6):.1f} FPS")


# def main() -> None:
#     ap = argparse.ArgumentParser(description="AeroTHON QR vision test harness")
#     src = ap.add_mutually_exclusive_group(required=True)
#     src.add_argument("--camera", type=int)
#     src.add_argument("--video", help="recording at full 1280x800 (no resizing)")
#     ap.add_argument("--mode", choices=["raw", "start", "target"], default="raw")
#     ap.add_argument("--reference", help="reference file for --mode target")
#     ap.add_argument("--save-reference", help="where --mode start writes the reference")
#     ap.add_argument("--altitude", type=float, default=10.0, help="m above ground (size filter)")
#     ap.add_argument("--qr-size", type=float, default=1.0)
#     ap.add_argument("--calib", help="calib.npz from calibrate_camera.py")
#     ap.add_argument("--exposure", type=int, default=None, help="manual exposure (V4L2 units, 100 us)")
#     ap.add_argument("--max-seconds", type=float, default=0.0)
#     ap.add_argument("--show", action="store_true")
#     ap.add_argument("--verbose", action="store_true")
#     run(ap.parse_args())


# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # aerothon_qr.py - QR vision for the AeroTHON delivery mission.  (vision only: no MAVLink here)

# # Camera : Arducam B0332 = OV9281 mono global shutter, 1280x800, 3 um pixels, 70 deg (H) lens,
# #          f = (1280/2)/tan(35 deg) = 914 px.  Calibrate it (calibrate_camera.py) before flying.
# # Target : black-on-white QR, 1.00 m x 1.00 m, plus a 3 inch (0.0762 m) white border on every side
# #          -> white box = 1.1524 m.   1 m QR = 91 px at 10 m, 183 px at 5 m  (box 105 px / 211 px).

# # Pipeline (white-box logic, as in qr_whiteboxcheck.py)
# #   1. find white squares (the border) on a 1/4 scale image, filtered by the size they MUST have at
# #      the current altitude, squareness, fill, and "dark/light pattern inside"
# #   2. refine the box at full resolution, decode inside it (pyzbar; CLAHE/Otsu rescue)
# #   3. boxes that do not decode are kept as undecoded candidates (nothing is lost)
# #   4. whole-frame decode every few frames as a safety net

# # Matching the target (the part that does not trust decoded text)
# #   The start QR may not decode to the same text as the target, or the target may be too small to
# #   decode at all, so every candidate is also compared by PATTERN: the QR is rectified to a canonical
# #   square, locally contrast-normalised, the three finder-pattern corners are masked out (every QR
# #   shares those, they would inflate decoy scores), and the correlation is taken over 4 rotations and
# #   a small shift range.  Measured on synthetic frames: same code 0.92-0.99 (10 m ... 5 m, blur,
# #   noise, tilt), different codes <= 0.59 even when the text differs by one character.

# #   accept if   text equal                              (strong)
# #          or   undecoded      and pattern >= 0.80
# #          or   decoded, text differs, pattern >= 0.85  (text is not trusted; pattern must be convincing)
# #   and, for pattern-only matches, the best must beat the runner-up by 0.15.
# # """
# # from __future__ import annotations

# # import argparse
# # import math
# # import sys
# # import threading
# # import time
# # from collections import Counter, deque
# # from dataclasses import dataclass, field
# # from typing import Callable, Deque, Iterator, List, Optional, Tuple

# # import cv2
# # import numpy as np
# # from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode


# # # ============================================================================ config
# # @dataclass
# # class QRConfig:
# #     # ---- camera: Arducam B0332 (OV9281) ----
# #     width: int = 1280
# #     height: int = 800
# #     fps: int = 30
# #     exposure_abs: Optional[int] = None      # manual exposure, V4L2 units (100 us); None = auto
# #     hfov_deg: float = 70.0                  # datasheet, nominal
# #     sensor_w_mm: float = 3.84               # 1280 px x 3 um
# #     calib: Optional[Tuple[float, float, float, float]] = None   # (fx, fy, cx, cy) from calibrate_camera.py
# #     latency_s: float = 0.05                 # exposure + USB + MJPG decode; measure it

# #     # ---- the printed QR ----
# #     qr_size_m: float = 1.0                  # QR modules only
# #     border_m: float = 0.0762                # 3 inch white border on each side

# #     # ---- white-box finder ----
# #     box_scale: float = 0.25
# #     white_min: int = 40                     # absolute floor for any first-pass threshold
# #     white_levels: Tuple[float, ...] = (0.85, 0.70, 0.55, 0.42, 0.30)   # first-pass thresholds, x brightest 0.1 % of the frame
# #     max_fits_per_level: int = 12            # cap on local fits per threshold level (CPU)
# #     box_squareness: float = 0.78            # strict tests, applied to the locally re-fitted box
# #     box_fill_min: float = 0.80
# #     box_local_contrast: float = 35.0        # white level minus ground level needed for a local fit
# #     box_size_tol: float = 0.40              # +-40 % of the size expected at this altitude
# #     box_min_px: int = 30                    # used only when the altitude is unknown
# #     box_max_px: int = 900
# #     inner_contrast_min: float = 70.0
# #     inner_dark_range: Tuple[float, float] = (0.15, 0.75)

# #     # ---- decoding ----
# #     fullframe_every: int = 4
# #     crop_margin: float = 0.25

# #     # ---- pattern matching ----
# #     patch: int = 128                        # canonical square, px
# #     mask_frac: float = 0.34                 # corner squares masked out (finder patterns)
# #     pattern_blur: float = 2.0               # px at canonical size
# #     shift_margin: int = 10                  # +- px of misregistration tolerated
# #     match_threshold: float = 0.80           # undecoded candidate vs reference
# #     conflict_threshold: float = 0.85        # decoded candidate with DIFFERENT text vs reference
# #     match_margin: float = 0.15              # best must beat the runner-up by this (pattern-only matches)
# #     same_threshold: float = 0.85            # "same code" test used while building the start reference

# #     # ---- voting ----
# #     start_window: int = 5
# #     start_needed: int = 3
# #     confirm_frames: int = 3                 # target seen in this many frames ...
# #     confirm_window_s: float = 1.0           # ... within this time

# #     # ---- derived ----
# #     @property
# #     def border_frac(self) -> float:
# #         return self.border_m / self.qr_size_m

# #     @property
# #     def box_m(self) -> float:
# #         return self.qr_size_m + 2.0 * self.border_m

# #     @property
# #     def fx(self) -> float:
# #         return self.calib[0] if self.calib else (self.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

# #     def expected_box_px(self, altitude_m: float) -> float:
# #         return self.box_m * self.fx / altitude_m

# #     def expected_qr_px(self, altitude_m: float) -> float:
# #         return self.qr_size_m * self.fx / altitude_m


# # # ============================================================================ data
# # @dataclass(eq=False)
# # class Candidate:
# #     text: Optional[str]                     # decoded text, None = undecoded white box
# #     quad: np.ndarray                        # inner QR corners, full-frame px, float32 (4,2), clockwise
# #     method: str                             # crop | rescue | direct | box
# #     frame_wh: Tuple[int, int]
# #     patch: Optional[np.ndarray] = None      # canonical normalised patch
# #     score: float = 0.0                      # pattern score against the reference
# #     verdict: str = ""                       # text | pattern | conflict | "" (not accepted)
# #     t: float = field(default_factory=time.time)

# #     @property
# #     def center(self) -> np.ndarray:
# #         return self.quad.mean(axis=0)

# #     @property
# #     def size_px(self) -> float:
# #         q = self.quad
# #         return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


# # @dataclass
# # class Reference:
# #     """The start QR as the matcher knows it."""
# #     patch: np.ndarray
# #     text: Optional[str] = None
# #     size_px: float = 0.0

# #     def save(self, path: str) -> None:
# #         np.savez(path, patch=self.patch, text=np.array(self.text or ""), has_text=np.array(self.text is not None),
# #                  size_px=np.array(self.size_px))

# #     @staticmethod
# #     def load(path: str) -> "Reference":
# #         d = np.load(path, allow_pickle=False)
# #         text = str(d["text"]) if bool(d["has_text"]) else None
# #         return Reference(d["patch"].astype(np.float32), text, float(d["size_px"]))


# # Detection = Tuple[float, float, float]      # (u, v, capture_time)


# # # ============================================================================ geometry helpers
# # def to_gray(frame: np.ndarray) -> np.ndarray:
# #     return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# # def order_quad(pts) -> np.ndarray:
# #     """4 corners, clockwise in image coordinates (start corner arbitrary: matching tries 4 rotations)."""
# #     p = np.asarray(pts, np.float32).reshape(-1, 2)
# #     if len(p) != 4:
# #         p = cv2.boxPoints(cv2.minAreaRect(p))
# #     c = p.mean(axis=0)
# #     ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
# #     return p[np.argsort(ang)].astype(np.float32)


# # def expand_quad(q: np.ndarray, f: float) -> np.ndarray:
# #     c = q.mean(axis=0)
# #     return (c + (q - c) * f).astype(np.float32)


# # def quad_side(q: np.ndarray) -> float:
# #     return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))


# # # ============================================================================ pattern matching
# # def canonical_patch(gray: np.ndarray, quad: np.ndarray, cfg: QRConfig) -> np.ndarray:
# #     """Rectify the QR to a cfg.patch square and normalise local contrast (illumination independent)."""
# #     K = cfg.patch
# #     q = order_quad(quad)
# #     side = quad_side(q)
# #     h, w = gray.shape[:2]
# #     x0, y0 = np.maximum(q.min(axis=0).astype(int) - 4, 0)
# #     x1, y1 = np.minimum(q.max(axis=0).astype(int) + 5, [w, h])
# #     roi = gray[y0:y1, x0:x1]
# #     if roi.size == 0:
# #         return np.zeros((K, K), np.float32)
# #     if side > 1.3 * K:                       # anti-alias before shrinking
# #         roi = cv2.GaussianBlur(roi, (0, 0), 0.45 * side / K)
# #     dst = np.array([[0, 0], [K - 1, 0], [K - 1, K - 1], [0, K - 1]], np.float32)
# #     m = cv2.getPerspectiveTransform(q - np.array([x0, y0], np.float32), dst)
# #     p = cv2.warpPerspective(roi, m, (K, K), flags=cv2.INTER_LINEAR).astype(np.float32)
# #     bg = cv2.GaussianBlur(p, (0, 0), K / 6.0)
# #     p = p - bg
# #     sd = np.sqrt(cv2.GaussianBlur(p * p, (0, 0), K / 6.0)) + 1e-3
# #     return np.clip(p / sd, -2.5, 2.5)


# # class PatternMatcher:
# #     """Scores any patch against one reference. Templates are prepared once."""

# #     def __init__(self, reference_patch: np.ndarray, cfg: QRConfig):
# #         self.cfg = cfg
# #         K, m = cfg.patch, cfg.shift_margin
# #         rb = cv2.GaussianBlur(reference_patch, (0, 0), cfg.pattern_blur) if cfg.pattern_blur else reference_patch
# #         self.templates = [np.ascontiguousarray(np.rot90(rb, k)[m:K - m, m:K - m]) for k in range(4)]
# #         n = K - 2 * m
# #         mask = np.ones((n, n), np.float32)
# #         c = max(0, int(round(cfg.mask_frac * K)) - m)
# #         if c > 0:
# #             mask[:c, :c] = 0
# #             mask[:c, -c:] = 0
# #             mask[-c:, :c] = 0
# #             mask[-c:, -c:] = 0
# #         self.mask = mask

# #     def score(self, patch: np.ndarray) -> float:
# #         cb = cv2.GaussianBlur(patch, (0, 0), self.cfg.pattern_blur) if self.cfg.pattern_blur else patch
# #         best = -1.0
# #         for t in self.templates:
# #             r = cv2.matchTemplate(cb, t, cv2.TM_CCORR_NORMED, mask=self.mask)
# #             v = float(np.nanmax(r)) if r.size else -1.0
# #             if np.isfinite(v) and v > best:
# #                 best = v
# #         return best


# # def pattern_score(a: np.ndarray, b: np.ndarray, cfg: QRConfig) -> float:
# #     return PatternMatcher(a, cfg).score(b)


# # # ============================================================================ detector
# # class QRDetector:
# #     """White boxes -> decode inside -> undecoded boxes kept as candidates (+ whole-frame safety net)."""

# #     def __init__(self, cfg: Optional[QRConfig] = None):
# #         self.cfg = cfg or QRConfig()
# #         self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
# #         self._n = 0

# #     # ---- low level ----
# #     @staticmethod
# #     def _zbar(img: np.ndarray):
# #         return zbar_decode(img, symbols=[ZBarSymbol.QRCODE])

# #     def _make(self, results, method: str, gray: np.ndarray, scale=1.0, ox=0.0, oy=0.0) -> List[Candidate]:
# #         h, w = gray.shape[:2]
# #         out = []
# #         for r in results:
# #             poly = np.array([[p.x, p.y] for p in r.polygon], np.float32)
# #             if len(poly) < 3:
# #                 continue
# #             quad = (order_quad(poly) / scale + np.array([ox, oy], np.float32)).astype(np.float32)
# #             c = Candidate(r.data.decode("utf-8", errors="replace"), quad, method, (w, h))
# #             c.patch = canonical_patch(gray, quad, self.cfg)
# #             out.append(c)
# #         return out

# #     @staticmethod
# #     def _duplicate(c: Candidate, others: List[Candidate]) -> bool:
# #         return any(np.linalg.norm(c.center - o.center) < 0.5 * max(o.size_px, c.size_px, 1.0) for o in others)

# #     # ---- white box ----
# #     def _inner(self, box: np.ndarray) -> np.ndarray:
# #         return expand_quad(box, 1.0 / (1.0 + 2.0 * self.cfg.border_frac))

# #     def _looks_like_qr(self, gray: np.ndarray, box: np.ndarray) -> bool:
# #         c = self.cfg
# #         inner = self._inner(box)
# #         h, w = gray.shape[:2]
# #         x0, y0 = np.maximum(inner.min(axis=0).astype(int), 0)
# #         x1, y1 = np.minimum(inner.max(axis=0).astype(int), [w, h])
# #         crop = gray[y0:y1:2, x0:x1:2]
# #         if crop.size < 100:
# #             return False
# #         lo, hi = np.percentile(crop, (5, 95))
# #         if hi - lo < c.inner_contrast_min:
# #             return False
# #         dark = float((crop < (lo + hi) / 2.0).mean())
# #         return c.inner_dark_range[0] <= dark <= c.inner_dark_range[1]

# #     def _fit_box(self, gray: np.ndarray, coarse: np.ndarray, exp: Optional[float]) -> Optional[np.ndarray]:
# #         """
# #         Stage 2: re-fit the white box at full resolution with a LOCAL threshold halfway between this
# #         blob's own white level and the ground level around it. Immune to shade (both levels drop) and to
# #         bright ground merging into the blob. Returns the 4 corners, or None if it is not a clean square.
# #         """
# #         c = self.cfg
# #         h, w = gray.shape[:2]
# #         side0 = quad_side(coarse)
# #         pad = 0.15 * side0
# #         x0, y0 = np.maximum(coarse.min(axis=0) - pad, 0).astype(int)
# #         x1, y1 = np.minimum(coarse.max(axis=0) + pad, [w, h]).astype(int)
# #         crop = gray[y0:y1, x0:x1]
# #         if crop.size == 0:
# #             return None
# #         inside = np.zeros(crop.shape, np.uint8)
# #         hull = cv2.convexHull((expand_quad(coarse, 1.12) - np.array([x0, y0], np.float32)).astype(np.int32))
# #         cv2.fillConvexPoly(inside, hull, 255)
# #         ring = crop[inside == 0]
# #         ground = float(np.median(ring)) if ring.size > 50 else float(np.percentile(crop, 20))
# #         white = float(np.percentile(crop[inside > 0], 98)) if inside.any() else float(np.percentile(crop, 98))
# #         if white - ground < c.box_local_contrast:
# #             return None
# #         m = (crop > 0.5 * (white + ground)).astype(np.uint8) * 255
# #         cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #         c0 = coarse.mean(axis=0) - np.array([x0, y0], np.float32)
# #         best, best_d = None, 1e9
# #         for cnt in cnts:
# #             if cv2.contourArea(cnt) < 0.3 * side0 * side0:
# #                 continue
# #             rect = cv2.minAreaRect(cnt)
# #             bw, bh = rect[1]
# #             if min(bw, bh) < 2 or min(bw, bh) / max(bw, bh) < c.box_squareness:
# #                 continue
# #             if cv2.contourArea(cnt) / (bw * bh) < c.box_fill_min:
# #                 continue
# #             side = 0.5 * (bw + bh)
# #             if exp:
# #                 if abs(side - exp) > c.box_size_tol * exp:
# #                     continue
# #             elif not (c.box_min_px <= side <= c.box_max_px):
# #                 continue
# #             d = float(np.linalg.norm(np.array(rect[0]) - c0))
# #             if d < best_d and d < 0.25 * side0:
# #                 best, best_d = rect, d
# #         if best is None:
# #             return None
# #         return order_quad(cv2.boxPoints(best)) + np.array([x0, y0], np.float32)

# #     def _white_boxes(self, gray: np.ndarray, altitude_m: Optional[float]) -> List[np.ndarray]:
# #         """
# #         Stage 1 (coarse, 1/4 scale): ladder of thresholds relative to the brightest paper in view; blobs of
# #         plausible size and squareness go on. Stage 2 (_fit_box, full resolution): strict local fit.
# #         A QR in sun is found on the top rungs, one in shade on a lower rung, and a blob merged with bright
# #         ground on one rung is clean on another.
# #         """
# #         c = self.cfg
# #         s = c.box_scale
# #         # max-pool (not area-average): a 9 px wide border averaged into 4x4 blocks loses a third of its
# #         # brightness and breaks up in dim light; max-pooling keeps thin white features at full level.
# #         # (the blob grows by <= 3 px; stage 2 recomputes the exact geometry at full resolution)
# #         k = int(round(1.0 / s))
# #         small = cv2.dilate(gray, np.ones((k, k), np.uint8))[::k, ::k]
# #         s = 1.0 / k
# #         top = float(np.percentile(small, 99.9))
# #         exp = c.expected_box_px(altitude_m) if altitude_m and altitude_m > 1.0 else None
# #         boxes: List[np.ndarray] = []
# #         tried: List[np.ndarray] = []
# #         for rel in c.white_levels:
# #             thr = max(c.white_min, rel * top)
# #             mask = cv2.threshold(small, thr, 255, cv2.THRESH_BINARY)[1]
# #             cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #             blobs = []
# #             for cnt in cnts:
# #                 if len(cnt) < 4:
# #                     continue
# #                 rect = cv2.minAreaRect(cnt)
# #                 bw, bh = rect[1]
# #                 if min(bw, bh) < 2:
# #                     continue
# #                 side = 0.5 * (bw + bh) / s
# #                 if exp:                                    # loose gate: a merged blob is bigger than a clean box
# #                     if not (0.5 * exp <= side <= 1.9 * exp):
# #                         continue
# #                 elif not (c.box_min_px <= side <= c.box_max_px * 1.5):
# #                     continue
# #                 sq = min(bw, bh) / max(bw, bh)
# #                 fill = cv2.contourArea(cnt) / (bw * bh)
# #                 if sq < 0.55 or fill < 0.55:
# #                     continue
# #                 if any(np.linalg.norm(np.array(rect[0]) / s - t) < 0.3 * side for t in tried):
# #                     continue                               # this spot already has a box
# #                 blobs.append((sq * fill, rect, side))
# #             for _, rect, side in sorted(blobs, key=lambda b: -b[0])[:c.max_fits_per_level]:
# #                 box = self._fit_box(gray, order_quad(cv2.boxPoints(rect) / s), exp)
# #                 if box is not None and self._looks_like_qr(gray, box):
# #                     boxes.append(box)
# #                     tried.append(box.mean(axis=0))
# #         return boxes

# #     # ---- decoding inside a box ----
# #     def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[Candidate]:
# #         h, w = gray.shape[:2]
# #         x0, y0 = quad.min(axis=0)
# #         x1, y1 = quad.max(axis=0)
# #         m = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
# #         x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
# #         x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
# #         crop = gray[y0:y1, x0:x1]
# #         if crop.size == 0:
# #             return []
# #         up = 2.0 if max(crop.shape) < 240 else 1.0
# #         if up != 1.0:
# #             crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
# #         res = self._zbar(crop)
# #         if res:
# #             return self._make(res, "crop", gray, up, x0, y0)
# #         c = self._clahe.apply(crop)
# #         sharp = cv2.addWeighted(c, 1.6, cv2.GaussianBlur(c, (0, 0), 2.0), -0.6, 0)
# #         res = self._zbar(sharp) or self._zbar(cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
# #         if res:
# #             return self._make(res, "rescue", gray, up, x0, y0)
# #         return []

# #     # ---- one frame ----
# #     def scan(self, gray: np.ndarray, altitude_m: Optional[float] = None,
# #              full: Optional[bool] = None) -> List[Candidate]:
# #         """All QR candidates in this frame (decoded or not). altitude_m = height above the ground."""
# #         self._n += 1
# #         h, w = gray.shape[:2]
# #         found: List[Candidate] = []
# #         for box in self._white_boxes(gray, altitude_m):
# #             inner = self._inner(box)
# #             probe = Candidate(None, inner, "box", (w, h))
# #             if self._duplicate(probe, found):
# #                 continue
# #             decoded = self._decode_crop(gray, inner)
# #             if decoded:
# #                 found += [d for d in decoded if not self._duplicate(d, found)]
# #             else:
# #                 probe.patch = canonical_patch(gray, inner, self.cfg)
# #                 found.append(probe)
# #         if full is None:
# #             full = self._n % self.cfg.fullframe_every == 0
# #         if full:
# #             found += [d for d in self._make(self._zbar(gray), "direct", gray) if not self._duplicate(d, found)]
# #         return found


# # # ============================================================================ start QR -> reference
# # class StartScanner:
# #     """
# #     Call once per frame while looking at the start QR (~5 m). Returns a Reference once the same code
# #     has been seen in `start_needed` of the last `start_window` frames. 'Same' = same decoded text OR
# #     same pattern, so it also works when the code never decodes.
# #     """

# #     def __init__(self, cfg: Optional[QRConfig] = None, detector: Optional[QRDetector] = None):
# #         self.cfg = cfg or QRConfig()
# #         self.det = detector or QRDetector(self.cfg)
# #         self.window: Deque[Optional[Candidate]] = deque(maxlen=self.cfg.start_window)

# #     def _same(self, a: Candidate, b: Candidate) -> bool:
# #         if a.text is not None and b.text is not None and a.text.strip() == b.text.strip():
# #             return True
# #         return pattern_score(a.patch, b.patch, self.cfg) >= self.cfg.same_threshold

# #     def update(self, gray: np.ndarray, altitude_m: Optional[float] = None) -> Optional[Reference]:
# #         cands = [c for c in self.det.scan(gray, altitude_m) if c.patch is not None]
# #         self.window.append(max(cands, key=lambda c: c.size_px) if cands else None)
# #         seen = [c for c in self.window if c is not None]
# #         if len(seen) < self.cfg.start_needed:
# #             return None
# #         best_group: List[Candidate] = []
# #         for a in seen:
# #             group = [b for b in seen if b is a or self._same(a, b)]
# #             if len(group) > len(best_group):
# #                 best_group = group
# #         if len(best_group) < self.cfg.start_needed:
# #             return None
# #         texts = Counter(c.text.strip() for c in best_group if c.text is not None)
# #         text = texts.most_common(1)[0][0] if texts else None
# #         pick = max(best_group, key=lambda c: (c.text is not None, c.size_px))
# #         return Reference(pick.patch.copy(), text, pick.size_px)


# # # ============================================================================ target decision
# # class TargetMatcher:
# #     """Decides which candidate in a frame (if any) is the target. Text is evidence, pattern decides."""

# #     def __init__(self, reference: Reference, cfg: Optional[QRConfig] = None):
# #         self.cfg = cfg or QRConfig()
# #         self.ref = reference
# #         self.pm = PatternMatcher(reference.patch, self.cfg)
# #         self._ref_text = reference.text.strip() if reference.text is not None else None

# #     # def match(self, cands: List[Candidate]) -> Optional[Candidate]:
# #     #     c = self.cfg
# #     #     for k in cands:
# #     #         k.verdict = ""
# #     #         k.score = self.pm.score(k.patch) if k.patch is not None else -1.0
# #     #         if self._ref_text is not None and k.text is not None and k.text.strip() == self._ref_text:
# #     #             k.verdict = "text"
# #     #         elif k.text is None and k.score >= c.match_threshold:
# #     #             k.verdict = "pattern"
# #     #         elif k.text is not None and k.score >= c.conflict_threshold:
# #     #             k.verdict = "conflict"            # decoded to other text, but the pattern says it is the same code
# #     #     accepted = [k for k in cands if k.verdict]
# #     #     if not accepted:
# #     #         return None
# #     #     best = max(accepted, key=lambda k: (k.verdict == "text", k.score))
# #     #     if best.verdict != "text":
# #     #         others = [k.score for k in cands if k is not best]
# #     #         if others and best.score - max(others) < c.match_margin:
# #     #             return None                        # two similar candidates: do not guess
# #     #     return best
    
# # def match(self, cands: List[Candidate]) -> Optional[Candidate]:
# #     cfg = self.cfg

# #     for candidate in cands:
# #         candidate.verdict = ""
# #         candidate.score = (
# #             self.pm.score(candidate.patch)
# #             if candidate.patch is not None
# #             else -1.0
# #         )

# #         text_matches = (
# #             self._ref_text is not None
# #             and candidate.text is not None
# #             and candidate.text.strip() == self._ref_text
# #         )

# #         if text_matches:
# #             candidate.verdict = "text"

# #         elif candidate.text is None:
# #             if candidate.score >= cfg.match_threshold:
# #                 candidate.verdict = "pattern"

# #         else:
# #             if candidate.score >= cfg.conflict_threshold:
# #                 candidate.verdict = "conflict"

# #     accepted = [c for c in cands if c.verdict]
# #     if not accepted:
# #         return None
# #     best = max(
# #         accepted,
# #         key=lambda c: (
# #             c.verdict == "text",
# #             c.score,
# #         ),
# #     )
# #     if best.verdict != "text":
# #         other_scores = [
# #             c.score for c in cands
# #             if c is not best and c.patch is not None
# #         ]

# #         if (
# #             other_scores
# #             and best.score - max(other_scores) < cfg.match_margin
# #         ):
# #             return None

# #     return best



# # # ============================================================================ tracking
# # class TargetTracker:
# #     """Per frame: scan -> decide -> publish (u, v, capture time). Keeps recent detections for confirmation."""

# #     def __init__(self, detector: QRDetector, matcher: TargetMatcher,
# #                  altitude_fn: Callable[[], Optional[float]] = lambda: None,
# #                  publish: Optional[Callable[[float, float, float], None]] = None,
# #                  undistort: Optional[Callable[[float, float], Tuple[float, float]]] = None):
# #         self.det, self.matcher, self.altitude_fn = detector, matcher, altitude_fn
# #         self.publish, self.undistort = publish, undistort
# #         self.cfg = matcher.cfg
# #         self.recent: Deque[Detection] = deque(maxlen=20)
# #         self.last: Optional[Candidate] = None
# #         self.frames = self.matches = 0

# #     def process(self, gray: np.ndarray, t_capture: float) -> Optional[Detection]:
# #         self.frames += 1
# #         cands = self.det.scan(gray, self.altitude_fn())
# #         hit = self.matcher.match(cands)
# #         self.last = hit
# #         if hit is None:
# #             return None
# #         u, v = (float(x) for x in hit.center)
# #         if self.undistort:
# #             u, v = self.undistort(u, v)
# #         d = (u, v, t_capture - self.cfg.latency_s)
# #         self.matches += 1
# #         self.recent.append(d)
# #         if self.publish:
# #             self.publish(*d)
# #         return d

# #     def confirmed(self) -> Optional[List[Detection]]:
# #         now = time.time()
# #         fresh = [d for d in self.recent if now - d[2] < self.cfg.confirm_window_s]
# #         return fresh[-self.cfg.confirm_frames:] if len(fresh) >= self.cfg.confirm_frames else None


# # # ============================================================================ camera
# # class FrameGrabber:
# #     """
# #     Reads the camera on a thread and always hands out the NEWEST frame together with its capture time
# #     (stamped as soon as cap.read() returns). Forces MJPG 1280x800, buffer 1.
# #     """

# #     def __init__(self, src=0, cfg: Optional[QRConfig] = None):
# #         cfg = cfg or QRConfig()
# #         backend = cv2.CAP_V4L2 if (isinstance(src, int) and sys.platform.startswith("linux")) else cv2.CAP_ANY
# #         self.cap = cv2.VideoCapture(src, backend)
# #         if not self.cap.isOpened():
# #             raise RuntimeError(f"Cannot open camera {src}")
# #         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # before the size
# #         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
# #         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
# #         self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
# #         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# #         if cfg.exposure_abs is not None:
# #             self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                         # 1 = manual (V4L2)
# #             self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_abs)
# #         self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# #         if self.size != (cfg.width, cfg.height):
# #             print(f"[WARN] camera delivers {self.size[0]}x{self.size[1]}, wanted {cfg.width}x{cfg.height}: "
# #                   f"the camera model and the size filter assume {cfg.width}x{cfg.height}", file=sys.stderr)
# #         self._lock, self._new, self._stop = threading.Lock(), threading.Event(), threading.Event()
# #         self._frame, self._idx, self._t = None, 0, 0.0
# #         self._thread = threading.Thread(target=self._loop, daemon=True)
# #         self._thread.start()

# #     def _loop(self) -> None:
# #         while not self._stop.is_set():
# #             ok, frame = self.cap.read()
# #             t = time.time()
# #             if not ok:
# #                 time.sleep(0.01)
# #                 continue
# #             with self._lock:
# #                 self._frame, self._idx, self._t = frame, self._idx + 1, t
# #             self._new.set()

# #     def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray, float]]:
# #         if not self._new.wait(timeout):
# #             return None
# #         self._new.clear()
# #         with self._lock:
# #             return self._idx, self._frame, self._t

# #     def close(self) -> None:
# #         self._stop.set()
# #         self._thread.join(timeout=1.0)
# #         self.cap.release()

# #     def __enter__(self):
# #         return self

# #     def __exit__(self, *exc):
# #         self.close()


# # class DetectionWorker(threading.Thread):
# #     """grab -> track continuously (also while the flight code is busy flying)."""

# #     def __init__(self, grabber, tracker: TargetTracker):
# #         super().__init__(daemon=True)
# #         self.grabber, self.tracker = grabber, tracker
# #         self.errors = 0
# #         self._stop_evt = threading.Event()

# #     def run(self) -> None:
# #         while not self._stop_evt.is_set():
# #             item = self.grabber.read(timeout=0.5)
# #             if item is None:
# #                 continue
# #             _, frame, t = item
# #             try:
# #                 self.tracker.process(to_gray(frame), t)
# #             except Exception as e:                 # the detector must never die mid-mission
# #                 self.errors += 1
# #                 print(f"[aerothon_qr] detector error #{self.errors}: {e!r}", file=sys.stderr)

# #     def stop(self) -> None:
# #         self._stop_evt.set()


# # def make_undistorter(K: np.ndarray, dist: np.ndarray):
# #     """(u, v) from the real lens -> (u, v) in the ideal pinhole image the flight code assumes."""
# #     def undistort(u: float, v: float) -> Tuple[float, float]:
# #         p = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist, P=K)
# #         return float(p[0, 0, 0]), float(p[0, 0, 1])
# #     return undistort


# # def load_calibration(path: str):
# #     """calib.npz from calibrate_camera.py -> ((fx, fy, cx, cy), K, dist)."""
# #     d = np.load(path)
# #     K, dist = d["K"], d["dist"]
# #     return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])), K, dist


# # def video_frames(path: str) -> Iterator[Tuple[int, np.ndarray, float]]:
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         raise RuntimeError(f"Cannot open video {path}")
# #     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# #     idx = 0
# #     while True:
# #         ok, frame = cap.read()
# #         if not ok:
# #             break
# #         idx += 1
# #         yield idx, frame, idx / fps
# #     cap.release()


# # # ============================================================================ test harness
# # def _draw(frame: np.ndarray, cands: List[Candidate], target: Optional[Candidate]) -> np.ndarray:
# #     img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
# #     for k in cands:
# #         col = (0, 255, 0) if k is target else ((0, 200, 0) if k.text is not None else (0, 200, 255))
# #         cv2.polylines(img, [k.quad.astype(np.int32)], True, col, 2)
# #         label = f"{k.method} {k.text[:14] if k.text else '-'} {k.score:.2f} {k.verdict}"
# #         cv2.putText(img, label, tuple(k.quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
# #     return img


# # def run(args) -> None:
# #     cfg = QRConfig(exposure_abs=args.exposure, qr_size_m=args.qr_size)
# #     if args.calib:
# #         cfg.calib = load_calibration(args.calib)[0]
# #     det = QRDetector(cfg)
# #     scanner = StartScanner(cfg, det) if args.mode == "start" else None
# #     tracker = None
# #     if args.mode == "target":
# #         if not args.reference:
# #             sys.exit("--mode target needs --reference FILE (made with --mode start --save-reference FILE)")
# #         tracker = TargetTracker(det, TargetMatcher(Reference.load(args.reference), cfg), lambda: args.altitude)

# #     print(f"f={cfg.fx:.0f}px  expected white box at {args.altitude:.0f} m = {cfg.expected_box_px(args.altitude):.0f}px, "
# #           f"QR = {cfg.expected_qr_px(args.altitude):.0f}px")
# #     grabber = FrameGrabber(args.camera, cfg) if args.camera is not None else None
# #     frames = iter(lambda: grabber.read(), None) if grabber else video_frames(args.video)
# #     times, n, with_hit = [], 0, 0
# #     t_begin = time.time()
# #     try:
# #         for item in frames:
# #             if item is None:
# #                 break
# #             idx, frame, t_cap = item
# #             t0 = time.perf_counter()
# #             gray = to_gray(frame)
# #             target = None
# #             if args.mode == "raw":
# #                 cands = det.scan(gray, args.altitude, full=True)
# #             elif args.mode == "start":
# #                 ref = scanner.update(gray, args.altitude)
# #                 cands = [c for c in scanner.window if c is not None][-1:]
# #                 if ref is not None:
# #                     print(f"START QR CONFIRMED at frame {idx}: text={ref.text!r} size={ref.size_px:.0f}px")
# #                     if args.save_reference:
# #                         ref.save(args.save_reference)
# #                         print(f"reference saved to {args.save_reference}")
# #                     break
# #             else:
# #                 tracker.process(gray, t_cap)
# #                 target = tracker.last
# #                 cands = [target] if target else []
# #                 if target and tracker.confirmed():
# #                     u, v = target.center
# #                     print(f"TARGET CONFIRMED at frame {idx}: pixel=({u:.0f},{v:.0f}) {target.verdict} "
# #                           f"score={target.score:.2f} text={target.text!r}")
# #                     break
# #             times.append((time.perf_counter() - t0) * 1000.0)
# #             n += 1
# #             with_hit += bool(cands)
# #             if args.verbose:
# #                 for k in cands:
# #                     print(f"[{idx}] {k.method:6s} size={k.size_px:5.0f}px text={k.text!r} score={k.score:.2f} {k.verdict}")
# #             if args.show:
# #                 cv2.imshow("qr", cv2.resize(_draw(frame, cands, target), None, fx=0.5, fy=0.5))
# #                 if cv2.waitKey(1) & 0xFF == ord("q"):
# #                     break
# #             if args.max_seconds and time.time() - t_begin > args.max_seconds:
# #                 break
# #     finally:
# #         if grabber:
# #             grabber.close()
# #         if args.show:
# #             cv2.destroyAllWindows()
# #     if n:
# #         t = np.array(times)
# #         print(f"\nframes={n} frames_with_QR={with_hit} ({100 * with_hit / n:.0f}%)  "
# #               f"ms/frame p50={np.percentile(t, 50):.0f} p95={np.percentile(t, 95):.0f} max={t.max():.0f}  "
# #               f"~{1000.0 / max(t.mean(), 1e-6):.1f} FPS")


# # def main() -> None:
# #     ap = argparse.ArgumentParser(description="AeroTHON QR vision test harness")
# #     src = ap.add_mutually_exclusive_group(required=True)
# #     src.add_argument("--camera", type=int)
# #     src.add_argument("--video", help="recording at full 1280x800 (no resizing)")
# #     ap.add_argument("--mode", choices=["raw", "start", "target"], default="raw")
# #     ap.add_argument("--reference", help="reference file for --mode target")
# #     ap.add_argument("--save-reference", help="where --mode start writes the reference")
# #     ap.add_argument("--altitude", type=float, default=10.0, help="m above ground (size filter)")
# #     ap.add_argument("--qr-size", type=float, default=1.0)
# #     ap.add_argument("--calib", help="calib.npz from calibrate_camera.py")
# #     ap.add_argument("--exposure", type=int, default=None, help="manual exposure (V4L2 units, 100 us)")
# #     ap.add_argument("--max-seconds", type=float, default=0.0)
# #     ap.add_argument("--show", action="store_true")
# #     ap.add_argument("--verbose", action="store_true")
# #     run(ap.parse_args())




# # if __name__ == "__main__":
# #     main()
