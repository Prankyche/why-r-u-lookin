"""
adapters/qr_adapter.py
----------------------
Wraps vision/qr/aerothon_qr.py (the tested QR vision library) and turns its output into FSM events.
The vision library itself is not modified. All settings come from config.py.

  SCAN_START_QR -> QR_VALID     start code confirmed (3 of last 5 frames, ~5 m)
  SEARCH_QR     -> QR_MATCH     delivery code matches the start code (~10 m)
  SEARCH_QR     -> QR_MISMATCH  a different code was decoded and nothing matched

The adapter opens no camera. The caller passes in frames and the current altitude.
"""

import logging
import os
from typing import List, Optional, Tuple

import config
from fsm.mission_fsm import Event, State
from vision.qr import aerothon_qr as aq

log = logging.getLogger("qr_adapter")


def build_qr_config() -> aq.QRConfig:
    """aerothon_qr.QRConfig built from config.py, with the calibration applied if calib.npz exists."""
    cfg = aq.QRConfig(
        width=config.QR_CAMERA_WIDTH,
        height=config.QR_CAMERA_HEIGHT,
        fps=config.QR_CAMERA_FPS,
        exposure_abs=config.QR_EXPOSURE_ABS,
        hfov_deg=config.QR_HFOV_DEG,
        latency_s=config.QR_LATENCY_S,
        qr_size_m=config.QR_SIZE_M,
        border_m=config.QR_BORDER_M,
        fullframe_every=config.QR_FULLFRAME_EVERY,
        start_window=config.QR_START_WINDOW,
        start_needed=config.QR_START_NEEDED,
        confirm_frames=config.QR_CONFIRM_FRAMES,
        confirm_window_s=config.QR_CONFIRM_WINDOW_S,
        match_threshold=config.QR_MATCH_THRESHOLD,
        conflict_threshold=config.QR_CONFLICT_THRESHOLD,
        match_margin=config.QR_MATCH_MARGIN,
    )
    if os.path.exists(config.QR_CALIB_PATH):
        cfg.calib = aq.load_calibration(config.QR_CALIB_PATH)[0]
    else:
        log.warning("QR: %s not found; using the datasheet focal length. Calibrate before flying.",
                    config.QR_CALIB_PATH)
    return cfg


class QRAdapter:
    def __init__(self, cfg: Optional[aq.QRConfig] = None):
        self.cfg = cfg or build_qr_config()
        self.det = aq.QRDetector(self.cfg)
        self._scanner = aq.StartScanner(self.cfg, self.det)
        self._matcher: Optional[aq.TargetMatcher] = None
        self.start_text: Optional[str] = None
        self._start_ref: Optional[aq.Reference] = None
        self._state: Optional[State] = None
        self._last_target: Optional[aq.Candidate] = None   # confirmed target of the last search frame
        self._altitude_m: float = config.SCAN_ALTITUDE_M

    def poll(self, state: State, frame, altitude_m: Optional[float] = None) -> List[Event]:
        """Call once per FSM tick with the current state, the newest camera frame and the altitude (m)."""
        if frame is None:
            return []
        if state is not self._state:
            self._enter(state)
        if altitude_m is not None:
            self._altitude_m = altitude_m
        try:
            gray = aq.to_gray(frame)
            if state is State.SCAN_START_QR:
                return self._scan_start(gray)
            if state is State.SEARCH_QR:
                return self._search_target(gray)
        except Exception:
            log.exception("QR processing failed on this frame; no event this tick.")
        return []

    def target_offset_m(self, altitude_m: float) -> Optional[Tuple[float, float]]:
        """(dx, dy) metres from the image centre to the confirmed target, in image axes.
        None if no target has been confirmed. Mapping to body/NED axes happens later."""
        if self._last_target is None:
            return None
        cx, cy, fx = self._intrinsics()
        u, v = self._last_target.center
        return ((float(u) - cx) * altitude_m / fx, (float(v) - cy) * altitude_m / fx)

    # ---- internals -------------------------------------------------------------

    def _intrinsics(self) -> Tuple[float, float, float]:
        """(cx, cy, fx) in pixels. Uses the calibration when available."""
        if self.cfg.calib is not None:
            fx, _fy, cx, cy = self.cfg.calib
            return cx, cy, fx
        return self.cfg.width / 2.0, self.cfg.height / 2.0, self.cfg.fx

    def _enter(self, state: State) -> None:
        self._state = state
        if state is State.SCAN_START_QR:
            self._scanner = aq.StartScanner(self.cfg, self.det)   # fresh vote window
            self._altitude_m = config.SCAN_ALTITUDE_M
            log.info("QR: start scan at %.1f m", config.SCAN_ALTITUDE_M)
        elif state is State.SEARCH_QR:
            self._last_target = None
            self._altitude_m = config.DELIVERY_ZONE_ALTITUDE_M
            if self._start_ref is None:
                log.error("QR: entered SEARCH_QR with no start code recorded.")
                return
            self._matcher = aq.TargetMatcher(self._start_ref, self.cfg)
            log.info("QR: searching for %r at %.1f m",
                     self.start_text, config.DELIVERY_ZONE_ALTITUDE_M)

    def _scan_start(self, gray) -> List[Event]:
        ref = self._scanner.update(gray, self._altitude_m)
        if ref is None:
            return []
        self.start_text = ref.text
        self._start_ref = ref
        log.info("QR: start code confirmed: %r", ref.text)
        return [Event.QR_VALID]

    def _search_target(self, gray) -> List[Event]:
        if self._matcher is None:
            return []
        cands = self.det.scan(gray, self._altitude_m)
        hit = self._matcher.match(cands)
        if hit is not None:
            self._last_target = hit
            log.info("QR: target confirmed (verdict=%s, score=%.2f)", hit.verdict, hit.score)
            return [Event.QR_MATCH]
        if any(c.text is not None and c.text.strip() != self.start_text for c in cands):
            return [Event.QR_MISMATCH]
        return []