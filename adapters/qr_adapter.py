"""
adapters/qr_adapter.py
----------------------
Wraps the white-border QR pipeline (vision/qr/qr_whiteborder.py) and turns its
output into FSM events. The pipeline itself is not modified.

  SCAN_START_QR -> QR_VALID     start code confirmed (3 of last 5 frames, ~5 m)
  SEARCH_QR     -> QR_MATCH     delivery code matches the start text (~10 m)
  SEARCH_QR     -> QR_MISMATCH  a different code was decoded and nothing matched

The adapter opens no camera. The caller passes in frames.
"""

import logging
from typing import List, Optional

import config
from fsm.mission_fsm import Event, State
from vision.qr.qr_whiteborder import QRConfig, QRPipeline, StartQR, ground_offset_m, to_gray

log = logging.getLogger("qr_adapter")


class QRAdapter:
    def __init__(self, cfg: Optional[QRConfig] = None):
        self.pipe = QRPipeline(cfg or QRConfig())
        self.start_text: Optional[str] = None
        self._start_ref = None
        self._state: Optional[State] = None
        self._last_target = None      # QRHit of the last confirmed target

    def poll(self, state: State, frame) -> List[Event]:
        """Call once per FSM tick with the current state and the newest camera frame."""
        if frame is None:
            return []
        if state is not self._state:
            self._enter(state)
        try:
            gray = to_gray(frame)
            if state is State.SCAN_START_QR:
                return self._scan_start(gray)
            if state is State.SEARCH_QR:
                return self._search_target(gray)
        except Exception:
            log.exception("QR processing failed on this frame; no event this tick.")
        return []

    def target_offset_m(self, altitude_m: float):
        """(dx, dy) metres from image centre to the confirmed target, in image axes.
        None if no target has been confirmed. Mapping to body/NED axes comes later."""
        if self._last_target is None:
            return None
        return ground_offset_m(self._last_target, altitude_m, self.pipe.cfg)

    # ---- internals -------------------------------------------------------------

    def _enter(self, state: State) -> None:
        self._state = state
        if state is State.SCAN_START_QR:
            self.pipe.set_altitude(config.SCAN_ALTITUDE_M)
            self.pipe.start_votes.clear()
            log.info("QR: start scan at %.1f m", config.SCAN_ALTITUDE_M)
        elif state is State.SEARCH_QR:
            self.pipe.set_altitude(config.DELIVERY_ZONE_ALTITUDE_M)
            self._last_target = None
            if self.start_text is None:
                log.error("QR: entered SEARCH_QR with no start code recorded.")
                return
            self.pipe.set_target(self.start_text, self._start_ref)
            log.info("QR: searching for %r at %.1f m",
                     self.start_text, config.DELIVERY_ZONE_ALTITUDE_M)

    def _scan_start(self, gray) -> List[Event]:
        start: Optional[StartQR] = self.pipe.update_start_scan(gray)
        if start is None:
            return []
        self.start_text = start.text
        self._start_ref = start.reference
        log.info("QR: start code confirmed: %r", start.text)
        return [Event.QR_VALID]

    def _search_target(self, gray) -> List[Event]:
        hit = self.pipe.update_target_search(gray)
        if hit is not None:
            self._last_target = hit
            log.info("QR: target confirmed: %r", self.start_text)
            return [Event.QR_MATCH]
        if any(h.data is not None and h.data.strip() != self.start_text
               for h in self.pipe.last_hits):
            return [Event.QR_MISMATCH]
        return []