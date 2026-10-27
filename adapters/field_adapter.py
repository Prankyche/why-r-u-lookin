"""
adapters/field_adapter.py
-------------------------
Runs the field sweep during SEARCH_QR.

Commands (tuples), executed by the runner:
  ("speed", m_per_s)            desired guided speed (the runner sends it AFTER the next target)
  ("goto_ne", north, east, yaw) position target (so BendyRuler keeps working), nose held at yaw
  ("hold",)                     no position / GPS yet
Events:
  SWEEP_DONE  last waypoint reached with no QR match
  UNSAFE      the sweep could not be planned (bad corners, drone too far from the field)
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from control.field_nav import FieldSweep, ll_to_ne, rect_polygon, sweep_speed
import config
from fsm.mission_fsm import Event, State

log = logging.getLogger("field_adapter")


@dataclass
class FieldOutput:
    events: List[Event] = field(default_factory=list)
    commands: List[tuple] = field(default_factory=list)


class FieldAdapter:
    def __init__(self, corners_ll: Optional[List[Tuple[float, float]]] = None):
        self.corners_ll = corners_ll
        self.sweep: Optional[FieldSweep] = None
        self._zone: Optional[FieldSweep] = None     # field polygon, kept after the sweep ends (IN_ZONE check)
        self._last_speed: Optional[float] = None
        self.origin_ne: Optional[Tuple[float, float]] = None   # where SEARCH_QR started
        self._failed = False
        self._pass = 0

    def _polygon(self, pos_ne, latlon):
        if self.corners_ll:
            if latlon is None:
                return None                           # wait for a GPS fix
            return [(pos_ne[0] + dn, pos_ne[1] + de)
                    for dn, de in (ll_to_ne(la, lo, latlon[0], latlon[1]) for la, lo in self.corners_ll)]
        return rect_polygon(pos_ne, config.FIELD_WIDTH_M, config.FIELD_HEIGHT_M, config.FIELD_START_NE_M)

    def poll(self, state: State, pos_ne: Optional[Tuple[float, float]], red_is_centred: bool,
             latlon: Optional[Tuple[float, float]] = None) -> FieldOutput:
        out = FieldOutput()
        if state is not State.SEARCH_QR:
            self.sweep = None
            self._last_speed = None
            self._pass = 0
            return out
        if self._failed or pos_ne is None:
            out.commands.append(("hold",))
            return out
        if self.sweep is None:                       # first tick in SEARCH_QR: plan from where the drone is
            poly = self._polygon(pos_ne, latlon)
            if poly is None:
                out.commands.append(("hold",))
                return out
            try:
                self.sweep = FieldSweep(poly, pos_ne)
            except ValueError as e:
                log.error("sweep not planned: %s", e)
                self._failed = True
                out.events.append(Event.UNSAFE)
                out.commands.append(("hold",))
                return out
            if self._zone is None:
                self._zone = self.sweep
                self.origin_ne = pos_ne
        tgt = self.sweep.update(pos_ne)
        if tgt is None:
            if self._pass + 1 < config.SWEEP_PASSES:
                # Nothing found: sweep again across the other way (cross-hatch), from where the drone is now.
                self._pass += 1
                first = config.SWEEP_LANE_EDGE
                other = first if self._pass % 2 == 0 else ("short" if first == "long" else "long")
                log.info("no QR after pass %d: starting pass %d (lanes along the %s edge)", self._pass, self._pass + 1, other)
                try:
                    self.sweep = FieldSweep(self._zone.poly, pos_ne, lane_edge=other)
                except ValueError as e:
                    log.error("second pass not planned: %s", e)
                    out.events.append(Event.SWEEP_DONE)
                    out.commands.append(("hold",))
                    return out
                self._last_speed = None
                tgt = self.sweep.target()
            else:
                out.events.append(Event.SWEEP_DONE)
                out.commands.append(("hold",))
                return out
        speed = sweep_speed(red_is_centred)
        if speed != self._last_speed:
            out.commands.append(("speed", speed))
            self._last_speed = speed
        out.commands.append(("goto_ne", tgt[0], tgt[1], self.sweep.yaw))
        return out

    def in_field(self, pos_ne) -> bool:
        return self._zone is not None and pos_ne is not None and self._zone.contains(pos_ne, margin=1.0)