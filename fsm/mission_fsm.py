"""
Mission 2 (SkyScan) autonomous mission state machine.

The FSM is pure logic: it consumes Events and returns the current State.
It never talks to the flight controller directly. Use on_enter() to map
states to commands, and feed sensor events through step().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

log = logging.getLogger("mission_fsm")


class State(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    SCAN_START_QR = auto()
    ALIGN_BANNER = auto()
    CORRIDOR_NAV = auto()
    ASCEND_TARGET = auto()
    SEARCH_QR = auto()
    DESCEND_DROP = auto()
    RELEASE = auto()
    ASCEND_RETURN = auto()
    ALIGN_EXIT_BANNER = auto()
    TURN_RETURN = auto()
    RETURN_CORRIDOR = auto()
    RETURN_HOME = auto()
    LAND = auto()
    RTL = auto()
    DONE = auto()


class Event(Enum):
    ARMED = auto()
    ALT_REACHED = auto()
    QR_VALID = auto()          # start QR decoded and validated (3-frame)
    QR_MATCH = auto()          # delivery QR matches start QR
    QR_MISMATCH = auto()       # delivery QR seen but does not match
    BANNER_FOUND = auto()
    CORRIDOR_EXIT = auto()
    IN_ZONE = auto()           # local position inside the field rectangle
    DROP_CONFIRMED = auto()    # payload motor run finished
    LANDED = auto()
    YAW_REACHED = auto()       # return turn complete (180 deg)
    TIMEOUT = auto()           # corridor or state timeout
    LINK_LOST = auto()
    LOW_BATT = auto()
    GEOFENCE_BREACH = auto()
    PILOT_TAKEOVER = auto()    # FC left GUIDED (RC switch)
    UNSAFE = auto()            # wall clearance or front sensor failure
    SWEEP_DONE = auto()        # field sweep finished (all passes) with no QR match
    HOME_REACHED = auto()      # back over the take-off point, ready to land


# Target altitude in metres for states that hold altitude.
ALTITUDE_M: dict[State, float] = {
    State.TAKEOFF: 5.0,
    State.SCAN_START_QR: 5.0,
    State.CORRIDOR_NAV: 3.0,
    State.ASCEND_TARGET: 10.0,
    State.SEARCH_QR: 10.0,
    State.DESCEND_DROP: 5.0,
    State.ASCEND_RETURN: 10.0,
    State.RETURN_CORRIDOR: 3.0,
    State.ALIGN_EXIT_BANNER: 10.0,
    State.TURN_RETURN: 10.0,
    State.RETURN_HOME: 5.0,
}

# Per-state timeouts in seconds. Tune on real flights.
TIMEOUTS_S: dict[State, float] = {
    State.TAKEOFF: 20,
    State.SCAN_START_QR: 30,
    State.ALIGN_BANNER: 30,
    State.CORRIDOR_NAV: 90,
    State.ASCEND_TARGET: 20,
    State.SEARCH_QR: 330,
    State.DESCEND_DROP: 20,
    State.RELEASE: 10,
    State.ASCEND_RETURN: 20,
    State.ALIGN_EXIT_BANNER: 60,
    State.TURN_RETURN: 20,
    State.RETURN_CORRIDOR: 90,
    State.RETURN_HOME: 40,
    State.LAND: 60,
}

# Normal flow: (state, event) -> next state.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.ARMED): State.TAKEOFF,
    (State.TAKEOFF, Event.ALT_REACHED): State.SCAN_START_QR,
    (State.SCAN_START_QR, Event.QR_VALID): State.ALIGN_BANNER,
    (State.ALIGN_BANNER, Event.BANNER_FOUND): State.CORRIDOR_NAV,
    (State.CORRIDOR_NAV, Event.CORRIDOR_EXIT): State.ASCEND_TARGET,
    (State.ASCEND_TARGET, Event.ALT_REACHED): State.SEARCH_QR,
    (State.SEARCH_QR, Event.QR_MISMATCH): State.SEARCH_QR,
    (State.SEARCH_QR, Event.SWEEP_DONE): State.RTL,
    (State.DESCEND_DROP, Event.ALT_REACHED): State.RELEASE,
    (State.RELEASE, Event.DROP_CONFIRMED): State.ASCEND_RETURN,
    (State.ASCEND_RETURN, Event.ALT_REACHED): State.ALIGN_EXIT_BANNER,
    (State.ALIGN_EXIT_BANNER, Event.BANNER_FOUND): State.TURN_RETURN,
    (State.TURN_RETURN, Event.YAW_REACHED): State.RETURN_CORRIDOR,
    (State.RETURN_CORRIDOR, Event.CORRIDOR_EXIT): State.RETURN_HOME,
    (State.RETURN_HOME, Event.HOME_REACHED): State.LAND,
    (State.LAND, Event.LANDED): State.DONE,
}

# Failsafes fire from any active state.
FAILSAFE: dict[Event, State] = {
    Event.LINK_LOST: State.RTL,
    Event.LOW_BATT: State.RTL,
    Event.GEOFENCE_BREACH: State.RTL,
    Event.PILOT_TAKEOVER: State.RTL,
    Event.UNSAFE: State.RTL,
    Event.TIMEOUT: State.RTL,
}

TERMINAL = (State.DONE, State.RTL)


@dataclass
class MissionFSM:
    state: State = State.IDLE
    entered_at: float = field(default_factory=time.monotonic)
    zone_ok: bool = False
    qr_matched: bool = False
    alt_ok: bool = False
    _gate_logged: object = None
    history: list[tuple[float, State, State]] = field(default_factory=list)
    on_enter: Optional[Callable[[State], None]] = None

    # ---- public API ------------------------------------------------------

    def step(self, event: Event) -> State:
        """Apply one event. Returns the (possibly new) state."""
        if self.state in TERMINAL:
            return self.state

        # 1. Failsafes override the normal flow.
        if event in FAILSAFE:
            self._goto(FAILSAFE[event], reason=event.name)
            return self.state

        # 2. Latch sensor facts that guard the drop.
        if event is Event.IN_ZONE:
            self.zone_ok = True
        if event is Event.QR_MATCH:
            self.qr_matched = True

        # 3. Search -> drop: needs the QR match.
        if self.state is State.SEARCH_QR and event is Event.QR_MATCH:
            self._goto(State.DESCEND_DROP, reason="QR_MATCH")
            return self.state

        # 4. Drop gate: release only when altitude reached AND QR matched AND inside the zone.
        #    Any of the three can arrive last, so re-check on each.
        if self.state is State.DESCEND_DROP and event in (Event.ALT_REACHED, Event.IN_ZONE, Event.QR_MATCH):
            if event is Event.ALT_REACHED:
                self.alt_ok = True
            if self.alt_ok and self.qr_matched and self.zone_ok:
                self._goto(State.RELEASE, reason="gate passed")
            else:
                key = (self.alt_ok, self.qr_matched, self.zone_ok)
                if key != self._gate_logged:                    # log when the reason changes, not on every tick
                    log.info("Drop gate held: alt_ok=%s qr_matched=%s zone_ok=%s", *key)
                    self._gate_logged = key
            return self.state

        # 5. Standard table lookup.
        nxt = TRANSITIONS.get((self.state, event))
        if nxt is not None:
            self._goto(nxt, reason=event.name)
        return self.state

    def tick(self) -> State:
        """Call every loop iteration. Handles per-state timeouts."""
        if self.state in TERMINAL:
            return self.state
        limit = TIMEOUTS_S.get(self.state)
        if limit is not None and self._elapsed() > limit:
            self._goto(State.RTL, reason=f"timeout in {self.state.name}")
        return self.state

    def is_done(self) -> bool:
        return self.state in TERMINAL

    # ---- internals -------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self.entered_at

    def _goto(self, new: State, reason: str) -> None:
        old = self.state
        self.history.append((time.monotonic(), old, new))
        log.info("%s -> %s (%s)", old.name, new.name, reason)

        # Reset per-search latches each time we (re)enter SEARCH_QR,
        # so a new delivery target is checked from scratch.
        if new is State.SEARCH_QR:
            self.qr_matched = False
            self.zone_ok = False

        if new is State.DESCEND_DROP:
            self.alt_ok = False
            self._gate_logged = None

        self.state = new
        if new is not old:
            self.entered_at = time.monotonic()      # a self-transition (QR_MISMATCH) must not restart the timeout
        if self.on_enter is not None:
            self.on_enter(new)