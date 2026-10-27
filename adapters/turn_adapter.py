"""
adapters/turn_adapter.py
------------------------
TURN_RETURN: turn the drone 180 degrees before the return corridor run. The corridor navigator
flies along whatever heading it has at the start, so the drone must face the way back first.

Commands (tuples), executed by the runner:
  ("yaw_to", target_rad)   hold current position and rotate to target_rad
  ("hold",)                 no heading yet: zero velocity
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import config
from fsm.mission_fsm import Event, State


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class TurnOutput:
    events: List[Event] = field(default_factory=list)
    commands: List[tuple] = field(default_factory=list)


class TurnAdapter:
    def __init__(self):
        self._target: Optional[float] = None

    def poll(self, state: State, yaw_rad: Optional[float], reference_yaw: Optional[float] = None) -> TurnOutput:
        out = TurnOutput()
        if state is not State.TURN_RETURN:
            self._target = None
            return out
        if yaw_rad is None:
            out.commands.append(("hold",))
            return out
        if self._target is None:
            # Reverse the CORRIDOR heading when it is known (the sweep leaves the nose along the lanes, not along
            # the corridor); otherwise reverse the current heading.
            self._target = wrap((reference_yaw if reference_yaw is not None else yaw_rad) + math.pi)
        err = wrap(self._target - yaw_rad)
        if abs(err) <= math.radians(config.RETURN_TURN_TOL_DEG):
            out.events.append(Event.YAW_REACHED)
            out.commands.append(("hold",))
        else:
            out.commands.append(("yaw_to", self._target))
        return out