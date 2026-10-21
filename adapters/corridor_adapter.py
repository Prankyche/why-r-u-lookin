"""
adapters/corridor_adapter.py
"""

import logging

from control.corridor_navigator import CorridorResult
from fsm.mission_fsm import Event

log = logging.getLogger("corridor_adapter")

RESULT_TO_EVENT = {
    CorridorResult.SUCCESS: Event.CORRIDOR_EXIT,
    CorridorResult.PILOT_TAKEOVER: Event.PILOT_TAKEOVER,
    CorridorResult.LINK_LOST: Event.LINK_LOST,
    CorridorResult.UNSAFE_ABORT: Event.UNSAFE,
    CorridorResult.SENSOR_LOST: Event.UNSAFE,
    CorridorResult.TIMEOUT: Event.TIMEOUT,
}

# Results where the drone must be told to stop commanding before the FSM moves on.
# PILOT_TAKEOVER is excluded: the pilot already has control, so we must not send commands.
HOLD_ON_RESULT = {
    CorridorResult.LINK_LOST,
    CorridorResult.UNSAFE_ABORT,
    CorridorResult.SENSOR_LOST,
    CorridorResult.TIMEOUT,
}


def run_corridor(navigator, vehicle, real_time=True) -> Event:
    """
    Run the existing blocking corridor navigation once and return the FSM event.

    navigator : CorridorNavigator (from control.corridor_navigator)
    vehicle   : the Vehicle (or FakeVehicle) the navigator is flying
    real_time : True on hardware / SITL, False for the offline test
    """
    # Refuse to start without a real position and heading.
    if vehicle.get_local_position() is None or vehicle.get_yaw() is None:
        log.error("Corridor start refused: position or heading unknown.")
        vehicle.hold_position()
        return Event.UNSAFE

    try:
        result = navigator.run(real_time=real_time)
    except Exception:
        log.exception("Corridor loop raised an exception; holding position.")
        vehicle.hold_position()
        return Event.UNSAFE

    if result in HOLD_ON_RESULT:
        vehicle.hold_position()

    event = RESULT_TO_EVENT.get(result)
    if event is None:
        log.error("Unknown corridor result %r; treating as unsafe.", result)
        vehicle.hold_position()
        return Event.UNSAFE
    return event