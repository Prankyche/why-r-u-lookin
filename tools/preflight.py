"""
tools/preflight.py
------------------
Go / no-go gate. `python main.py --preflight` prints it; a real run refuses to arm while any BLOCK remains
(override with --skip-preflight, at your own risk).

  BLOCK  the run would fail or be unsafe by construction
  WARN   probably wrong or unverified
  INFO   things only a human can confirm on the bench

check_config()   needs no hardware.      check_vehicle(v) needs a connected Vehicle (or sim/fake_fc.py).
"""
import os
from collections import namedtuple

import config

Issue = namedtuple("Issue", "level key message")


def check_config(real=True):
    out = []
    add = lambda lvl, key, msg: out.append(Issue(lvl, key, msg))
    if real and config.SENSOR_SOURCE != "real":
        add("BLOCK", "SENSOR_SOURCE", f"is {config.SENSOR_SOURCE!r}: the corridor would run on simulated sensors. Set it to 'real'.")
    if real and config.PAYLOAD_GPIO_FORWARD is None:
        add("BLOCK", "PAYLOAD_GPIO_FORWARD", "is None: the payload motor is not configured, so nothing can be dropped.")
    if config.PAYLOAD_RUN_S == 2.0:
        add("WARN", "PAYLOAD_RUN_S", "is still the 2.0 s placeholder: measure the real release time on the bench.")
    has_fence = bool(getattr(config, "FIELD_FENCE_FILE", None) or getattr(config, "FIELD_CORNERS_LL", None))
    if real and getattr(config, "FIELD_FENCE_FILE", None) and not os.path.exists(config.FIELD_FENCE_FILE):
        add("BLOCK", "FIELD_FENCE_FILE", f"{config.FIELD_FENCE_FILE!r} does not exist.")
    if not has_fence and not real:
        add("INFO", "FIELD_FENCE_FILE", "not set: fine in SITL, where the field comes from the simulated world.")
    elif not has_fence:
        add("BLOCK", "FIELD_FENCE_FILE", "no field corners set (FIELD_FENCE_FILE / FIELD_CORNERS_LL): the sweep would fly a "
                                         f"{config.FIELD_WIDTH_M:g} x {config.FIELD_HEIGHT_M:g} m rectangle from wherever the drone is.")
    if config.EXIT_BANNER_NE_M is None:
        add("WARN", "EXIT_BANNER_NE_M", "is None: after the drop the drone only centres on a blue banner already in view, "
                                         "otherwise it hovers until the timeout and goes to RTL.")
    if config.BATTERY_LOW_VOLTS is None:
        add("WARN", "BATTERY_LOW_VOLTS", "is None: LOW_BATT relies on the FC reporting a percentage (needs BATT_CAPACITY).")
    if config.GEOFENCE_RADIUS_M is None:
        add("WARN", "GEOFENCE_RADIUS_M", "is None: no horizontal limit from the script. Set the FC's own fence as well.")
    if not os.path.exists(config.QR_CALIB_PATH):
        add("WARN", "QR_CALIB_PATH", f"{config.QR_CALIB_PATH!r} not found: the QR camera uses the datasheet focal length.")
    if config.SWEEP_CAM_ACROSS == "h":
        add("INFO", "SWEEP_CAM_ACROSS", "'h' = long side across the lanes. Confirm frame.shape is 800x1280 (Arducam) / 480x640 (Pi cam).")
    add("INFO", "CORRIDOR_LENGTH_M", f"is {config.CORRIDOR_LENGTH_M} m: confirm against the field drawing (22 ft = 6.7 m).")
    add("INFO", "SIGNS", "verify BANNER_*_SIGN and QR_OFFSET_*_SIGN on the bench (props off): the drone must move TOWARD the target.")
    add("INFO", "TIMEOUTS", "FSM timeouts are guesses: time banner alignment and the corridor on the bench.")
    return out


def check_vehicle(v, real=True):
    """Checks that need the autopilot link. Returns Issues. real=False (SITL) turns the battery blocker into a warning."""
    out = []
    add = lambda lvl, key, msg: out.append(Issue(lvl, key, msg))
    if v.get_local_position() is None or v.get_yaw() is None:
        add("BLOCK", "position", "no LOCAL_POSITION_NED / ATTITUDE from the FC (EKF not ready).")
    if v.get_latlon() is None:
        add("BLOCK", "gps", "no GLOBAL_POSITION_INT lat/lon: the GPS field corners cannot be used.")
    age = v.heartbeat_age_s()
    if age is None or age > 2.0:
        add("BLOCK", "heartbeat", f"FC heartbeat is stale ({age}).")
    pct, volts = v.get_battery()
    if pct is None and (volts is None or config.BATTERY_LOW_VOLTS is None):
        add("BLOCK" if real else "WARN", "battery", "no usable battery data: LOW_BATT would be disabled. Set BATT_CAPACITY on the FC or BATTERY_LOW_VOLTS.")
    elif pct is not None and pct < 60:
        add("WARN", "battery", f"only {pct:.0f} % charged.")
    if v.is_armed():
        add("WARN", "armed", "the vehicle is already armed.")
    return out


def report(issues, title="PRE-FLIGHT"):
    order = {"BLOCK": 0, "WARN": 1, "INFO": 2}
    print(f"\n=== {title} ===")
    for i in sorted(issues, key=lambda i: order[i.level]):
        print(f"  [{i.level:5s}] {i.key}: {i.message}")
    n = sum(i.level == "BLOCK" for i in issues)
    print(f"  -> {'NO-GO: ' + str(n) + ' blocker(s)' if n else 'GO (no blockers; read the WARN/INFO lines)'}\n")
    return n == 0