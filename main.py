"""
main.py -- Mission 2 (SkyScan) runner.

Wires the FSM, the vehicle, the two cameras and the adapters, and executes their commands.
One loop at MAIN_LOOP_HZ:  read sensors -> failsafe checks -> handler for the current state -> FSM.

    python main.py --fake        no hardware: kinematic fake vehicle + synthetic camera frames
    python main.py               real drone (props-on!  bench-test everything first)

Blocking pieces: the corridor run (CorridorNavigator.run) and the payload release. The FSM
timeouts cannot interrupt those; the navigator has CORRIDOR_TIMEOUT_S and the payload has PAYLOAD_RUN_S.
"""
import argparse
import logging
import math
import threading
import time
from typing import Callable, Optional

import config
from adapters.banner_adapter import BannerAdapter
from adapters.corridor_adapter import run_corridor
from adapters.field_adapter import FieldAdapter
from adapters.qr_adapter import QRAdapter
from adapters.redzone_adapter import RedZoneAdapter
from adapters.turn_adapter import TurnAdapter
from control.field_nav import load_field_corners
from fsm.mission_fsm import ALTITUDE_M, Event, MissionFSM, State

log = logging.getLogger("main")

ALTITUDE_STATES = (State.TAKEOFF, State.ASCEND_TARGET, State.DESCEND_DROP, State.ASCEND_RETURN)


class Runner:
    def __init__(self, vehicle, qr_frames: Callable, pi_frames: Callable, navigator_factory: Callable,
                 payload: Callable, qr: Optional[QRAdapter] = None, hz: Optional[float] = None,
                 sleep: Callable = time.sleep, clock: Callable = time.monotonic,
                 field_corners=None, stop_before=None):
        self.v = vehicle
        self.qr_frames, self.pi_frames = qr_frames, pi_frames          # () -> frame or None
        self.navigator_factory = navigator_factory                     # () -> object with .run(real_time=)
        self.payload = payload                                         # () -> None, blocks while the motor runs
        self.qr = qr or QRAdapter()
        self.banner_in = BannerAdapter(State.ALIGN_BANNER)
        self.banner_out = BannerAdapter(State.ALIGN_EXIT_BANNER, hsv=config.EXIT_BANNER_HSV,
                                        min_area=config.EXIT_BANNER_MIN_AREA_PX)
        self.red = RedZoneAdapter()
        self.field = FieldAdapter(field_corners if field_corners is not None else load_field_corners())
        self.turn = TurnAdapter()
        self.period = 1.0 / (hz or config.MAIN_LOOP_HZ)
        self.sleep, self.clock = sleep, clock
        self.stop_before = stop_before               # State name: when the FSM is about to enter it, land on the spot instead
        self._stopping = False
        self._stop_landing = False
        # position-target bookkeeping: a target is sent ONCE, because every SET_POSITION_TARGET makes
        # ArduPilot re-plan (a wobble), and the speed is sent AFTER it (ArduPilot resets it to WP_SPD).
        self._sent_key = None
        self._sent_t = 0.0
        self._sent_speed = None
        self._want_speed = config.FORWARD_SPEED_MAX_MPS
        self._last_pos = None          # (t, (x, y, z)) for the measured speed
        self._meas_speed = None

        self.fsm = MissionFSM(on_enter=self._on_enter)
        self._fsm_t0 = self.fsm.entered_at
        self._t0 = clock()
        self.trace = []                 # (t, state, north, east, down, yaw) every tick
        self._state_log = []            # (t, state) at every state entry
        self._cur = State.IDLE
        self._anchor = None            # (north, east) held during altitude changes / turns
        self._match_ne = None          # N/E of the delivery QR itself (not of the drone when it was confirmed)
        self._corridor_yaw = None      # heading of the outbound corridor run: the return turn reverses THIS
        self._pilot = False            # pilot took over: never fight them with RTL
        self._ran = False              # blocking one-shot already done in this state
        self._home_ne = None           # N/E where the drone armed (the take-off point): it must land back here
        self._home_phase = 0           # RETURN_HOME: 0 = climb on the spot, 1 = fly over the take-off point
        self._bat_low_since = None
        self._geo_since = None
        self._bat_warned = False

    # ---------------------------------------------------------------- helpers
    def _pos(self):
        p = self.v.get_local_position()
        return None if p is None else (p[0], p[1], p[2])

    def _ne(self):
        p = self._pos()
        return None if p is None else (p[0], p[1])

    def _alt(self):
        return self.v.get_relative_altitude()

    def _step(self, event: Event):
        if event is Event.PILOT_TAKEOVER:
            self._pilot = True
        self.fsm.step(event)

    # ---- command helpers (all vehicle commands go through these) ----
    def _hold(self):
        self._sent_key = None
        self.v.hold_position()

    def _vel(self, forward, right):
        self._sent_key = None
        self.v.send_velocity_body(forward, right, 0.0)

    def _set_speed(self, mps):
        try:
            self.v.set_speed(mps)
        except Exception:
            log.exception("set_speed failed")
        self._sent_speed = mps

    def _send_target(self, n, e, z, yaw=None):
        """Position target (local NED, z negative up). Sent once per new target; re-sent only if the
        drone is not moving toward it (a lost packet), at most every RESEND_S."""
        key = (round(n, 2), round(e, 2), round(z, 2), None if yaw is None else round(yaw, 3))
        now = self.clock()
        stalled = (key == self._sent_key and now - self._sent_t > config.TARGET_RESEND_S
                   and self._meas_speed is not None and self._meas_speed < 0.3
                   and self._last_pos is not None and math.dist(self._last_pos[1], (n, e, z)) > 1.0)
        if key != self._sent_key or stalled:
            self.v.send_position_target_local(n, e, z, yaw)
            self._set_speed(self._want_speed)          # AFTER the target
            self._sent_key, self._sent_t = key, now

    def _hold_alt(self, target_alt_m, ne=None):
        ne = ne or self._anchor or self._ne()
        if ne is None:
            self._hold()
            return
        self._send_target(ne[0], ne[1], -target_alt_m)

    def _over_target(self, ne, alt):
        """Local N/E of the confirmed delivery QR: the drone position plus the camera offset rotated by the heading."""
        off = getattr(self.qr, "target_offset_m", lambda a: None)(alt if alt is not None else config.DELIVERY_ZONE_ALTITUDE_M)
        yaw = self.v.get_yaw()
        if off is None or yaw is None or ne is None:
            return ne
        fwd, right = config.QR_OFFSET_FORWARD_SIGN * off[1], config.QR_OFFSET_RIGHT_SIGN * off[0]
        return (ne[0] + fwd * math.cos(yaw) - right * math.sin(yaw),
                ne[1] + fwd * math.sin(yaw) + right * math.cos(yaw))

    def _alt_reached(self, target):
        a = self._alt()
        return a is not None and abs(a - target) <= config.ALT_TOLERANCE_M

    def _change_alt_blocking(self, target) -> bool:
        """Climb/descend to `target` holding the current N/E. False on timeout."""
        ne = self._ne()
        t0 = time.monotonic()
        while time.monotonic() - t0 < config.ALT_CHANGE_TIMEOUT_S:
            self._track_speed()
            self._hold_alt(target, ne)
            if self._alt_reached(target):
                return True
            self.sleep(self.period)
        return False

    def _track_speed(self):
        p, now = self._pos(), self.clock()
        if p is not None:
            if self._last_pos is not None and now > self._last_pos[0]:
                self._meas_speed = math.dist(p, self._last_pos[1]) / (now - self._last_pos[0])
            self._last_pos = (now, p)

    def _exec(self, commands, alt_target=None):
        for c in commands:
            kind = c[0]
            if kind == "hold":
                self._hold()
            elif kind == "velocity_body":
                self._vel(c[1], c[2])
            elif kind == "speed":
                self._want_speed = c[1]
                if self._sent_key is not None and c[1] != self._sent_speed:
                    self._set_speed(c[1])              # speed change on the leg already being flown
            elif kind == "goto_ne":
                z = -(alt_target if alt_target is not None else ALTITUDE_M[State.SEARCH_QR])
                self._send_target(c[1], c[2], z, c[3] if len(c) > 3 else None)
            elif kind == "yaw_to":
                ne = self._anchor or self._ne()
                if ne is not None:
                    self._send_target(ne[0], ne[1], -ALTITUDE_M[State.TURN_RETURN], c[1])
            else:
                log.error("unknown command %r", c)

    # ---------------------------------------------------------------- state entry
    def _on_enter(self, new: State):
        if new is self._cur:           # self-transition (QR_MISMATCH): keep the sweep going
            return
        self._cur = new
        self._ran = False
        self._state_log.append((round(self.clock() - self._t0, 2), new.name))
        if self.stop_before is not None and new is self.stop_before:
            log.warning('--stop-before %s: landing here instead of entering it', new.name)
            self._stopping = True
            return
        self._anchor = self._ne()
        self._home_phase = 0
        if new is State.DESCEND_DROP and self._match_ne is not None:
            self._anchor = self._match_ne
        self._sent_key = None                  # a new state always sends its own target
        if new is State.SEARCH_QR:
            self._want_speed = config.SWEEP_SPEED_NORMAL_MS
        else:
            # The long transit to the exit banner uses the sweep speed; everything else the corridor speed.
            self._want_speed = (config.SWEEP_SPEED_NORMAL_MS if new in (State.ALIGN_EXIT_BANNER, State.DESCEND_DROP)
                                else config.FORWARD_SPEED_MAX_MPS)
            self._set_speed(self._want_speed)
        if new is State.RTL and not self._pilot:
            try:
                self.v.set_mode("RTL", wait=False)      # fire and forget: a dead link would block a confirmation wait
            except Exception:
                log.exception("RTL mode change failed")
        log.info("entered %s", new.name)

    # ---------------------------------------------------------------- failsafes
    def _failsafes(self):
        s = self.fsm.state
        if s in (State.IDLE, State.DONE, State.RTL):
            return
        age = getattr(self.v, "heartbeat_age_s", lambda: None)()
        if age is not None and age > config.LINK_LOST_TIMEOUT_S:
            self._step(Event.LINK_LOST)
            return
        if config.PILOT_TAKEOVER_CHECK and s is not State.LAND and not self._stopping:
            mode = getattr(self.v, "get_mode", lambda: None)()
            if mode is not None and mode != config.GUIDED_MODE_NAME:
                self._step(Event.PILOT_TAKEOVER)
                return
        if s not in (State.LAND, State.RELEASE) and not self._stopping:   # RTL would interrupt a landing / a drop
            self._battery_check()
            if not self.fsm.is_done():
                self._geofence_check()

    def _held_for(self, bad: bool, since_attr: str, debounce: float) -> bool:
        now = self.clock()
        if not bad:
            setattr(self, since_attr, None)
            return False
        if getattr(self, since_attr) is None:
            setattr(self, since_attr, now)
        return now - getattr(self, since_attr) >= debounce

    def _battery_check(self):
        pct, volts = getattr(self.v, "get_battery", lambda: (None, None))()
        if pct is None and (volts is None or config.BATTERY_LOW_VOLTS is None):
            if not self._bat_warned:
                log.warning("no usable battery data (set BATT_CAPACITY on the FC or config.BATTERY_LOW_VOLTS): "
                            "LOW_BATT is DISABLED")
                self._bat_warned = True
            return
        low = ((pct is not None and pct < config.BATTERY_LOW_PCT) or
               (volts is not None and config.BATTERY_LOW_VOLTS is not None and volts < config.BATTERY_LOW_VOLTS))
        if self._held_for(low, "_bat_low_since", config.BATTERY_DEBOUNCE_S):
            log.error("battery low (%s %%, %s V): RTL", pct, volts)
            self._step(Event.LOW_BATT)

    def _geofence_check(self):
        alt, ne = self._alt(), self._ne()
        out = (alt is not None and alt > config.GEOFENCE_MAX_ALT_M) or (
            config.GEOFENCE_RADIUS_M is not None and self._home_ne is not None and ne is not None
            and math.dist(ne, self._home_ne) > config.GEOFENCE_RADIUS_M)
        if self._held_for(out, "_geo_since", config.GEOFENCE_DEBOUNCE_S):
            log.error("geofence breach (alt %s m, N/E %s): RTL", alt, ne)
            self._step(Event.GEOFENCE_BREACH)

    def _run_bounded(self, fn, timeout):
        """Run a blocking call in a thread so a hang cannot freeze the failsafes. Returns True only if it finished OK."""
        box = {}
        def work():
            try:
                fn()
                box["ok"] = True
            except Exception as e:
                box["err"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            log.error("%s did not finish within %.1f s", getattr(fn, "__name__", "call"), timeout)
            stop = getattr(fn, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    log.exception("stop failed")
        elif "err" in box:
            log.error("call failed: %r", box["err"])
        return box.get("ok", False)

    # ---------------------------------------------------------------- one tick
    def tick(self):
        self._failsafes()
        s = self.fsm.state
        if s in (State.DONE, State.RTL):
            return
        if self._stopping:                                   # --stop-before: land where we are, then end the run
            if not self._stop_landing:
                self._stop_landing = True
                self._hold()
                self.v.set_mode("LAND")
            elif self.v.is_armed() is False:
                self.fsm._goto(State.DONE, reason="stop-before")
            return
        self._track_speed()
        pos = self._pos()
        ne = None if pos is None else (pos[0], pos[1])
        alt = self._alt()
        if pos is not None:
            yaw_now = self.v.get_yaw()
            self.trace.append((round(self.clock() - self._t0, 2), s.name, round(pos[0], 2), round(pos[1], 2),
                               round(pos[2], 2), None if yaw_now is None else round(yaw_now, 3)))

        if s is State.IDLE:
            self.v.set_mode(config.GUIDED_MODE_NAME)
            if self.v.arm():
                self._home_ne = self._ne()
                self._step(Event.ARMED)

        elif s in ALTITUDE_STATES:
            target = ALTITUDE_M[s]
            if s is State.TAKEOFF:
                if not self._ran:
                    self._ran = True
                    self.v.takeoff(target)            # blocks until altitude (or its own timeout)
            else:
                self._hold_alt(target)
            reached = self._alt_reached(target)
            if s is State.DESCEND_DROP and self._anchor is not None and ne is not None:
                reached = reached and math.dist(ne, self._anchor) <= config.DROP_XY_TOL_M   # over the QR, not just low
            if reached:
                self._step(Event.ALT_REACHED)
            if s is State.DESCEND_DROP and self.field.in_field(ne):
                self._step(Event.IN_ZONE)

        elif s is State.SCAN_START_QR:
            self._hold()
            for e in self.qr.poll(s, self.qr_frames(), alt):
                self._step(e)

        elif s in (State.ALIGN_BANNER, State.ALIGN_EXIT_BANNER):
            ad = self.banner_in if s is State.ALIGN_BANNER else self.banner_out
            wp = None
            if (s is State.ALIGN_EXIT_BANNER and config.EXIT_BANNER_NE_M is not None
                    and self.field.origin_ne is not None):
                o, d = self.field.origin_ne, config.EXIT_BANNER_NE_M
                wp = (o[0] + d[0], o[1] + d[1])
            if wp is not None and (ne is None or math.dist(ne, wp) > config.EXIT_BANNER_ARRIVE_M):
                # Coarse: fly to the measured exit-banner position. The banner adapter only creeps
                # (BANNER_MAX_SPEED_MS), so it takes over for the last metre, not the whole transit.
                self._exec([("goto_ne", wp[0], wp[1])], alt_target=ALTITUDE_M.get(s))
            else:
                out = ad.poll(s, self.pi_frames())
                self._exec(out.commands, alt_target=ALTITUDE_M.get(s))
                for e in out.events:
                    self._step(e)

        elif s in (State.CORRIDOR_NAV, State.RETURN_CORRIDOR):
            if not self._ran:
                self._ran = True
                if not self._change_alt_blocking(ALTITUDE_M[s]):
                    self._hold()
                    self._step(Event.TIMEOUT)
                    return
                if s is State.CORRIDOR_NAV:
                    self._corridor_yaw = self.v.get_yaw()
                self._step(run_corridor(self.navigator_factory(), self.v))

        elif s is State.SEARCH_QR:
            red = self.red.poll(self.pi_frames())
            fo = self.field.poll(s, ne, red, self.v.get_latlon())
            self._exec(fo.commands, alt_target=ALTITUDE_M[s])
            if self.field.in_field(ne):
                self._step(Event.IN_ZONE)
            for e in self.qr.poll(s, self.qr_frames(), alt):
                if e is Event.QR_MATCH:
                    self._match_ne = self._over_target(ne, alt)
                self._step(e)
                if self.fsm.state is not State.SEARCH_QR:
                    return
            for e in fo.events:
                self._step(e)

        elif s is State.RELEASE:
            if not self._ran:
                self._ran = True
                self._hold()
                if self._run_bounded(self.payload, config.PAYLOAD_RUN_S + 3.0):
                    self._step(Event.DROP_CONFIRMED)
                else:
                    log.error("payload release failed; the RELEASE timeout will RTL")

        elif s is State.TURN_RETURN:
            out = self.turn.poll(s, self.v.get_yaw(), self._corridor_yaw)
            self._exec(out.commands)
            for e in out.events:
                self._step(e)

        elif s is State.RETURN_HOME:
            target = ALTITUDE_M[s]
            if not config.LAND_AT_TAKEOFF or self._home_ne is None:
                self._step(Event.HOME_REACHED)
            elif self._home_phase == 0:                      # climb on the spot, clear of the corridor walls
                self._hold_alt(target)
                if self._alt_reached(target):
                    self._home_phase = 1
            else:                                            # then fly back over the take-off point
                self._send_target(self._home_ne[0], self._home_ne[1], -target)
                if (ne is not None and math.dist(ne, self._home_ne) <= config.HOME_TOL_M
                        and self._alt_reached(target)):
                    self._step(Event.HOME_REACHED)

        elif s is State.LAND:
            # Non-blocking: switch to LAND once, then watch for disarm. The link-loss check keeps running meanwhile.
            if not self._ran:
                self._ran = True
                self._sent_key = None
                self.v.set_mode("LAND")
            elif self.v.is_armed() is False:
                if self._home_ne is not None and ne is not None:
                    log.info("landed %.2f m from the take-off point (rule: land at the take-off point)",
                             math.dist(ne, self._home_ne))
                self._step(Event.LANDED)

        self.fsm.tick()

    def trace_dict(self) -> dict:
        """Everything needed to check the flight against the plan afterwards (see tools/plot_path.py)."""
        from control.field_nav import footprint_m
        z = self.field._zone
        return {"version": 1, "final_state": self.fsm.state.name, "home": list(self._home_ne) if self._home_ne else None,
                "poly": [list(p) for p in z.poly] if z else None, "passes": self.field.passes,
                "sweep_alt_m": config.DELIVERY_ZONE_ALTITUDE_M, "cam_across": config.SWEEP_CAM_ACROSS,
                "footprint_m": list(footprint_m(config.DELIVERY_ZONE_ALTITUDE_M)), "arrival_radius_m": config.SWEEP_ARRIVAL_RADIUS_M,
                "states": self._state_log, "samples": self.trace}

    def summary(self) -> str:
        """Per-state time used against its FSM timeout: this is how to tune TIMEOUTS_S from a real or SITL run."""
        from fsm.mission_fsm import TIMEOUTS_S
        rows, t_prev = [], self._fsm_t0
        for t, old, new in self.fsm.history:
            dur = t - t_prev
            limit = TIMEOUTS_S.get(old)
            rows.append(f"  {old.name:18s} {dur:6.1f} s" + (f"   of {limit:>3g} s  ({100 * dur / limit:3.0f} %)" if limit else ""))
            t_prev = t
        total = (self.fsm.history[-1][0] - self._fsm_t0) if self.fsm.history else 0.0
        return "state timings:\n" + "\n".join(rows) + f"\n  total {total:.0f} s -> ended in {self.fsm.state.name}"

    def run(self, max_ticks: Optional[int] = None) -> State:
        n = 0
        while not self.fsm.is_done():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:
                log.exception("tick crashed; holding and going to RTL")
                try:
                    self._hold()
                except Exception:
                    pass
                self._step(Event.UNSAFE)
            n += 1
            if max_ticks is not None and n >= max_ticks:
                break
            self.sleep(max(0.0, self.period - (time.monotonic() - t0)))
        return self.fsm.state


# ====================================================================== hardware wiring
class _Cam:
    """Newest-frame reader. QR camera uses the tested FrameGrabber; Pi camera tries picamera2."""
    def __init__(self, index, qr_cam: bool):
        self._cam2 = None
        self._grab = None
        if qr_cam:
            from adapters.qr_adapter import build_qr_config
            from vision.qr import qr_combined as aq
            self._grab = aq.FrameGrabber(index, build_qr_config())
        else:
            try:
                from picamera2 import Picamera2
                self._cam2 = Picamera2()
                self._cam2.configure(self._cam2.create_video_configuration(
                    main={"size": (config.PI_CAMERA_WIDTH, config.PI_CAMERA_HEIGHT), "format": "BGR888"}))
                self._cam2.start()
            except Exception:
                import cv2
                self._cap = cv2.VideoCapture(index)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.PI_CAMERA_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.PI_CAMERA_HEIGHT)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self._grab is not None:
            item = self._grab.read(timeout=0.2)
            return None if item is None else item[1]
        if self._cam2 is not None:
            return self._cam2.capture_array()
        ok, f = self._cap.read()
        return f if ok else None


def _payload_gpio() -> Callable:
    if config.PAYLOAD_GPIO_FORWARD is None:
        raise RuntimeError("config.PAYLOAD_GPIO_FORWARD is None: the payload motor is not configured. "
                           "Set the pin(s) in config.py before a real flight.")
    from gpiozero import OutputDevice
    fwd = OutputDevice(config.PAYLOAD_GPIO_FORWARD)
    rev = OutputDevice(config.PAYLOAD_GPIO_REVERSE) if config.PAYLOAD_GPIO_REVERSE is not None else None

    def run():
        if rev is not None:
            rev.off()
        fwd.on()
        time.sleep(config.PAYLOAD_RUN_S)
        fwd.off()
    run.stop = lambda: (fwd.off(), rev.off() if rev is not None else None)
    return run


def build_real() -> Runner:
    from control.corridor_navigator import CorridorNavigator
    from drivers.mavlink_connection import Vehicle
    from drivers.real_sensors import RealCorridorSensors
    from flight_logs.flight_logger import FlightLogger

    load_field_corners()                       # fail now (bad fence file), not after takeoff
    payload = _payload_gpio()
    v = Vehicle()
    v.connect()
    time.sleep(1.0)
    pct, volts = v.get_battery()
    if pct is None and (volts is None or config.BATTERY_LOW_VOLTS is None):
        print("WARNING: no battery data from the FC and no BATTERY_LOW_VOLTS set: LOW_BATT is DISABLED. "
              "Set BATT_CAPACITY on the FC or BATTERY_LOW_VOLTS in config.py.")
    else:
        print(f"Battery: {pct if pct is not None else '?'} %  {volts if volts is not None else '?'} V "
              f"(RTL below {config.BATTERY_LOW_PCT} %)")
    sensors = RealCorridorSensors(v)
    qr_cam, pi_cam = _Cam(config.QR_CAMERA_INDEX, True), _Cam(config.PI_CAMERA_INDEX, False)
    import os
    runs = []

    def make_navigator():
        """A fresh navigator per corridor run, each with its OWN log file (the default name would overwrite the outbound log)."""
        runs.append(1)
        os.makedirs("flight_logs", exist_ok=True)
        log_path = os.path.join("flight_logs", time.strftime("corridor_%Y%m%d_%H%M%S") + f"_run{len(runs)}.csv")
        return CorridorNavigator(v, sensors, FlightLogger(log_path))

    return Runner(v, qr_cam.read, pi_cam.read, navigator_factory=make_navigator, payload=payload)


def write_trace(runner, directory="flight_logs"):
    import json
    import os
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "trace_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(path, "w") as f:
        json.dump(runner.trace_dict(), f)
    print(f"flight trace saved: {path}\n  check it:  python -m tools.plot_path {path}")
    return path


def parse_target(text):
    """'none' -> None (never found); 'A,B' -> (A, B) metres from the field's first corner."""
    t = text.strip().lower()
    if t in ("none", "off", "-"):
        return None
    try:
        a, b = (float(x) for x in t.split(","))
    except ValueError:
        raise SystemExit(f"--sitl-target must be 'none' or 'A,B' in metres (e.g. 25,21), got {text!r}")
    return (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="run against sim/fake_mission.py (no hardware, simulated vehicle)")
    ap.add_argument("--sitl", metavar="CONNECT", nargs="?", const="tcp:127.0.0.1:5762",
                    help="real driver + MAVLink to an autopilot with no sensors (ArduPilot SITL / Mission Planner), "
                         "simulated cameras/corridor/payload. Default tcp:127.0.0.1:5762")
    ap.add_argument("--sitl-target", metavar="A,B|none", default="25,21",
                    help="--sitl only: where the simulated delivery QR sits, in metres from the field's first corner "
                         "(A along the 40 m edge, B along the 30 m edge). 'none' = the QR is never found, so the drone flies the "
                         "WHOLE sweep (every lane, both passes) and ends in RTL. Default 25,21 sits beside lane 2.")
    ap.add_argument("--stop-before", metavar="STATE", help="land on the spot when the FSM is about to enter STATE (e.g. CORRIDOR_NAV)")
    ap.add_argument("--preflight", action="store_true", help="print the go/no-go checks and exit")
    ap.add_argument("--skip-preflight", action="store_true", help="fly even with BLOCK items (at your own risk)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stop = None
    if args.stop_before:
        try:
            stop = State[args.stop_before.upper()]
        except KeyError:
            raise SystemExit(f"unknown state {args.stop_before!r}; choose from {[x.name for x in State]}")
        if stop in (State.IDLE, State.TAKEOFF, State.DONE, State.RTL):
            raise SystemExit("--stop-before needs a state after TAKEOFF (e.g. SCAN_START_QR, CORRIDOR_NAV, SEARCH_QR)")
    from tools.preflight import check_config, check_vehicle, report
    if args.fake:
        from sim.fake_mission import build_fake
        runner = build_fake()
        runner.stop_before = stop
    elif args.sitl:
        from sim.sitl_mission import build_sitl
        try:
            runner = build_sitl(args.sitl, stop_before=stop, target_ab=parse_target(args.sitl_target))
        except (OSError, RuntimeError) as e:
            raise SystemExit(
                f"\nCannot reach the autopilot at {args.sitl}: {e}\n"
                "  1. Start the simulator first (Mission Planner: Ctrl+F -> Simulation -> Multirotor) and wait until the map shows\n"
                "     the vehicle and the messages say 'EKF3 ... is using GPS'.\n"
                "  2. Mission Planner uses port 5760 itself; scripts connect on 5762 (the default) or 5763.\n"
                "  3. If you started the simulator some other way, pass its address:  python main.py --sitl tcp:127.0.0.1:PORT")
        if not report(check_config(real=False) + check_vehicle(runner.v, real=False), "PRE-FLIGHT (SITL)") and not args.skip_preflight:
            print("Blockers found. Fix them, or pass --skip-preflight.")
            return
    else:
        issues = check_config(real=True)
        if args.preflight or any(i.level == "BLOCK" for i in issues):
            ok = report(issues, "PRE-FLIGHT (config)")
            if args.preflight or (not ok and not args.skip_preflight):
                if not ok:
                    print("Refusing to build the real vehicle with blockers. Fix them, or pass --skip-preflight.")
                return
        try:
            runner = build_real()
        except (OSError, RuntimeError) as e:
            raise SystemExit(f"\nCannot reach the autopilot at {config.MAVLINK_CONNECTION_STRING}: {e}\n"
                             "  Check MAVProxy is running and owns the serial port (--out=udp:127.0.0.1:14550).")
        runner.stop_before = stop
        if not report(check_config(real=True) + check_vehicle(runner.v), "PRE-FLIGHT (vehicle)") and not args.skip_preflight:
            print("Blockers found. Fix them, or pass --skip-preflight.")
            return
        if config.CONFIRM_BEFORE_ARM:
            input("Props on? Area clear? Press ENTER to arm... ")
    try:
        final = runner.run()
        log.info("mission ended in %s", final.name)
        print(runner.summary())
    finally:
        write_trace(runner)


if __name__ == "__main__":
    main()