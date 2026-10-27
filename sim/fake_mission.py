"""
sim/fake_mission.py
-------------------
Whole-mission dry run with no hardware:  python main.py --fake

  SimVehicle     kinematic drone. Emulates ArduPilot resetting the guided speed to WP_SPD on every new
                 position target, so a speed command sent BEFORE the target is visibly ignored.
  SimWorld       draws synthetic downward-camera frames (green banner, red zone, blue banner)
  StubQR         stands in for the real QRAdapter (needs real QR images). The delivery code is "seen" only when it
                 lies fully inside the QR camera footprint (field_nav.footprint_m), using the drone's real heading.
  StubNavigator  corridor run = move CORRIDOR_LENGTH_M along the current heading

The vision adapters (banner, red zone), field planner, sweep, turn, FSM and the Runner are the REAL code.
The field is a GPS-fenced 40 x 30 m rectangle rotated by `rotate_deg` and offset from the corridor exit,
so the sweep has to find it from the corner coordinates alone.
"""
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from control.corridor_navigator import CorridorResult
from control.field_nav import footprint_m, lane_spacing_m, ne_to_ll, plan_sweep
from fsm.mission_fsm import Event, State

LAT0, LON0 = 12.9716510, 80.0442180     # where the sim drone is armed (local origin)
PX_PER_M_AT_1M = 530.0                  # Pi Camera V2, 640 px wide, ~62 deg HFOV
GREEN_BGR = (64, 177, 0)                # #00B140
BLUE_BGR = (255, 0, 0)                  # #0000FF
RED_BGR = (0, 0, 255)
QR_HALF_M = 0.6                         # QR (1 m) + border: it must be fully inside the footprint to decode


class SimVehicle:
    CLIMB_MPS = 2.0
    YAW_RATE = 0.6
    WP_SPD = 0.5                        # what ArduPilot falls back to on every new position target

    def __init__(self):
        self.x = self.y = 0.0
        self.z = 0.0                    # NED down, so airborne is negative
        self.yaw = 0.0
        self.mode, self.armed, self.speed = "STABILIZE", False, self.WP_SPD
        self.t = 0.0
        self.tgt: Optional[Tuple[float, float, float]] = None
        self.tgt_yaw: Optional[float] = None
        self.vel = (0.0, 0.0)           # body forward/right
        self.speed_log: List[float] = []
        self.speeds_flown = set()       # horizontal speeds actually flown under a position target
        self.payload_runs = 0
        self.target_sends = 0
        self.target_log: List[Tuple[float, float, float, float]] = []
        self.battery_pct = 100.0
        self.battery_volts = 16.0
        self.yaw_log: List[Optional[float]] = []

    # --- telemetry
    def get_local_position(self): return (self.x, self.y, self.z)
    def get_yaw(self): return self.yaw
    def get_latlon(self): return ne_to_ll(self.x, self.y, LAT0, LON0)
    def get_relative_altitude(self, apply_sensor_bias=False): return -self.z
    def get_mode(self): return self.mode
    def get_battery(self): return (self.battery_pct, self.battery_volts)
    def heartbeat_age_s(self): return 0.0
    def is_armed(self): return self.armed

    # --- commands
    def set_mode(self, m, wait=True): self.mode = m
    def arm(self, **kw): self.armed = True; return True
    def disarm(self): self.armed = False
    def takeoff(self, alt): self.z = -alt; return True
    def land(self, **kw): self.z, self.armed, self.mode = 0.0, False, "LAND"; return True
    def set_speed(self, s): self.speed = s; self.speed_log.append(s)
    def hold_position(self): self.tgt = None; self.tgt_yaw = None; self.vel = (0.0, 0.0)

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        self.tgt, self.vel = None, (vx, vy)

    def send_position_target_local(self, x, y, z, yaw=None):
        self.tgt, self.vel, self.tgt_yaw = (x, y, z), (0.0, 0.0), yaw
        self.speed = self.WP_SPD                 # ArduPilot resets the guided speed here
        self.target_sends += 1
        self.target_log.append((self.t, x, y, z))
        self.yaw_log.append(yaw)

    # --- physics
    def advance(self, dt):
        self.t += dt
        if self.mode == "LAND":                      # descend at 1 m/s and disarm on touchdown
            self.tgt, self.vel = None, (0.0, 0.0)
            self.z = min(0.0, self.z + 1.0 * dt)
            if self.z >= 0.0:
                self.armed = False
            return
        if self.tgt is not None:
            tx, ty, tz = self.tgt
            d = math.hypot(tx - self.x, ty - self.y)
            step = min(d, self.speed * dt)
            if d > 1e-9:
                self.x += (tx - self.x) / d * step
                self.y += (ty - self.y) / d * step
                if step > 0:
                    self.speeds_flown.add(round(self.speed, 2))
            dz = tz - self.z
            self.z += max(-self.CLIMB_MPS * dt, min(self.CLIMB_MPS * dt, dz))
            if self.tgt_yaw is not None:
                err = (self.tgt_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += max(-self.YAW_RATE * dt, min(self.YAW_RATE * dt, err))
        else:
            f, r = self.vel
            self.x += (f * math.cos(self.yaw) - r * math.sin(self.yaw)) * dt
            self.y += (f * math.sin(self.yaw) + r * math.cos(self.yaw)) * dt


class SimWorld:
    """Objects on the ground in local NED (north, east, side_m, colour)."""

    def __init__(self, veh: SimVehicle, objects):
        self.v, self.objects = veh, objects

    def pi_frame(self, size=(640, 480)):
        w, h = size
        img = np.full((h, w, 3), 100, np.uint8)
        alt = max(0.5, -self.v.z)
        s = PX_PER_M_AT_1M / alt
        c, sn = math.cos(self.v.yaw), math.sin(self.v.yaw)
        for n, e, side, bgr in self.objects:
            dn, de = n - self.v.x, e - self.v.y
            fwd, right = dn * c + de * sn, -dn * sn + de * c
            cx, cy, half = w / 2 + right * s, h / 2 - fwd * s, side * s / 2
            cv2.rectangle(img, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), bgr, -1)
        return img


class StubQR:
    def __init__(self, veh: SimVehicle, target_ne, frames_needed=5):
        self.v, self.target, self.need, self.seen = veh, target_ne, frames_needed, 0

    def visible(self) -> bool:
        alt = -self.v.z
        if abs(alt - config.DELIVERY_ZONE_ALTITUDE_M) > 1.5:
            return False
        across, along = footprint_m(alt)
        dn, de = self.target[0] - self.v.x, self.target[1] - self.v.y
        fwd = dn * math.cos(self.v.yaw) + de * math.sin(self.v.yaw)
        right = -dn * math.sin(self.v.yaw) + de * math.cos(self.v.yaw)
        return abs(right) <= across / 2 - QR_HALF_M and abs(fwd) <= along / 2 - QR_HALF_M

    def target_offset_m(self, altitude_m):
        """(dx, dy) metres from the image centre to the target in image axes (x right, y down), like the real adapter."""
        dn, de = self.target[0] - self.v.x, self.target[1] - self.v.y
        fwd = dn * math.cos(self.v.yaw) + de * math.sin(self.v.yaw)
        right = -dn * math.sin(self.v.yaw) + de * math.cos(self.v.yaw)
        return (right, -fwd)

    def poll(self, state, frame, altitude_m=None):
        if state is State.SCAN_START_QR:
            self.seen += 1
            return [Event.QR_VALID] if self.seen >= self.need else []
        if state is State.SEARCH_QR and self.visible():
            return [Event.QR_MATCH]
        return []


class StubNavigator:
    def __init__(self, veh): self.v = veh
    def run(self, real_time=True):
        self.v.x += config.CORRIDOR_LENGTH_M * math.cos(self.v.yaw)
        self.v.y += config.CORRIDOR_LENGTH_M * math.sin(self.v.yaw)
        return CorridorResult.SUCCESS


def field_corners_ne(rotate_deg=25.0, corner_from_exit=(8.0, 6.0), long_m=40.0, short_m=30.0):
    """Field corners in local NED: corner 1 offset from the corridor exit, edge 1->2 (40 m) at azimuth rotate_deg,
    edge 1->4 (30 m) 90 deg clockwise from it."""
    L = config.CORRIDOR_LENGTH_M
    c0 = (L + corner_from_exit[0], 0.0 + corner_from_exit[1])
    a = math.radians(rotate_deg)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    return [c0,
            (c0[0] + long_m * u[0], c0[1] + long_m * u[1]),
            (c0[0] + long_m * u[0] + short_m * v[0], c0[1] + long_m * u[1] + short_m * v[1]),
            (c0[0] + short_m * v[0], c0[1] + short_m * v[1])]


def build_fake(target_ne=None, rotate_deg=25.0, corner_from_exit=(8.0, 6.0)):
    """target_ne=None: put the delivery QR on the 4th lane's far side. rotate_deg: field rotation from north."""
    from main import Runner
    veh = SimVehicle()
    L = config.CORRIDOR_LENGTH_M
    exit_n, exit_e = L, 0.0
    poly = field_corners_ne(rotate_deg, corner_from_exit)
    corners_ll = [ne_to_ll(n, e, LAT0, LON0) for n, e in poly]
    plan = plan_sweep(poly, (exit_n, exit_e), lane_spacing_m(), config.SWEEP_MARGIN_M, config.SWEEP_LANE_EDGE)
    wps = plan.waypoints
    lane_mid = lambda i: ((wps[2 * i][0] + wps[2 * i + 1][0]) / 2, (wps[2 * i][1] + wps[2 * i + 1][1]) / 2)
    target = target_ne or lane_mid(plan.lanes - 1)
    red = lane_mid(min(1, plan.lanes - 1))
    world = SimWorld(veh, [
        (0.4, -0.3, 1.5, GREEN_BGR),                                       # entry banner near the take-off point
        (red[0], red[1], 3.0, RED_BGR),                                    # red zone on lane 2
        (exit_n + 0.3, exit_e + 0.2, 1.5, BLUE_BGR),                      # exit banner at the corridor exit
    ])
    config.EXIT_BANNER_NE_M = (0.0, 0.0)
    runner = Runner(veh, qr_frames=lambda: object(), pi_frames=world.pi_frame,
                    navigator_factory=lambda: StubNavigator(veh),
                    payload=lambda: setattr(veh, "payload_runs", veh.payload_runs + 1),
                    qr=StubQR(veh, target), sleep=lambda dt: veh.advance(dt),
                    clock=lambda: veh.t, field_corners=corners_ll)
    log, entered, entry_pos = [], [0.0], {}
    orig = runner.fsm.on_enter
    def hook(s):
        log.append((veh.t, s)); entered[0] = veh.t; entry_pos[s] = (veh.x, veh.y); orig(s)
    runner.fsm.on_enter = hook
    runner.fsm._elapsed = lambda: veh.t - entered[0]          # FSM state timeouts run on SIM time, like on the drone
    runner.sim, runner.sim_log, runner.sim_target = veh, log, target
    runner.sim_entry_pos = entry_pos
    runner.sim_poly, runner.sim_plan, runner.sim_red = poly, plan, red
    return runner