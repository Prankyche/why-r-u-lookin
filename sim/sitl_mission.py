"""
sim/sitl_mission.py
-------------------
Run the whole mission with the REAL driver (drivers/mavlink_connection.Vehicle) against any MAVLink autopilot that has no
sensors/cameras attached: ArduPilot SITL (Mission Planner: Ctrl+F -> Simulation), or sim/fake_fc.py.

  python main.py --sitl tcp:127.0.0.1:5762

Real: Vehicle, MAVLink, FSM, runner, banner/red-zone/field/turn adapters, planner, failsafes.
Simulated: the cameras (synthetic frames drawn from the drone's real pose), the QR detector (a stub that "sees" the QR when it
is inside the QR camera footprint), the corridor run (flies the corridor length straight ahead), the payload (logged).
The world is built around the drone's pose at start, so it works wherever SITL's home is.
"""
import logging
import math
import time

import config
from control.field_nav import lane_spacing_m, ne_to_ll, plan_sweep
from sim.fake_mission import (BLUE_BGR, GREEN_BGR, RED_BGR, SimWorld, StubQR)
from control.corridor_navigator import CorridorResult

log = logging.getLogger("sitl_mission")


class PoseView:
    """x, y, z, yaw of the real vehicle, read live, so SimWorld/StubQR can draw what the cameras would see."""
    def __init__(self, v):
        self.v = v

    @property
    def x(self): return (self.v.get_local_position() or (0.0, 0.0, 0.0))[0]
    @property
    def y(self): return (self.v.get_local_position() or (0.0, 0.0, 0.0))[1]
    @property
    def z(self): return (self.v.get_local_position() or (0.0, 0.0, 0.0))[2]
    @property
    def yaw(self): return self.v.get_yaw() or 0.0


class MavCorridorStub:
    """Stands in for CorridorNavigator: flies CORRIDOR_LENGTH_M straight ahead on the heading it has (position target)."""
    def __init__(self, v, timeout_s=120.0):
        self.v, self.timeout = v, timeout_s

    def run(self, real_time=True):
        x, y, z = self.v.get_local_position()
        yaw = self.v.get_yaw()
        tx, ty = x + config.CORRIDOR_LENGTH_M * math.cos(yaw), y + config.CORRIDOR_LENGTH_M * math.sin(yaw)
        self.v.send_position_target_local(tx, ty, z, yaw)
        self.v.set_speed(config.FORWARD_SPEED_MAX_MPS)
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            p = self.v.get_local_position()
            if p and math.hypot(tx - p[0], ty - p[1]) < 0.5:
                return CorridorResult.SUCCESS
            time.sleep(0.1)
        return CorridorResult.TIMEOUT


def build_sitl(connect=None, rotate_deg=25.0, corner_from_exit=(8.0, 6.0), target_ab=(25.0, 21.0), stop_before=None):
    """Connect, wait for a pose, build the synthetic world around it and return a Runner ready to .run()."""
    from drivers.mavlink_connection import Vehicle
    from main import Runner
    v = Vehicle(connect)
    v.connect()
    t0 = time.time()
    while time.time() - t0 < 30 and (v.get_local_position() is None or v.get_yaw() is None or v.get_latlon() is None):
        time.sleep(0.2)
    pos, yaw0, ll0 = v.get_local_position(), v.get_yaw(), v.get_latlon()
    if pos is None or yaw0 is None or ll0 is None:
        raise RuntimeError("no position/heading/GPS from the autopilot within 30 s")
    x0, y0 = pos[0], pos[1]

    def loc(fwd, right):                           # drone-initial frame (forward, right) -> local N/E
        return (x0 + fwd * math.cos(yaw0) - right * math.sin(yaw0), y0 + fwd * math.sin(yaw0) + right * math.cos(yaw0))

    exit_f = config.CORRIDOR_LENGTH_M
    c0 = loc(exit_f + corner_from_exit[0], corner_from_exit[1])
    a = yaw0 + math.radians(rotate_deg)
    u, w = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    poly = [c0, (c0[0] + 40 * u[0], c0[1] + 40 * u[1]),
            (c0[0] + 40 * u[0] + 30 * w[0], c0[1] + 40 * u[1] + 30 * w[1]), (c0[0] + 30 * w[0], c0[1] + 30 * w[1])]
    corners_ll = [ne_to_ll(n - x0, e - y0, ll0[0], ll0[1]) for n, e in poly]
    exit_ne = loc(exit_f, 0.0)
    plan = plan_sweep(poly, exit_ne, lane_spacing_m(), config.SWEEP_MARGIN_M, config.SWEEP_LANE_EDGE)
    wps = plan.waypoints
    mid = lambda i: ((wps[2 * i][0] + wps[2 * i + 1][0]) / 2, (wps[2 * i][1] + wps[2 * i + 1][1]) / 2)
    if target_ab is None:                              # no QR anywhere: the sweep must fly every lane of every pass
        target = (x0 + 10000.0, y0 + 10000.0)
    else:
        target = (c0[0] + target_ab[0] * u[0] + target_ab[1] * w[0], c0[1] + target_ab[0] * u[1] + target_ab[1] * w[1])
    red = mid(min(1, plan.lanes - 1))
    ent = loc(0.4, 0.3)
    ex = loc(exit_f + 0.3, 0.2)
    pose = PoseView(v)
    world = SimWorld(pose, [(ent[0], ent[1], 1.5, GREEN_BGR), (red[0], red[1], 3.0, RED_BGR), (ex[0], ex[1], 1.5, BLUE_BGR)])
    config.EXIT_BANNER_NE_M = (0.0, 0.0)
    runner = Runner(v, qr_frames=lambda: object(), pi_frames=world.pi_frame,
                    navigator_factory=lambda: MavCorridorStub(v),
                    payload=lambda: log.info("payload released (simulated)"),
                    qr=StubQR(pose, target), field_corners=corners_ll, stop_before=stop_before)
    runner.sim_target, runner.sim_poly, runner.sim_plan, runner.sim_start = target, poly, plan, (x0, y0, yaw0)
    return runner