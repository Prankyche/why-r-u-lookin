"""
Whole mission through the REAL driver (drivers/mavlink_connection.Vehicle) over MAVLink/UDP against sim/fake_fc.py.
This is the nearest thing to the drone that runs on a laptop: it covers arm, mode changes, takeoff, position and velocity
targets, speed commands, battery telemetry, landing and every failsafe. Each test takes about 30-60 s.
"""
import math
import random
import time

import pytest

import config
from fsm.mission_fsm import State
from sim.fake_fc import FakeFC
from sim.sitl_mission import build_sitl

pytestmark = pytest.mark.slow


@pytest.fixture
def rig():
    saved = {k: getattr(config, k) for k in ("EXIT_BANNER_NE_M", "SWEEP_SPEED_NORMAL_MS", "SWEEP_SPEED_BOOST_MS",
                                             "FIELD_CORNERS_LL", "BATTERY_DEBOUNCE_S")}
    config.SWEEP_SPEED_NORMAL_MS, config.SWEEP_SPEED_BOOST_MS = 3.0, 4.0
    made = []

    def make(stop_before=None, yaw_deg=30.0, **fc_kw):
        port = random.randint(15000, 19000)
        fc = FakeFC(port, yaw0=math.radians(yaw_deg), time_scale=4.0, **fc_kw)
        fc.start()
        made.append(fc)
        return fc, build_sitl(f"udpin:127.0.0.1:{port}", stop_before=stop_before)

    yield make
    for fc in made:
        fc.stop()
    for k, v in saved.items():
        setattr(config, k, v)


def run_until(runner, predicate, max_ticks=20000):
    for _ in range(max_ticks):
        runner.tick()
        if predicate() or runner.fsm.is_done():
            return
        time.sleep(0.02)


def test_full_mission_with_the_real_driver(rig):
    fc, r = rig()
    assert r.run(max_ticks=40000) is State.DONE
    assert fc.armed is False and fc.z == 0.0
    assert math.hypot(fc.x, fc.y) <= config.HOME_TOL_M + 0.5                 # landed back at the take-off point
    assert [m for _, m in fc.mode_log] == ["GUIDED", "LAND"]                 # no stray mode changes, no RTL
    assert fc.n_targets < 40 and fc.n_vel > 0                                # targets sent once; banner centring used velocity


def test_speed_commands_follow_every_new_target(rig):
    fc, r = rig()
    r.run(max_ticks=40000)
    cmds = [s for _, s in fc.speed_cmds]
    # (the red-zone boost is not asserted here: at 4x time scale the drone crosses the red zone faster than the 10 Hz loop can
    #  see it; that rule is covered at realistic speed in test_runner_fake.py)
    assert config.SWEEP_SPEED_NORMAL_MS in cmds and config.FORWARD_SPEED_MAX_MPS in cmds
    assert len(fc.speed_cmds) >= fc.n_targets - 2                            # the speed is re-sent after (not before) each target


def test_only_the_expected_headings_are_ever_commanded(rig):
    """Initial heading (corridor), sweep heading (nose along the lanes), corridor heading + 180 (return turn) - nothing else."""
    fc, r = rig(yaw_deg=30.0)
    r.run(max_ticks=40000)
    deg = {round(math.degrees(t[4])) for t in fc.targets if t[4] is not None}
    assert deg == {30, 55, -150}, deg
    sweep = [t for t in fc.targets if t[4] is not None and abs(t[3] + 10.0) < 0.1 and round(math.degrees(t[4])) == 55]
    assert len(sweep) >= 3                                                     # one yaw-holding target per waypoint flown (it stops at the QR)
    # targets without a yaw (altitude holds, descent over the QR, return home) must not rotate the drone
    assert any(t[4] is None for t in fc.targets)


def test_pilot_flipping_the_mode_stops_the_script_and_does_not_fight_back(rig):
    fc, r = rig()
    run_until(r, lambda: r.fsm.state is State.SEARCH_QR)
    fc._set_mode(5)                                                          # pilot flips the switch to LOITER
    run_until(r, lambda: r.fsm.is_done(), max_ticks=200)
    assert r.fsm.state is State.RTL
    time.sleep(0.5)
    assert [m for _, m in fc.mode_log][-1] == "LOITER"                       # we did NOT command RTL over the pilot


def test_link_loss_goes_to_rtl_and_commands_rtl(rig):
    fc, r = rig()
    run_until(r, lambda: r.fsm.state is State.SEARCH_QR)
    fc.silent = True
    t0 = time.time()
    run_until(r, lambda: r.fsm.is_done(), max_ticks=2000)
    assert r.fsm.state is State.RTL and time.time() - t0 < config.LINK_LOST_TIMEOUT_S + 3     # not frozen waiting for a reply
    time.sleep(0.5)                                                                           # the command is fire-and-forget
    assert [m for _, m in fc.mode_log][-1] == "RTL"                                           # ...and it still went out


def test_low_battery_goes_to_rtl_and_the_fc_flies_home(rig):
    config.BATTERY_DEBOUNCE_S = 0.5
    fc, r = rig()
    run_until(r, lambda: r.fsm.state is State.SEARCH_QR)
    fc.battery_pct = 10
    run_until(r, lambda: r.fsm.is_done(), max_ticks=2000)
    assert r.fsm.state is State.RTL
    time.sleep(0.5)
    assert [m for _, m in fc.mode_log][-1] == "RTL"


def test_stop_before_corridor_lands_after_the_banner_alignment(rig):
    fc, r = rig(stop_before=State.CORRIDOR_NAV)
    assert r.run(max_ticks=20000) is State.DONE
    assert fc.armed is False
    assert r._corridor_yaw is None                                           # the corridor run never started
    assert math.hypot(fc.x, fc.y) < 3.0                                      # never went down the corridor