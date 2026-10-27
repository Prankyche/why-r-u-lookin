"""End-to-end dry runs of main.Runner against sim/fake_mission.py (no hardware)."""
import math

import pytest

import config
from control.corridor_navigator import CorridorResult
from fsm.mission_fsm import State
from sim.fake_mission import StubNavigator, build_fake


@pytest.fixture(autouse=True)
def restore_config():
    keep = {k: getattr(config, k) for k in ("EXIT_BANNER_NE_M", "FIELD_CORNERS_LL", "FIELD_FENCE_FILE", "LAND_AT_TAKEOFF",
                                            "SWEEP_PASSES", "BATTERY_LOW_PCT", "GEOFENCE_MAX_ALT_M", "GEOFENCE_RADIUS_M",
                                            "PAYLOAD_RUN_S")}
    yield
    for k, v in keep.items():
        setattr(config, k, v)


@pytest.fixture
def mission():
    return build_fake()


def states(r):
    return [s for _, s in r.sim_log]


def test_full_mission_reaches_done_in_order(mission):
    assert mission.run(max_ticks=20000) is State.DONE
    assert states(mission) == [
        State.TAKEOFF, State.SCAN_START_QR, State.ALIGN_BANNER, State.CORRIDOR_NAV, State.ASCEND_TARGET,
        State.SEARCH_QR, State.DESCEND_DROP, State.RELEASE, State.ASCEND_RETURN, State.ALIGN_EXIT_BANNER,
        State.TURN_RETURN, State.RETURN_CORRIDOR, State.RETURN_HOME, State.LAND, State.DONE]


def test_payload_released_once_over_the_target(mission):
    mission.run(max_ticks=20000)
    assert mission.sim.payload_runs == 1
    tn, te = mission.sim_target
    assert math.dist(mission._match_ne, (tn, te)) < 0.5           # the drone flew OVER the QR, not just near it


@pytest.mark.parametrize("a,b", [(2, 2), (20, 15), (38, 28), (10, 28), (30, 3)])
def test_delivery_code_found_anywhere_in_the_rotated_field(a, b):
    """The target is placed at (a, b) m from the field corner; the planned sweep must find it and release."""
    from sim.fake_mission import field_corners_ne
    poly = field_corners_ne()
    u = ((poly[1][0] - poly[0][0]) / 40, (poly[1][1] - poly[0][1]) / 40)
    v = ((poly[3][0] - poly[0][0]) / 30, (poly[3][1] - poly[0][1]) / 30)
    tgt = (poly[0][0] + a * u[0] + b * v[0], poly[0][1] + a * u[1] + b * v[1])
    r = build_fake(target_ne=tgt)
    assert r.run(max_ticks=30000) is State.DONE and r.sim.payload_runs == 1


def test_target_outside_the_field_ends_in_rtl_not_a_drop():
    from sim.fake_mission import field_corners_ne
    poly = field_corners_ne()
    r = build_fake(target_ne=(poly[0][0] - 40.0, poly[0][1] - 40.0))
    assert r.run(max_ticks=30000) is State.RTL and r.sim.payload_runs == 0


def test_drone_starts_the_sweep_at_the_field_not_at_the_corridor_exit(mission):
    mission.run(max_ticks=20000)
    plan = mission.sim_plan
    assert math.dist(plan.waypoints[0], mission.sim_poly[plan.home_corner]) < 8.0
    assert plan.home_corner == 0                                   # nearest corner to the corridor exit


def test_each_position_target_is_sent_once_not_every_tick(mission):
    mission.run(max_ticks=20000)
    ticks = len(mission.sim_log)
    assert mission.sim.target_sends < 40                           # a few per state + one per sweep waypoint
    assert mission.sim.target_sends < 0.02 * (mission.sim.t / mission.period)


def test_speed_is_sent_after_the_target_so_it_is_not_reset(mission):
    mission.run(max_ticks=20000)
    flown = mission.sim.speeds_flown
    assert config.SWEEP_SPEED_NORMAL_MS in flown and config.SWEEP_SPEED_BOOST_MS in flown
    assert config.FORWARD_SPEED_MAX_MPS in flown


def test_nose_held_along_the_lanes_during_the_sweep(mission):
    mission.run(max_ticks=20000)
    yaws = {round(y, 3) for y in mission.sim.yaw_log if y is not None and abs(y - mission.sim_plan.yaw) < 1e-6}
    assert yaws == {round(mission.sim_plan.yaw, 3)}


def test_corridor_speed_restored_after_the_sweep(mission):
    mission.run(max_ticks=20000)
    assert mission.sim.speed_log[-1] == config.FORWARD_SPEED_MAX_MPS


def test_drone_faces_back_along_the_corridor_for_the_return(mission):
    """The sweep leaves the nose along the lanes (25 deg); the return must still reverse the CORRIDOR heading."""
    mission.run(max_ticks=20000)
    want = mission._corridor_yaw + math.pi
    assert abs((mission.sim.yaw - want + math.pi) % (2 * math.pi) - math.pi) < math.radians(8)
    assert abs(math.degrees(mission.sim_plan.yaw)) > 10          # i.e. the nose really had turned away first


def test_payload_dropped_over_the_qr_not_where_it_was_first_seen(mission):
    seen = {}
    mission.payload = lambda: seen.setdefault("pos", (mission.sim.x, mission.sim.y))
    mission.run(max_ticks=20000)
    assert math.dist(seen["pos"], mission.sim_target) < config.DROP_XY_TOL_M


def test_link_loss_goes_to_rtl_and_commands_rtl(mission):
    mission.run(max_ticks=300)
    mission.sim.heartbeat_age_s = lambda: config.LINK_LOST_TIMEOUT_S + 1
    assert mission.run(max_ticks=5) is State.RTL and mission.sim.mode == "RTL"


def test_pilot_takeover_stops_the_script_without_fighting_the_pilot(mission):
    mission.run(max_ticks=300)
    mission.sim.mode = "LOITER"
    assert mission.run(max_ticks=5) is State.RTL and mission.sim.mode == "LOITER"


def test_corridor_failure_goes_to_rtl(mission):
    class Unsafe(StubNavigator):
        def run(self, real_time=True):
            return CorridorResult.UNSAFE_ABORT
    mission.navigator_factory = lambda: Unsafe(mission.sim)
    assert mission.run(max_ticks=20000) is State.RTL


def test_payload_failure_does_not_confirm_the_drop(mission):
    def boom():
        raise RuntimeError("motor driver fault")
    mission.payload = boom
    mission.run(max_ticks=3000)
    assert State.ASCEND_RETURN not in states(mission) and mission.fsm.state in (State.RELEASE, State.RTL)


def test_crashing_tick_holds_and_goes_to_rtl(mission):
    mission.run(max_ticks=100)
    def boom():
        raise ValueError("bad frame")
    mission.pi_frames = boom
    assert mission.run(max_ticks=2000) is State.RTL


def test_bad_field_coordinates_go_to_rtl_not_off_to_the_wrong_place():
    r = build_fake()
    r.field.corners_ll = [(13.5 + d, 80.1 + e) for d, e in [(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)]]   # ~60 km away
    assert r.run(max_ticks=20000) is State.RTL and r.sim.payload_runs == 0


# ---------------------------------------------------------------- landing, battery, geofence, passes, bounded steps
def test_lands_back_at_the_takeoff_point(mission):
    assert mission.run(max_ticks=30000) is State.DONE
    assert math.dist((mission.sim.x, mission.sim.y), mission._home_ne) <= config.HOME_TOL_M
    assert State.RETURN_HOME in states(mission)
    assert mission.sim.armed is False and mission.sim.z == 0.0


def test_landing_at_takeoff_can_be_switched_off(mission):
    config.LAND_AT_TAKEOFF = False
    assert mission.run(max_ticks=30000) is State.DONE


def test_return_home_climbs_before_it_moves(mission):
    """Leaving the corridor at 3 m: the climb to 5 m must happen on the spot, not diagonally across the walls."""
    mission.run(max_ticks=30000)
    t_home = [t for t, s in mission.sim_log if s is State.RETURN_HOME][0]
    first = [tg for tg in mission.sim.target_log if tg[0] >= t_home][0]
    ex, ey = mission.sim_entry_pos[State.RETURN_HOME]
    assert abs(first[1] - ex) < 0.05 and abs(first[2] - ey) < 0.05 and abs(first[3] + 5.0) < 1e-6
    second = [tg for tg in mission.sim.target_log if tg[0] >= t_home][1]
    assert math.dist((second[1], second[2]), mission._home_ne) < 0.05         # then straight to the take-off point


def test_low_battery_mid_sweep_goes_to_rtl(mission):
    mission.run(max_ticks=800)
    assert mission.fsm.state in (State.SEARCH_QR, State.CORRIDOR_NAV, State.ALIGN_BANNER, State.ASCEND_TARGET)
    mission.sim.battery_pct = config.BATTERY_LOW_PCT - 5
    assert mission.run(max_ticks=100) is State.RTL and mission.sim.mode == "RTL"


def test_battery_dip_shorter_than_the_debounce_is_ignored(mission):
    mission.run(max_ticks=800)
    mission.sim.battery_pct = 5.0
    mission.run(max_ticks=int(config.BATTERY_DEBOUNCE_S / mission.period) - 5)
    mission.sim.battery_pct = 80.0
    assert mission.run(max_ticks=30000) is State.DONE


def test_low_battery_does_not_interrupt_the_landing(mission):
    for _ in range(100000):
        mission.tick(); mission.sleep(mission.period)
        if mission.fsm.state is State.LAND:
            break
    assert mission.fsm.state is State.LAND
    mission.sim.battery_pct = 1.0
    assert mission.run(max_ticks=3000) is State.DONE


def test_no_battery_data_warns_and_flies_on(mission, caplog):
    mission.sim.get_battery = lambda: (None, None)
    with caplog.at_level("WARNING"):
        assert mission.run(max_ticks=30000) is State.DONE
    assert "LOW_BATT is DISABLED" in caplog.text


def test_height_ceiling_breach_goes_to_rtl(mission):
    mission.run(max_ticks=800)
    mission.sim.z = -(config.GEOFENCE_MAX_ALT_M + 8.0)          # far enough that it cannot descend back within the debounce
    assert mission.run(max_ticks=100) is State.RTL


def test_brief_overshoot_that_corrects_itself_is_not_a_breach(mission):
    mission.run(max_ticks=800)
    mission.sim.z = -(config.GEOFENCE_MAX_ALT_M + 0.5)          # 0.5 m over: the drone is back under well inside the debounce
    mission.run(max_ticks=60)
    assert mission.fsm.state is not State.RTL


def test_radius_geofence_breach_goes_to_rtl(mission):
    config.GEOFENCE_RADIUS_M = 20.0
    assert mission.run(max_ticks=30000) is State.RTL         # the field is further than 20 m from the take-off point


def test_missing_qr_gets_a_second_pass_before_giving_up():
    from sim.fake_mission import field_corners_ne
    poly = field_corners_ne()
    r = build_fake(target_ne=(poly[0][0] - 40.0, poly[0][1] - 40.0))        # not in the field
    config.SWEEP_PASSES = 2
    assert r.run(max_ticks=60000) is State.RTL
    two = r.sim.t
    config.SWEEP_PASSES = 1
    r1 = build_fake(target_ne=(poly[0][0] - 40.0, poly[0][1] - 40.0))
    assert r1.run(max_ticks=60000) is State.RTL
    assert two > r1.sim.t + 60                                              # the second pass really flew


def test_search_timeout_is_not_extended_by_mismatches(mission):
    from fsm.mission_fsm import Event
    for _ in range(100000):
        mission.tick(); mission.sleep(mission.period)
        if mission.fsm.state is State.SEARCH_QR:
            break
    t0 = mission.fsm.entered_at
    for _ in range(5):
        mission.fsm.step(Event.QR_MISMATCH)
    assert mission.fsm.entered_at == t0 and mission.fsm.state is State.SEARCH_QR


def test_a_hung_payload_does_not_freeze_the_runner(mission):
    import time as _t
    config.PAYLOAD_RUN_S = 0.2
    stopped = []
    def hang():
        _t.sleep(60)                      # a daemon thread: it must not block the runner or pytest
    hang.stop = lambda: stopped.append(True)
    mission.payload = hang
    t0 = _t.time()
    mission.run(max_ticks=3000)
    assert _t.time() - t0 < 30 and stopped and State.ASCEND_RETURN not in states(mission)   # far below the 60 s hang


def test_stop_before_lands_on_the_spot_and_never_enters_the_state(mission):
    mission.stop_before = State.CORRIDOR_NAV
    assert mission.run(max_ticks=20000) is State.DONE
    assert mission._corridor_yaw is None                              # the corridor never started
    assert mission.sim.mode == "LAND" and mission.sim.armed is False
    assert State.SEARCH_QR not in states(mission)


def test_stop_before_a_late_state_flies_up_to_it_first(mission):
    mission.stop_before = State.DESCEND_DROP
    assert mission.run(max_ticks=40000) is State.DONE
    assert State.SEARCH_QR in states(mission) and mission.sim.payload_runs == 0


def test_stopping_does_not_trip_the_pilot_takeover_failsafe(mission):
    mission.stop_before = State.SEARCH_QR
    assert mission.run(max_ticks=20000) is State.DONE                 # LAND mode is not "the pilot took over"