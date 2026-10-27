import math

import pytest

import config
from adapters.field_adapter import FieldAdapter
from control.field_nav import (FieldSweep, footprint_m, inside, lane_spacing_m, ll_to_ne, load_field_corners,
                               ne_to_ll, parse_fence, path_length_m, plan_sweep, rect_polygon, sweep_speed)
from fsm.mission_fsm import Event, State


@pytest.fixture(autouse=True)
def restore_config():
    keep = {k: getattr(config, k) for k in ("FIELD_FENCE_FILE", "FIELD_CORNERS_LL", "SWEEP_CAM_ACROSS", "SWEEP_OVERLAP", "SWEEP_PASSES",
                                            "SWEEP_SPACING_M", "FIELD_START_NE_M", "FIELD_MAX_START_DIST_M")}
    yield
    for k, v in keep.items():
        setattr(config, k, v)


def rect(home, long_m, short_m, az_deg):
    """Corners: edge 1->2 (long) at azimuth az from north, edge 1->4 (short) 90 deg clockwise from it."""
    a = math.radians(az_deg)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    h = home
    return [h, (h[0] + long_m * u[0], h[1] + long_m * u[1]),
            (h[0] + long_m * u[0] + short_m * v[0], h[1] + long_m * u[1] + short_m * v[1]),
            (h[0] + short_m * v[0], h[1] + short_m * v[1])]


def covered(poly, plan, across_m, along_m, qr_half=0.6, grid=1.0):
    """Every `grid` m point of the field must have the whole QR inside the camera footprint at some moment."""
    pts = [(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:])]
    samples = []
    for (p, q) in pts:
        n = max(1, int(math.dist(p, q) / 0.25))
        samples += [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n + 1)]
    c, s = math.cos(plan.yaw), math.sin(plan.yaw)
    h, a, _, b = poly
    la, lb = math.dist(h, a), math.dist(h, b)
    ua, ub = ((a[0] - h[0]) / la, (a[1] - h[1]) / la), ((b[0] - h[0]) / lb, (b[1] - h[1]) / lb)
    missed = []
    for i in range(int(la) + 1):
        for j in range(int(lb) + 1):
            pn, pe = h[0] + i * ua[0] + j * ub[0], h[1] + i * ua[1] + j * ub[1]
            ok = any(abs(-(pn - x) * s + (pe - y) * c) <= across_m / 2 - qr_half and
                     abs((pn - x) * c + (pe - y) * s) <= along_m / 2 - qr_half for x, y in samples)
            if not ok:
                missed.append((i, j))
    return missed


# ---------------------------------------------------------------- geometry / parsing
def test_ll_ne_roundtrip():
    lat, lon = ne_to_ll(12.3, -4.5, 12.97, 80.04)
    n, e = ll_to_ne(lat, lon, 12.97, 80.04)
    assert abs(n - 12.3) < 1e-3 and abs(e + 4.5) < 1e-3


def test_parse_fence_formats_and_errors():
    txt = "# c\n12.9716510 80.0442180\n12.9716510,80.0445000\n12.9719000 80.0445000\n12.9719000 80.0442180\n"
    assert len(parse_fence(txt)) == 4
    assert len(parse_fence("1.1000000,2.2000000;1.1000000,2.3000000;1.2000000,2.3000000;1.2000000,2.2000000")) == 4
    assert len(parse_fence(txt + "12.9716510 80.0442180\n")) == 4          # closed polygon
    with pytest.raises(ValueError):
        parse_fence("1.1 2.2\n1.1 2.3\n1.2 2.3")


def test_load_field_corners_none_file_and_inline(tmp_path):
    assert load_field_corners() is None
    p = tmp_path / "fence.txt"
    p.write_text("12.9716510 80.0442180\n12.9716510 80.0445000\n12.9719000 80.0445000\n12.9719000 80.0442180\n")
    config.FIELD_FENCE_FILE = str(p)
    assert len(load_field_corners()) == 4
    config.FIELD_FENCE_FILE = None
    config.FIELD_CORNERS_LL = [(12.9716510, 80.0442180), (12.9716510, 80.0445), (12.9719, 80.0445), (12.9719, 80.0442180)]
    assert len(load_field_corners()) == 4


def test_inside_and_edge_tolerance():
    poly = rect((0, 0), 40, 30, 0)
    assert inside(poly, (20, 15)) and not inside(poly, (-3, 15))
    assert inside(poly, (-0.5, 15), tol=1.0) and not inside(poly, (-2.0, 15), tol=1.0)


# ---------------------------------------------------------------- camera footprint / spacing
def test_footprint_and_spacing_defaults():
    across, along = footprint_m(10.0, "h")
    assert abs(across - 14.0) < 0.1 and abs(along - 8.75) < 0.1
    assert footprint_m(10.0, "v") == (along, across)
    assert abs(lane_spacing_m(10.0) - across * (1 - config.SWEEP_OVERLAP)) < 1e-6
    config.SWEEP_SPACING_M = 7.0
    assert lane_spacing_m() == 7.0


def test_overlap_exceeds_the_qr_size():
    across = footprint_m(config.DELIVERY_ZONE_ALTITUDE_M)[0]
    assert across * config.SWEEP_OVERLAP > 2 * 0.6          # a 1.2 m QR can never fall between two lanes


# ---------------------------------------------------------------- planner
# Reference: l_sweep_fov.py fence_plan on a 40 x 30 m field rotated 25 deg, drone on corner 1, spacing 9.6518 m.
# Offsets (N, E) from the take-off point, taken from that script's own output.
REFERENCE = [(-1.13, 3.61), (34.22, 20.09), (31.05, 26.88), (-4.30, 10.41),
             (-7.47, 17.20), (27.88, 33.68), (24.71, 40.48), (-10.64, 24.00)]


def test_plan_matches_the_reference_script():
    poly = rect((0.0, 0.0), 40, 30, 25)
    plan = plan_sweep(poly, (0.0, 0.0), 9.6518, 0.5, "long")
    assert plan.lanes == 4 and len(plan.waypoints) == 8
    for (n, e), (rn, re_) in zip(plan.waypoints, REFERENCE):
        assert abs(n - rn) < 0.05 and abs(e - re_) < 0.05


def test_lanes_alternate_stay_inside_and_hold_heading_along_the_lane():
    poly = rect((0.0, 0.0), 40, 30, 25)
    plan = plan_sweep(poly, (0.0, 0.0), 9.6518, 0.5, "long")
    assert all(inside(poly, p) for p in plan.waypoints)
    d = [math.dist(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:])]
    assert abs(d[0] - 39.0) < 0.1 and abs(d[1] - plan.spacing) < 0.1
    assert abs(plan.yaw - math.radians(25)) < 1e-6


def test_start_from_anywhere_uses_the_nearest_corner_and_goes_to_it():
    poly = rect((100.0, 50.0), 40, 30, 25)
    for far in [(60.0, 20.0), (100.0, 140.0), (190.0, 60.0)]:
        plan = plan_sweep(poly, far, 9.6518, 0.5, "long")
        nearest = min(range(4), key=lambda i: math.dist(poly[i], far))
        assert plan.home_corner == nearest
        assert math.dist(plan.waypoints[0], poly[nearest]) < 8.0      # first waypoint is near that corner
        assert all(inside(poly, p) for p in plan.waypoints)


def test_too_far_or_bad_corners_refused():
    poly = rect((0.0, 0.0), 40, 30, 0)
    with pytest.raises(ValueError):
        plan_sweep(poly, (900.0, 900.0), 9.65, 0.5, "long", max_start_dist=150.0)
    with pytest.raises(ValueError):
        plan_sweep([(0, 0), (0, 1), (1, 1), (1, 0)], (0, 0), 9.65, 0.5, "long")   # 1 m field
    with pytest.raises(ValueError):
        plan_sweep(poly, (0, 0), 0.1, 0.5, "long")


@pytest.mark.parametrize("rot", [0, 25, 90, 200, 313])
@pytest.mark.parametrize("across", ["h", "v"])
def test_every_point_of_the_field_is_seen_with_the_whole_qr(rot, across):
    config.SWEEP_CAM_ACROSS = across
    poly = rect((20.0, -10.0), 40, 30, rot)
    plan = plan_sweep(poly, (0.0, 0.0), lane_spacing_m(), 0.5, "long")
    fa, fl = footprint_m(config.DELIVERY_ZONE_ALTITUDE_M, across)
    assert covered(poly, plan, fa, fl) == []


def test_wrong_camera_axis_assumption_would_leave_gaps():
    """Documents why SWEEP_CAM_ACROSS must be verified: planned for 'h' but mounted 'v' misses ground."""
    config.SWEEP_CAM_ACROSS = "h"
    poly = rect((0.0, 0.0), 40, 30, 0)
    plan = plan_sweep(poly, (0.0, 0.0), lane_spacing_m(), 0.5, "long")
    fa, fl = footprint_m(config.DELIVERY_ZONE_ALTITUDE_M, "v")
    assert covered(poly, plan, fa, fl) != []


def test_default_30x40_path_is_much_shorter_than_before():
    poly = rect((0.0, 0.0), 40, 30, 0)
    plan = plan_sweep(poly, (0.0, 0.0), lane_spacing_m(), 0.5, "long")
    assert plan.lanes == 3 and path_length_m(plan.waypoints) < 150


# ---------------------------------------------------------------- walker / speed
def test_speed_rule():
    assert sweep_speed(True) == config.SWEEP_SPEED_BOOST_MS and sweep_speed(False) == config.SWEEP_SPEED_NORMAL_MS
    assert config.SWEEP_SPEED_NORMAL_MS == 1.5 and 2.5 <= config.SWEEP_SPEED_BOOST_MS <= 3.0


def test_sweep_walks_waypoints_and_finishes():
    sw = FieldSweep(rect((0.0, 0.0), 40, 30, 0), (0.0, 0.0))
    for wp in list(sw.waypoints):
        sw.update(wp)
    assert sw.done and sw.target() is None and sw.update((0, 0)) is None


def test_contains_is_the_field():
    sw = FieldSweep(rect((0.0, 0.0), 40, 30, 0), (0.0, 0.0))
    assert sw.contains((20.0, 15.0)) and not sw.contains((-5.0, 15.0))
    assert sw.contains((-0.5, 0.0), margin=1.0)


# ---------------------------------------------------------------- adapter
def corners_ll_for(poly, lat0=12.9716510, lon0=80.0442180):
    return [ne_to_ll(n, e, lat0, lon0) for n, e in poly]


def test_adapter_gps_corners_need_a_fix_then_plan_from_the_drone_position():
    poly = rect((30.0, 10.0), 40, 30, 25)
    ad = FieldAdapter(corners_ll_for(poly))
    assert ad.poll(State.SEARCH_QR, (0.0, 0.0), False, None).commands == [("hold",)]     # no GPS yet
    out = ad.poll(State.SEARCH_QR, (0.0, 0.0), False, (12.9716510, 80.0442180))
    kinds = [c[0] for c in out.commands]
    assert "speed" in kinds and "goto_ne" in kinds
    goto = [c for c in out.commands if c[0] == "goto_ne"][0]
    assert len(goto) == 4 and abs(goto[3] - math.radians(25)) < 1e-6                     # heading held
    assert all(inside(poly, w, tol=0.1) for w in ad.sweep.waypoints)


def test_adapter_rectangle_fallback_and_start_offset():
    config.FIELD_START_NE_M = (2.0, 3.0)
    ad = FieldAdapter(None)
    ad.poll(State.SEARCH_QR, (10.0, 20.0), False, None)
    assert ad.sweep.poly[0] == (12.0, 23.0)


def test_adapter_speed_changes_only_on_change():
    ad = FieldAdapter(None)
    ad.poll(State.SEARCH_QR, (0.0, 0.0), False)
    out = ad.poll(State.SEARCH_QR, (0.0, 1.0), False)
    assert not any(c[0] == "speed" for c in out.commands)
    out = ad.poll(State.SEARCH_QR, (0.0, 2.0), True)
    assert ("speed", config.SWEEP_SPEED_BOOST_MS) in out.commands


def walk(ad, passes_seen=None):
    """Drive the adapter through every waypoint of the current sweep; returns the events it emitted."""
    events = []
    for wp in list(ad.sweep.waypoints):
        events += ad.poll(State.SEARCH_QR, wp, False).events
    return events


def test_single_pass_emits_sweep_done_once_all_waypoints_reached():
    config.SWEEP_PASSES = 1
    ad = FieldAdapter(None)
    ad.poll(State.SEARCH_QR, (0.0, 0.0), False)
    assert walk(ad) == [Event.SWEEP_DONE]


def test_second_pass_is_cross_hatched_then_done():
    config.SWEEP_PASSES = 2
    ad = FieldAdapter(None)
    ad.poll(State.SEARCH_QR, (0.0, 0.0), False)
    first_yaw = ad.sweep.yaw
    assert walk(ad) == []                                   # pass 1 finished: no SWEEP_DONE yet
    assert abs(math.remainder(ad.sweep.yaw - first_yaw, math.pi) ) > math.radians(60)   # lanes now run the other way
    out = ad.poll(State.SEARCH_QR, ad.sweep.waypoints[0], False)
    assert any(c[0] == "goto_ne" for c in out.commands)
    assert walk(ad) == [Event.SWEEP_DONE]                   # pass 2 finished: now done


def test_second_pass_also_covers_the_whole_field():
    config.SWEEP_PASSES = 2
    ad = FieldAdapter(None)
    ad.poll(State.SEARCH_QR, (0.0, 0.0), False)
    walk(ad)
    plan = ad.sweep.plan
    fa, fl = footprint_m(config.DELIVERY_ZONE_ALTITUDE_M)
    assert covered(ad.sweep.poly, plan, fa, fl) == []


def test_adapter_bad_corners_or_far_away_is_unsafe_not_a_crash():
    poly = rect((0.0, 0.0), 40, 30, 0)
    out = FieldAdapter(corners_ll_for(poly)).poll(State.SEARCH_QR, (0.0, 0.0), False, (13.5, 80.0442180))   # GPS ~60 km away
    assert Event.UNSAFE in out.events and out.commands == [("hold",)]


def test_adapter_idle_outside_search_keeps_zone_and_origin():
    ad = FieldAdapter(None)
    assert ad.poll(State.TAKEOFF, (0, 0), False).commands == []
    ad.poll(State.SEARCH_QR, (10.0, 0.0), False)
    ad.poll(State.DESCEND_DROP, (30.0, 10.0), False)
    assert ad.sweep is None and ad.in_field((30.0, 10.0)) and not ad.in_field((300.0, 10.0))
    assert ad.origin_ne == (10.0, 0.0)


def test_adapter_holds_without_position():
    assert FieldAdapter(None).poll(State.SEARCH_QR, None, False).commands == [("hold",)]