import math
import random

import pytest

import config
from sim.fake_fc import FakeFC
from tools.preflight import check_config, check_vehicle, report


@pytest.fixture(autouse=True)
def restore_config():
    keep = {k: getattr(config, k) for k in ("PAYLOAD_GPIO_FORWARD", "FIELD_CORNERS_LL", "FIELD_FENCE_FILE", "EXIT_BANNER_NE_M",
                                            "SENSOR_SOURCE", "PAYLOAD_RUN_S", "BATTERY_LOW_VOLTS", "GEOFENCE_RADIUS_M")}
    yield
    for k, v in keep.items():
        setattr(config, k, v)


def levels(issues, level):
    return {i.key for i in issues if i.level == level}


def test_default_config_is_a_no_go_for_a_real_flight():
    b = levels(check_config(real=True), "BLOCK")
    assert {"PAYLOAD_GPIO_FORWARD", "FIELD_FENCE_FILE"} <= b


def test_filled_in_config_is_a_go():
    config.PAYLOAD_GPIO_FORWARD = 17
    config.PAYLOAD_RUN_S = 1.3
    config.FIELD_CORNERS_LL = [(12.9716510, 80.0442180), (12.9716510, 80.0445000), (12.9719000, 80.0445000), (12.9719000, 80.0442180)]
    config.EXIT_BANNER_NE_M = (1.0, 2.0)
    config.BATTERY_LOW_VOLTS = 14.0
    config.GEOFENCE_RADIUS_M = 80.0
    issues = check_config(real=True)
    assert not levels(issues, "BLOCK")
    assert not (levels(issues, "WARN") & {"PAYLOAD_RUN_S", "EXIT_BANNER_NE_M", "BATTERY_LOW_VOLTS", "GEOFENCE_RADIUS_M"})


def test_missing_fence_file_is_a_blocker(tmp_path):
    config.PAYLOAD_GPIO_FORWARD = 17
    config.FIELD_FENCE_FILE = str(tmp_path / "nope.txt")
    assert "FIELD_FENCE_FILE" in levels(check_config(real=True), "BLOCK")


def test_simulated_sensors_block_a_real_flight_but_not_a_sim_run():
    config.SENSOR_SOURCE = "obslog"
    assert "SENSOR_SOURCE" in levels(check_config(real=True), "BLOCK")
    assert "SENSOR_SOURCE" not in levels(check_config(real=False), "BLOCK")


def test_report_returns_false_with_blockers_and_true_without(capsys):
    assert report(check_config(real=True)) is False
    config.PAYLOAD_GPIO_FORWARD = 17
    config.FIELD_CORNERS_LL = [(1.0, 2.0)] * 4
    assert report(check_config(real=True)) is True
    assert "GO" in capsys.readouterr().out


def test_vehicle_checks_against_the_fake_autopilot():
    from drivers.mavlink_connection import Vehicle
    import time
    port = random.randint(19100, 19900)
    fc = FakeFC(port); fc.start()
    try:
        v = Vehicle(f"udpin:127.0.0.1:{port}"); v.connect()
        time.sleep(1.0)
        assert not levels(check_vehicle(v), "BLOCK")                     # healthy autopilot: GPS, pose, battery, heartbeat
        fc.battery_pct = 40
        time.sleep(1.0)
        assert "battery" in levels(check_vehicle(v), "WARN")
        fc.silent = True
        time.sleep(2.5)
        assert "heartbeat" in levels(check_vehicle(v), "BLOCK")          # a stale link blocks the flight
    finally:
        fc.stop()