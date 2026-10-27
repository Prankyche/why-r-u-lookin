import math

import numpy as np

import config
from adapters.banner_adapter import BannerAdapter, find_banner
from adapters.turn_adapter import TurnAdapter, wrap
from fsm.mission_fsm import Event, MissionFSM, State
from test.test_banner_and_redzone import BLUE_BGR, GREEN_BGR, scene


def test_exit_adapter_finds_blue_and_ignores_green():
    ad = BannerAdapter(State.ALIGN_EXIT_BANNER, hsv=config.EXIT_BANNER_HSV,
                       min_area=config.EXIT_BANNER_MIN_AREA_PX)
    events = []
    for _ in range(config.BANNER_ALIGNED_FRAMES):
        events += ad.poll(State.ALIGN_EXIT_BANNER, scene(BLUE_BGR, (320, 240))).events
    assert events == [Event.BANNER_FOUND]

    ad2 = BannerAdapter(State.ALIGN_EXIT_BANNER, hsv=config.EXIT_BANNER_HSV,
                        min_area=config.EXIT_BANNER_MIN_AREA_PX)
    for _ in range(config.BANNER_ALIGNED_FRAMES + 5):
        assert ad2.poll(State.ALIGN_EXIT_BANNER, scene(GREEN_BGR, (320, 240))).events == []


def test_entry_adapter_ignores_blue_exit_banner():
    ad = BannerAdapter(State.ALIGN_BANNER)
    for _ in range(config.BANNER_ALIGNED_FRAMES + 5):
        assert ad.poll(State.ALIGN_BANNER, scene(BLUE_BGR, (320, 240))).events == []


def test_turn_targets_the_reverse_heading_and_finishes():
    ta = TurnAdapter()
    out = ta.poll(State.TURN_RETURN, 0.0)
    assert out.commands == [("yaw_to", wrap(math.pi))]   # wrapped to [-pi, pi)
    assert out.events == []
    out = ta.poll(State.TURN_RETURN, math.pi - math.radians(config.RETURN_TURN_TOL_DEG) / 2)
    assert out.events == [Event.YAW_REACHED]


def test_turn_handles_wraparound():
    ta = TurnAdapter()
    start = math.radians(170)                 # reverse heading is -10 deg, across the 180 line
    ta.poll(State.TURN_RETURN, start)
    out = ta.poll(State.TURN_RETURN, wrap(start + math.pi))
    assert out.events == [Event.YAW_REACHED]


def test_turn_holds_without_heading():
    out = TurnAdapter().poll(State.TURN_RETURN, None)
    assert out.commands == [("hold",)] and out.events == []


def test_return_path_order_in_the_fsm():
    f = MissionFSM()
    f.state = State.ASCEND_RETURN
    seen = []
    for ev in [Event.ALT_REACHED, Event.BANNER_FOUND, Event.YAW_REACHED, Event.CORRIDOR_EXIT, Event.HOME_REACHED]:
        f.step(ev)
        seen.append(f.state)
    assert seen == [State.ALIGN_EXIT_BANNER, State.TURN_RETURN, State.RETURN_CORRIDOR, State.RETURN_HOME, State.LAND]


def test_turn_reverses_the_reference_heading_when_given():
    ta = TurnAdapter()
    out = ta.poll(State.TURN_RETURN, math.radians(25), reference_yaw=0.0)      # nose is at 25 deg, corridor ran at 0
    assert out.commands == [("yaw_to", wrap(math.pi))]