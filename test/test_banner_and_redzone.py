import cv2
import numpy as np

import config
from adapters.banner_adapter import BannerAdapter, centring_velocity, find_banner
from adapters.redzone_adapter import RedZoneAdapter, red_centred
from fsm.mission_fsm import Event, State

GREEN_BGR = (64, 177, 0)     # #00B140
BLUE_BGR = (255, 0, 0)       # #0000FF
RED_BGR = (0, 0, 255)


def scene(bgr, centre, size=(640, 480), side=90):
    """Grey 640x480 frame with a square of colour `bgr` centred at pixel `centre` (x, y)."""
    img = np.full((size[1], size[0], 3), 100, np.uint8)
    x, y = centre
    cv2.rectangle(img, (x - side // 2, y - side // 2), (x + side // 2, y + side // 2), bgr, -1)
    return img


def test_exact_colours_match_the_configured_hsv():
    g = cv2.cvtColor(np.uint8([[GREEN_BGR]]), cv2.COLOR_BGR2HSV)[0, 0]
    b = cv2.cvtColor(np.uint8([[BLUE_BGR]]), cv2.COLOR_BGR2HSV)[0, 0]
    assert abs(int(g[0]) - config.BANNER_HSV[0]) <= 1
    assert abs(int(b[0]) - config.EXIT_BANNER_HSV[0]) <= 1


def test_find_banner_centred_and_offset():
    ex, ey = find_banner(scene(GREEN_BGR, (320, 240)))
    assert abs(ex) < 0.02 and abs(ey) < 0.02
    ex, ey = find_banner(scene(GREEN_BGR, (480, 240)))
    assert ex > 0.4 and abs(ey) < 0.02


def test_find_banner_none_when_absent_or_too_small():
    assert find_banner(np.full((480, 640, 3), 100, np.uint8)) is None
    assert find_banner(scene(GREEN_BGR, (320, 240), side=20)) is None


def test_centring_directions():
    fwd, right, centred = centring_velocity(0.5, 0.0)          # banner right of centre -> move right
    assert right > 0 and fwd == 0 and not centred
    fwd, right, centred = centring_velocity(0.0, -0.5)         # banner above centre (ahead) -> move forward
    assert fwd > 0 and right == 0 and not centred
    assert centring_velocity(0.0, 0.0)[2]


def test_speed_is_clamped():
    fwd, right, _ = centring_velocity(1.0, 1.0)
    assert abs(right) <= config.BANNER_MAX_SPEED_MS and abs(fwd) <= config.BANNER_MAX_SPEED_MS


def test_entry_adapter_needs_consecutive_centred_frames():
    ad = BannerAdapter(State.ALIGN_BANNER)
    events = []
    for _ in range(config.BANNER_ALIGNED_FRAMES - 1):
        events += ad.poll(State.ALIGN_BANNER, scene(GREEN_BGR, (320, 240))).events
    assert events == []
    assert ad.poll(State.ALIGN_BANNER, scene(GREEN_BGR, (320, 240))).events == [Event.BANNER_FOUND]


def test_lost_banner_resets_the_count_and_holds():
    ad = BannerAdapter(State.ALIGN_BANNER)
    for _ in range(config.BANNER_ALIGNED_FRAMES - 1):
        ad.poll(State.ALIGN_BANNER, scene(GREEN_BGR, (320, 240)))
    out = ad.poll(State.ALIGN_BANNER, np.full((480, 640, 3), 100, np.uint8))
    assert out.commands == [("hold",)]
    assert ad.poll(State.ALIGN_BANNER, scene(GREEN_BGR, (320, 240))).events == []


def test_adapter_ignores_other_states_and_missing_frames():
    ad = BannerAdapter(State.ALIGN_BANNER)
    assert ad.poll(State.CORRIDOR_NAV, scene(GREEN_BGR, (320, 240))).commands == []
    assert ad.poll(State.ALIGN_BANNER, None).commands == []


def test_red_centred_true_only_when_centred_and_big_enough():
    assert red_centred(scene(RED_BGR, (320, 240), side=120))
    assert not red_centred(scene(RED_BGR, (80, 80), side=120))      # in the corner
    assert not red_centred(scene(RED_BGR, (320, 240), side=40))     # tiny blob
    assert not red_centred(scene(GREEN_BGR, (320, 240), side=120))  # wrong colour
    assert not red_centred(None)
    assert RedZoneAdapter().poll(scene(RED_BGR, (320, 240), side=120))