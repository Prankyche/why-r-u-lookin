"""tools/bench.py rehearsed on synthetic videos (no cameras needed)."""
import math
import os

import cv2
import numpy as np
import pytest

import config
from tools import bench

GREEN, BLUE, RED = (64, 177, 0), (255, 0, 0), (0, 0, 255)


def make_video(path, bgr=None, centre=(320, 240), size=(640, 480), side=110, frames=60):
    w, h = size
    img = np.full((h, w, 3), 100, np.uint8)
    if bgr is not None:
        cv2.rectangle(img, (centre[0] - side // 2, centre[1] - side // 2), (centre[0] + side // 2, centre[1] + side // 2), bgr, -1)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for _ in range(frames):
        vw.write(img)
    vw.release()
    return str(path)


def collect(fn, *a, **k):
    lines = []
    fn(*a, out=lines.append, **k)
    return "\n".join(lines)


def test_real_sensors_imports_from_the_repo_root():
    import drivers.real_sensors as rs                       # used to fail: 'from tof_interface import' only worked inside drivers/
    assert hasattr(rs, "RealCorridorSensors")


def test_banner_to_the_right_says_move_right_and_one_ahead_says_forward(tmp_path):
    right = bench._VideoSource(make_video(tmp_path / "r.avi", GREEN, centre=(520, 240)))
    assert "RIGHT" in collect(bench.run_banner, right, seconds=1.5)
    ahead = bench._VideoSource(make_video(tmp_path / "a.avi", GREEN, centre=(320, 90)))
    assert "FORWARD" in collect(bench.run_banner, ahead, seconds=1.5)         # banner above the image centre = in front of the nose


def test_banner_exit_mode_ignores_green_and_finds_blue(tmp_path):
    green = bench._VideoSource(make_video(tmp_path / "g.avi", GREEN))
    assert "no banner in view" in collect(bench.run_banner, green, exit_banner=True, seconds=1.2)
    blue = bench._VideoSource(make_video(tmp_path / "b.avi", BLUE, centre=(480, 240)))
    assert "RIGHT" in collect(bench.run_banner, blue, exit_banner=True, seconds=1.2)


def test_red_command(tmp_path):
    assert "CENTRED" in collect(bench.run_red, bench._VideoSource(make_video(tmp_path / "r.avi", RED, side=140)), seconds=1.2)
    assert "not centred" in collect(bench.run_red, bench._VideoSource(make_video(tmp_path / "n.avi", RED, centre=(60, 60), side=140)), seconds=1.2)


def test_hsv_command_reads_the_banner_colour(tmp_path):
    text = collect(bench.run_hsv, bench._VideoSource(make_video(tmp_path / "g.avi", GREEN)), seconds=1.2)
    h = int(text.split("centre HSV = (")[1].split(",")[0])
    assert abs(h - config.BANNER_HSV[0]) <= 2


def test_cameras_command_checks_the_expected_shapes(tmp_path):
    qr = bench._VideoSource(make_video(tmp_path / "q.avi", size=(1280, 800)))
    pi = bench._VideoSource(make_video(tmp_path / "p.avi", size=(640, 480)))
    text = collect(bench.run_cameras, qr, pi, save_dir=str(tmp_path))
    assert text.count("OK, the long side is the WIDTH") == 2
    assert os.path.exists(tmp_path / "bench_qr.png") and os.path.exists(tmp_path / "bench_pi.png")
    rotated = bench._VideoSource(make_video(tmp_path / "rot.avi", size=(800, 1280)))      # portrait frames: wrong for the sweep maths
    assert "DIFFERENT" in collect(bench.run_cameras, rotated, pi, save_dir=str(tmp_path))


def test_a_missing_camera_frame_is_reported_not_crashed(tmp_path):
    class Dead:
        def read(self): return None
    text = collect(bench.run_cameras, Dead(), Dead(), save_dir=str(tmp_path))
    assert text.count("NO FRAME") == 2


def test_qr_command_reports_forward_and_right_of_the_drone(tmp_path):
    qrcode = pytest.importorskip("qrcode")
    from PIL import Image
    f_px = (config.QR_CAMERA_WIDTH / 2) / math.tan(math.radians(config.QR_HFOV_DEG / 2))
    mod, bord = int(round(f_px / 5.0 * config.QR_SIZE_M)), int(round(f_px / 5.0 * config.QR_BORDER_M))
    qr = np.pad(np.array(qrcode.make("AEROTHON_START_CODE_HERE", border=0).convert("L").resize((mod, mod), Image.NEAREST)), bord, constant_values=255)
    canvas = np.full((800, 1280), 70, np.uint8)
    y0, x0 = 400 - qr.shape[0] // 2 + 60, 640 - qr.shape[1] // 2 + 100                   # 100 px right of and 60 px below the centre
    canvas[y0:y0 + qr.shape[0], x0:x0 + qr.shape[1]] = qr
    path = str(tmp_path / "qr.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 800))
    for _ in range(90):
        vw.write(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    vw.release()
    text = collect(bench.run_qr, bench._VideoSource(path), alt=5.0, seconds=7)
    assert "start code confirmed" in text
    assert f"{100 * 5 / f_px:.2f} m RIGHT" in text and f"{60 * 5 / f_px:.2f} m BACKWARD" in text


def test_payload_command_asks_first_and_does_nothing_without_yes():
    lines = []
    bench.run_payload(out=lines.append, ask=lambda _: "no")
    assert lines == ["cancelled"]