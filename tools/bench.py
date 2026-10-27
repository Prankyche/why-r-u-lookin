"""
tools/bench.py
--------------
Bench checks for the real hardware. PROPS OFF. Nothing here arms the drone or spins a motor except `payload`.

  python -m tools.bench cameras            frame shapes + saves bench_qr.png / bench_pi.png (is the long side across?)
  python -m tools.bench hsv                HSV of the middle of the Pi-cam image: hold the banner there, copy into config
  python -m tools.bench banner [--exit]    where the banner is in the image and which way the drone would move
  python -m tools.bench red                is red centred in the Pi-cam view?
  python -m tools.bench qr [--alt 5]       start-code scan, then where the QR is relative to the drone (forward / right)
  python -m tools.bench sensors            left / right / front distances (needs the flight controller link)
  python -m tools.bench payload            run the payload motor once (asks first)

Sources: --qr-source / --pi-source take a camera index (default from config) or a video file (to rehearse on a laptop).
"""
import argparse
import os
import sys
import time

import cv2

import config


class _VideoSource:
    """A video file that loops, so the commands can be rehearsed without cameras."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self.dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        time.sleep(self.dt)
        return frame if ok else None


def open_source(source, qr):
    if source is None:
        source = config.QR_CAMERA_INDEX if qr else config.PI_CAMERA_INDEX
    if isinstance(source, str) and not source.isdigit():
        return _VideoSource(source)
    from main import _Cam
    return _Cam(int(source), qr)


def _frames(cam, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        f = cam.read()
        if f is not None:
            yield f


def run_cameras(qr_cam, pi_cam, out=print, save_dir="flight_logs"):
    os.makedirs(save_dir, exist_ok=True)
    expect = {"QR (Arducam)": ((800, 1280), qr_cam, "bench_qr.png"), "Pi cam": ((480, 640), pi_cam, "bench_pi.png")}
    for name, (shape, cam, fname) in expect.items():
        frame = next(_frames(cam, 5.0), None)
        if frame is None:
            out(f"{name}: NO FRAME within 5 s: check the cable, the index in config.py, and that nothing else holds the camera")
            continue
        h, w = frame.shape[:2]
        cv2.imwrite(os.path.join(save_dir, fname), frame)
        ok = (h, w) == shape
        out(f"{name}: frame is {h} x {w} (height x width), expected {shape[0]} x {shape[1]}  -> "
            + ("OK, the long side is the WIDTH" if ok else "DIFFERENT: the sweep spacing, banner thresholds and QR geometry assume the expected shape"))
        out(f"   saved {os.path.join(save_dir, fname)}: look at it. Is the image's long side pointing LEFT-RIGHT across the drone? "
            "(that is what SWEEP_CAM_ACROSS='h' assumes)")


def run_hsv(pi_cam, seconds=20, out=print):
    out("Hold the banner over the middle of the Pi-cam image. Reading the HSV of the centre patch...")
    last = 0.0
    for f in _frames(pi_cam, seconds):
        if time.time() - last < 0.5:
            continue
        last = time.time()
        h, w = f.shape[:2]
        patch = f[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        med = [int(sorted(hsv[:, i])[len(hsv) // 2]) for i in range(3)]
        out(f"centre HSV = ({med[0]}, {med[1]}, {med[2]})   configured: green {config.BANNER_HSV}, blue {config.EXIT_BANNER_HSV} "
            f"(hue tolerance +-{config.BANNER_HUE_TOL})")


def run_banner(pi_cam, exit_banner=False, seconds=30, out=print):
    from adapters.banner_adapter import centring_velocity, find_banner
    hsv = config.EXIT_BANNER_HSV if exit_banner else config.BANNER_HSV
    area = config.EXIT_BANNER_MIN_AREA_PX if exit_banner else config.BANNER_MIN_AREA_PX
    out(f"Looking for the {'BLUE exit' if exit_banner else 'GREEN entry'} banner (HSV {hsv}). "
        "Put it on the drone's RIGHT, then in FRONT of the nose, and check the commanded direction.")
    last = 0.0
    for f in _frames(pi_cam, seconds):
        if time.time() - last < 0.5:
            continue
        last = time.time()
        pos = find_banner(f, hsv, area)
        if pos is None:
            out("no banner in view (too small, wrong colour, or not there)")
            continue
        fwd, right, centred = centring_velocity(*pos)
        side = "RIGHT" if right > 0 else "LEFT" if right < 0 else "-"
        ahead = "FORWARD" if fwd > 0 else "BACKWARD" if fwd < 0 else "-"
        out(f"banner at ex={pos[0]:+.2f} ey={pos[1]:+.2f}  ->  drone would move {ahead} {abs(fwd):.3f} m/s, {side} {abs(right):.3f} m/s"
            + ("   [centred]" if centred else ""))


def run_red(pi_cam, seconds=30, out=print):
    from adapters.redzone_adapter import red_centred
    last = 0.0
    for f in _frames(pi_cam, seconds):
        if time.time() - last < 0.5:
            continue
        last = time.time()
        out("red is CENTRED -> sweep would speed up" if red_centred(f) else "red not centred (or not in view)")


def run_qr(qr_cam, alt=5.0, seconds=60, out=print):
    from adapters.qr_adapter import QRAdapter
    from fsm.mission_fsm import Event, State
    qr = QRAdapter()
    out("Show the QR to the Arducam (about the mission distance). Scanning for the start code...")
    state, last = State.SCAN_START_QR, 0.0
    for f in _frames(qr_cam, seconds):
        evs = qr.poll(state, f, alt)
        if state is State.SCAN_START_QR and Event.QR_VALID in evs:
            out(f"start code confirmed: {qr.start_text!r}. Now matching it again as the delivery code (move the QR around)...")
            state = State.SEARCH_QR
        elif state is State.SEARCH_QR and Event.QR_MATCH in evs:
            off = qr.target_offset_m(alt)
            if off is not None and time.time() - last > 0.5:
                last = time.time()
                fwd, right = config.QR_OFFSET_FORWARD_SIGN * off[1], config.QR_OFFSET_RIGHT_SIGN * off[0]
                out(f"QR matched: {abs(fwd):.2f} m {'FORWARD' if fwd > 0 else 'BACKWARD'}, {abs(right):.2f} m {'RIGHT' if right > 0 else 'LEFT'} "
                    f"of the drone (at {alt:g} m). Check this against where you are holding it.")
        elif state is State.SEARCH_QR and Event.QR_MISMATCH in evs:
            out("a DIFFERENT QR was seen (mismatch)")


def run_sensors(seconds=15, out=print):
    from drivers.mavlink_connection import Vehicle
    from drivers.real_sensors import RealCorridorSensors
    v = Vehicle(); v.connect()
    s = RealCorridorSensors(v)
    t0 = time.time()
    while time.time() - t0 < seconds:
        out(str(s.read(0.0, 0.0)))
        time.sleep(0.5)


def run_payload(out=print, ask=input):
    from main import _payload_gpio
    if ask("PROPS OFF and the payload clear of fingers? Type yes to run the motor once: ").strip().lower() != "yes":
        out("cancelled")
        return
    run = _payload_gpio()
    t0 = time.time()
    run()
    out(f"motor ran for {time.time() - t0:.2f} s (PAYLOAD_RUN_S = {config.PAYLOAD_RUN_S}). Did the payload release fully?")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["cameras", "hsv", "banner", "red", "qr", "sensors", "payload"])
    ap.add_argument("--exit", action="store_true", help="banner: look for the blue exit banner")
    ap.add_argument("--alt", type=float, default=5.0, help="qr: altitude to assume, m")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--qr-source"); ap.add_argument("--pi-source")
    a = ap.parse_args(argv)
    if a.what == "sensors":
        return run_sensors(a.seconds)
    if a.what == "payload":
        return run_payload()
    if a.what == "cameras":
        return run_cameras(open_source(a.qr_source, True), open_source(a.pi_source, False))
    if a.what == "qr":
        return run_qr(open_source(a.qr_source, True), a.alt, a.seconds)
    pi = open_source(a.pi_source, False)
    {"hsv": lambda: run_hsv(pi, a.seconds), "banner": lambda: run_banner(pi, a.exit, a.seconds), "red": lambda: run_red(pi, a.seconds)}[a.what]()


if __name__ == "__main__":
    sys.exit(main())