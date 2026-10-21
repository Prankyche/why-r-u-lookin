#qr white border
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode

PATCH = 96                              
WIDE_EXPAND = 1.125                      
WIDE = int(round(PATCH * WIDE_EXPAND))    

# --------------------------------------------------------------------------- config
@dataclass
class QRConfig:
    width: int = 1280
    height: int = 800
    fps: int = 30
    exposure_abs: Optional[int] = None  
    focal_mm: float = 6.0
    sensor_w_mm: float = 3.84
    qr_size_m: float = 1.0
    border_frac: float = 0.0762        
    box_scale: float = 0.25
    white_min: int = 140               
    box_squareness: float = 0.78       
    box_fill_min: float = 0.80         
    box_size_tol: float = 0.45         
    box_min_px: int = 40               
    box_max_px: int = 800
    inner_contrast_min: float = 80.0   
    inner_dark_range: Tuple[float, float] = (0.15, 0.75)
    # decoding
    fullframe_every: int = 4            
    crop_margin: float = 0.25            
    vote_window: int = 5
    vote_needed: int = 3
    match_threshold: float = 0.80
    match_margin: float = 0.12


# --------------------------------------------------------------------------- data
@dataclass(eq=False)
class QRHit:
    data: Optional[str]                  # decoded text, or None for an undecoded candidate
    quad: np.ndarray                     # (4, 2) float32, full-frame px, clockwise
    method: str                          # crop | rescue | direct (decoded) | box (undecoded)
    frame_wh: Tuple[int, int]
    patch: Optional[np.ndarray] = None   # PATCH x PATCH flat view of the code
    wide: Optional[np.ndarray] = None    # WIDE x WIDE view with margin (candidates only)
    score: float = 0.0
    t: float = field(default_factory=time.time)

    @property
    def center(self) -> np.ndarray:
        return self.quad.mean(axis=0)

    @property
    def size_px(self) -> float:
        q = self.quad
        return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))

    @property
    def offset_px(self) -> Tuple[float, float]:
        w, h = self.frame_wh
        c = self.center
        return float(c[0] - w / 2.0), float(c[1] - h / 2.0)


@dataclass(eq=False)
class StartQR:
    text: str
    reference: np.ndarray                # PATCH x PATCH image for the fallback matcher
    hit: QRHit


# --------------------------------------------------------------------------- geometry
def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def order_quad(pts) -> np.ndarray:
    """4 corners in a consistent clockwise order (start point arbitrary; matching tries 4 rotations)."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p))
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)].astype(np.float32)


def expand_quad(q: np.ndarray, f: float) -> np.ndarray:
    c = q.mean(axis=0)
    return (c + (q - c) * f).astype(np.float32)


def warp_patch(gray: np.ndarray, quad: np.ndarray, size: int) -> np.ndarray:
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(gray, m, (size, size), flags=cv2.INTER_LINEAR)


def match_score(reference: np.ndarray, wide: np.ndarray) -> float:
    """Best normalised correlation of the reference inside `wide`, over 4 rotations."""
    best = -1.0
    for k in range(4):
        r = np.ascontiguousarray(np.rot90(reference, k))
        res = cv2.matchTemplate(wide, r, cv2.TM_CCOEFF_NORMED)
        v = float(np.nanmax(res)) if res.size else -1.0
        if np.isfinite(v) and v > best:
            best = v
    return best


def ground_offset_m(hit: QRHit, altitude_m: float, cfg: QRConfig) -> Tuple[float, float]:
    """(dx, dy) metres of the QR centre from the image centre, in IMAGE axes
    (x right, y down). Map to body/NED axes according to how the camera is mounted."""
    w, _ = hit.frame_wh
    gsd = altitude_m * cfg.sensor_w_mm / (cfg.focal_mm * w)      # metres per pixel
    dx, dy = hit.offset_px
    return dx * gsd, dy * gsd


# --------------------------------------------------------------------------- pipeline
class QRPipeline:
    def __init__(self, cfg: Optional[QRConfig] = None):
        self.cfg = cfg or QRConfig()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._n = 0
        self._expected_box_px: Optional[float] = None
        self.last_hits: List[QRHit] = []
        self.start_votes: Deque[Optional[QRHit]] = deque(maxlen=self.cfg.vote_window)
        self.target_votes: Deque[bool] = deque(maxlen=self.cfg.vote_window)
        self.target_text: Optional[str] = None
        self.reference: Optional[np.ndarray] = None
        self.last_target_hit: Optional[QRHit] = None

    def set_altitude(self, altitude_m: Optional[float]) -> None:
        """Tell the box finder how big a QR box should look (None = accept any size)."""
        if not altitude_m:
            self._expected_box_px = None
            return
        c = self.cfg
        box_m = c.qr_size_m * (1.0 + 2.0 * c.border_frac)
        footprint_m = altitude_m * c.sensor_w_mm / c.focal_mm
        self._expected_box_px = box_m / footprint_m * c.width

    # ---- low level -------------------------------------------------------------
    @staticmethod
    def _zbar(img: np.ndarray):
        return zbar_decode(img, symbols=[ZBarSymbol.QRCODE])

    def _hits_from_zbar(self, results, method: str, gray_full: np.ndarray,
                        scale: float = 1.0, ox: float = 0.0, oy: float = 0.0) -> List[QRHit]:
        h, w = gray_full.shape[:2]
        hits = []
        for r in results:
            poly = np.array([[p.x, p.y] for p in r.polygon], np.float32)
            if len(poly) < 3:
                continue
            quad = (order_quad(poly) / scale + np.array([ox, oy], np.float32)).astype(np.float32)
            hits.append(QRHit(
                data=r.data.decode("utf-8", errors="replace"),
                quad=quad, method=method, frame_wh=(w, h),
                patch=warp_patch(gray_full, quad, PATCH)))
        return hits

    def _direct(self, gray: np.ndarray) -> List[QRHit]:
        """pyzbar on the whole frame (the safety net)."""
        return self._hits_from_zbar(self._zbar(gray), "direct", gray)

    # ---- white-box finder ------------------------------------------------------
    def _looks_like_qr(self, gray: np.ndarray, box: np.ndarray) -> bool:
        """Inside the white border there must be a high-contrast, mixed dark/light pattern."""
        c = self.cfg
        inner = expand_quad(box, 1.0 / (1.0 + 2.0 * c.border_frac))
        h, w = gray.shape[:2]
        x0, y0 = np.maximum(inner.min(axis=0).astype(int), 0)
        x1, y1 = np.minimum(inner.max(axis=0).astype(int), [w, h])
        crop = gray[y0:y1:2, x0:x1:2]
        if crop.size < 100:
            return False
        lo, hi = np.percentile(crop, (5, 95))
        if hi - lo < c.inner_contrast_min:
            return False
        dark = float((crop < (lo + hi) / 2.0).mean())
        return c.inner_dark_range[0] <= dark <= c.inner_dark_range[1]

    def _white_boxes(self, gray: np.ndarray) -> List[np.ndarray]:
        """Outer quads (full-frame px) of white-bordered, QR-looking squares."""
        c = self.cfg
        s = c.box_scale
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        otsu, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.threshold(small, max(otsu, c.white_min), 255, cv2.THRESH_BINARY)[1]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        exp = self._expected_box_px
        boxes = []
        for cnt in cnts:
            if len(cnt) < 4:
                continue
            rect = cv2.minAreaRect(cnt)
            bw, bh = rect[1]
            if min(bw, bh) < 2:
                continue
            side = 0.5 * (bw + bh) / s
            if exp:
                if abs(side - exp) > c.box_size_tol * exp:
                    continue
            elif not (c.box_min_px <= side <= c.box_max_px):
                continue
            if min(bw, bh) / max(bw, bh) < c.box_squareness:
                continue
            if cv2.contourArea(cnt) / (bw * bh) < c.box_fill_min:
                continue
            quad = order_quad(cv2.boxPoints(rect) / s)
            if self._looks_like_qr(gray, quad):
                boxes.append(quad)
        return boxes

    # ---- decoding inside a box -------------------------------------------------
    def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[QRHit]:
        """Plain decode of a small crop, then filtered decodes of that same crop."""
        h, w = gray.shape[:2]
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        m = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
        x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
        x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        up = 2.0 if max(crop.shape) < 240 else 1.0
        if up != 1.0:
            crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        res = self._zbar(crop)
        if res:
            return self._hits_from_zbar(res, "crop", gray, scale=up, ox=x0, oy=y0)
        c = self._clahe.apply(crop)
        sharp = cv2.addWeighted(c, 1.6, cv2.GaussianBlur(c, (0, 0), 2.0), -0.6, 0)
        res = self._zbar(sharp)
        if not res:
            res = self._zbar(cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
        if res:
            return self._hits_from_zbar(res, "rescue", gray, scale=up, ox=x0, oy=y0)
        return []

    @staticmethod
    def _duplicate(center: np.ndarray, size: float, hits: List[QRHit]) -> bool:
        return any(np.linalg.norm(center - h.center) < 0.5 * max(h.size_px, size, 1.0) for h in hits)

    def _scan(self, gray: np.ndarray, full: Optional[bool] = None) -> List[QRHit]:
        """One frame: white boxes -> decode inside -> (undecoded boxes kept for matching),
        plus the whole-frame safety net every `fullframe_every` frames (or when full=True)."""
        self._n += 1
        h, w = gray.shape[:2]
        hits: List[QRHit] = []
        b = self.cfg.border_frac
        for box in self._white_boxes(gray):
            qr = expand_quad(box, 1.0 / (1.0 + 2.0 * b))
            size = float(np.mean(np.linalg.norm(qr - np.roll(qr, -1, axis=0), axis=1)))
            if self._duplicate(qr.mean(axis=0), size, hits):
                continue
            decoded = self._decode_crop(gray, qr)
            if decoded:
                hits += [d for d in decoded if not self._duplicate(d.center, d.size_px, hits)]
            elif self.reference is not None:
                hits.append(QRHit(None, qr, "box", (w, h),
                                  wide=warp_patch(gray, expand_quad(qr, WIDE_EXPAND), WIDE)))
        if full is None:
            full = self._n % self.cfg.fullframe_every == 0
        if full:
            hits += [d for d in self._direct(gray) if not self._duplicate(d.center, d.size_px, hits)]
        self.last_hits = hits
        return hits

    def detect_all(self, gray: np.ndarray, full: bool = True) -> List[QRHit]:
        """Every QR in the frame (for testing / custom logic)."""
        return self._scan(gray, full=full)

    # ---- mission helpers -------------------------------------------------------
    def update_start_scan(self, gray: np.ndarray) -> Optional[StartQR]:
        """Call once per frame at ~5 m. Returns StartQR once the same text is seen in
        `vote_needed` of the last `vote_window` frames."""
        decoded = [h for h in self._scan(gray) if h.data]
        self.start_votes.append(max(decoded, key=lambda h: h.size_px) if decoded else None)
        texts = Counter(h.data.strip() for h in self.start_votes if h)
        if texts:
            text, cnt = texts.most_common(1)[0]
            if cnt >= self.cfg.vote_needed:
                latest = next(h for h in reversed(self.start_votes) if h and h.data.strip() == text)
                return StartQR(text, latest.patch, latest)
        return None

    def set_target(self, text: str, reference: Optional[np.ndarray] = None) -> None:
        self.target_text = text.strip()
        self.reference = reference
        self.target_votes.clear()
        self.last_target_hit = None

    def _text_match(self, h: QRHit) -> bool:
        """Decoded code: the exact text decides (never overridden by the image matcher)."""
        return self.target_text is not None and h.data is not None and h.data.strip() == self.target_text

    def update_target_search(self, gray: np.ndarray) -> Optional[QRHit]:
        """Call once per frame at ~10 m. Returns the target QRHit once it is confirmed
        in `vote_needed` of the last `vote_window` frames (check hit.t for freshness)."""
        hits = self._scan(gray)
        matches = [h for h in hits if self._text_match(h)]
        for h in matches:
            h.score = 1.0
        if not matches and self.reference is not None:
            # image-match fallback, only among codes that would not decode
            cands = [h for h in hits if h.data is None and h.wide is not None]
            for h in cands:
                h.score = match_score(self.reference, h.wide)
            cands.sort(key=lambda h: -h.score)
            if cands and cands[0].score >= self.cfg.match_threshold:
                runner_up = cands[1].score if len(cands) > 1 else -1.0
                if cands[0].score - runner_up >= self.cfg.match_margin:
                    matches = [cands[0]]

        self.target_votes.append(bool(matches))
        if matches:
            self.last_target_hit = max(matches, key=lambda h: (h.score, h.size_px))
        if sum(self.target_votes) >= self.cfg.vote_needed:
            return self.last_target_hit
        return None


# --------------------------------------------------------------------------- frame sources
class FrameGrabber:
    """Reads the camera on a background thread and always returns the NEWEST frame, so a
    slow detector never works on stale frames. Forces 1280x800 MJPG (OpenCV often
    defaults to 640x480 on USB cameras)."""

    def __init__(self, src=0, cfg: Optional[QRConfig] = None):
        cfg = cfg or QRConfig()
        backend = cv2.CAP_V4L2 if (isinstance(src, int) and sys.platform.startswith("linux")) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # set before size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cfg.exposure_abs is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                       # 1 = manual (V4L2)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_abs)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.size != (cfg.width, cfg.height):
            print(f"[WARN] camera delivers {self.size[0]}x{self.size[1]}, wanted {cfg.width}x{cfg.height}",
                  file=sys.stderr)
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._idx = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._idx = frame, self._idx + 1
            self._new.set()

    def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        with self._lock:
            return self._idx, self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def camera_frames(grabber: FrameGrabber) -> Iterator[Tuple[int, np.ndarray]]:
    stalls = 0
    while stalls < 5:
        item = grabber.read()
        if item is None:
            stalls += 1
            continue
        stalls = 0
        yield item


def video_frames(path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """Full-resolution frames from a recording (no resizing)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame
    cap.release()


# --------------------------------------------------------------------------- test harness
def _draw(frame: np.ndarray, hits: List[QRHit]) -> np.ndarray:
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for h in hits:
        col = (0, 200, 0) if h.data is not None else (0, 200, 255)
        cv2.polylines(img, [h.quad.astype(np.int32)], True, col, 2)
        label = f"{h.method} {h.data[:18] if h.data else f'{h.score:.2f}'}"
        cv2.putText(img, label, tuple(h.quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img


def run(args) -> None:
    cfg = QRConfig(exposure_abs=args.exposure)
    pipe = QRPipeline(cfg)
    pipe.set_altitude(args.altitude)
    if args.mode == "target":
        if not args.target:
            sys.exit("--mode target needs --target TEXT")
        pipe.set_target(args.target)

    grabber = FrameGrabber(args.camera, cfg) if args.camera is not None else None
    frames = camera_frames(grabber) if grabber else video_frames(args.video)

    times, n, with_hit, methods = [], 0, 0, Counter()
    t_begin = time.time()
    try:
        for idx, frame in frames:
            t0 = time.perf_counter()
            gray = to_gray(frame)
            result = None
            if args.mode == "raw":
                hits = pipe.detect_all(gray)
            elif args.mode == "start":
                result = pipe.update_start_scan(gray)
                hits = pipe.last_hits
            else:
                result = pipe.update_target_search(gray)
                hits = pipe.last_hits
            times.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            if hits:
                with_hit += 1
                methods.update(h.method for h in hits)
                if args.verbose:
                    for h in hits:
                        print(f"[{idx}] {h.method:9s} size={h.size_px:5.0f}px data={h.data!r} score={h.score:.2f}")
            if result is not None:
                if args.mode == "start":
                    print(f"START QR CONFIRMED at frame {idx}: {result.text!r}")
                else:
                    dx, dy = ground_offset_m(result, args.altitude, cfg)
                    print(f"TARGET CONFIRMED at frame {idx}: offset {dx:+.2f} m, {dy:+.2f} m (image axes)")
                break
            if args.show:
                cv2.imshow("qr", cv2.resize(_draw(frame, hits), None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_seconds and time.time() - t_begin > args.max_seconds:
                break
    finally:
        if grabber:
            grabber.close()
        if args.show:
            cv2.destroyAllWindows()

    if n:
        t = np.array(times)
        print(f"\nframes={n}  frames_with_QR={with_hit} ({100 * with_hit / n:.1f}%)  methods={dict(methods)}")
        print(f"ms/frame: p50={np.percentile(t, 50):.1f}  p95={np.percentile(t, 95):.1f}  max={t.max():.1f}"
              f"   ~{1000.0 / max(t.mean(), 1e-6):.1f} FPS")


def main() -> None:
    ap = argparse.ArgumentParser(description="AeroTHON QR detection test harness")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="camera index")
    src.add_argument("--video", help="recorded video (full resolution, no resizing)")
    ap.add_argument("--mode", choices=["raw", "start", "target"], default="raw",
                    help="raw: all hits | start: 3-of-5 start-QR scan | target: find --target text")
    ap.add_argument("--target", help="target QR text for --mode target")
    ap.add_argument("--altitude", type=float, default=10.0, help="metres: expected-size hint + offset printout")
    ap.add_argument("--exposure", type=int, default=None, help="manual exposure (V4L2 units)")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--show", action="store_true", help="preview window (needs a display)")
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
# from __future__ import annotations

# import argparse
# import sys
# import threading
# import time
# from collections import Counter, deque
# from dataclasses import dataclass, field
# from typing import Deque, Iterator, List, Optional, Tuple

# import cv2
# import numpy as np
# from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode

# PATCH = 96                              
# WIDE_EXPAND = 1.125                      
# WIDE = int(round(PATCH * WIDE_EXPAND))    

# # --------------------------------------------------------------------------- config
# @dataclass
# class QRConfig:
#     width: int = 1280
#     height: int = 800
#     fps: int = 30
#     exposure_abs: Optional[int] = None  
#     focal_mm: float = 6.0
#     sensor_w_mm: float = 3.84
#     qr_size_m: float = 1.0
#     border_frac: float = 0.0762        
#     box_scale: float = 0.25
#     white_min: int = 140               
#     box_squareness: float = 0.78       
#     box_fill_min: float = 0.80         
#     box_size_tol: float = 0.45         
#     box_min_px: int = 40               
#     box_max_px: int = 800
#     inner_contrast_min: float = 80.0   
#     inner_dark_range: Tuple[float, float] = (0.15, 0.75)
#     # decoding
#     fullframe_every: int = 4            
#     crop_margin: float = 0.25            
#     vote_window: int = 5
#     vote_needed: int = 3
#     match_threshold: float = 0.80
#     match_margin: float = 0.12


# # --------------------------------------------------------------------------- data
# @dataclass(eq=False)
# class QRHit:
#     data: Optional[str]                  # decoded text, or None for an undecoded candidate
#     quad: np.ndarray                     # (4, 2) float32, full-frame px, clockwise
#     method: str                          # crop | rescue | direct (decoded) | box (undecoded)
#     frame_wh: Tuple[int, int]
#     patch: Optional[np.ndarray] = None   # PATCH x PATCH flat view of the code
#     wide: Optional[np.ndarray] = None    # WIDE x WIDE view with margin (candidates only)
#     score: float = 0.0
#     t: float = field(default_factory=time.time)

#     @property
#     def center(self) -> np.ndarray:
#         return self.quad.mean(axis=0)

#     @property
#     def size_px(self) -> float:
#         q = self.quad
#         return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))

#     @property
#     def offset_px(self) -> Tuple[float, float]:
#         w, h = self.frame_wh
#         c = self.center
#         return float(c[0] - w / 2.0), float(c[1] - h / 2.0)


# @dataclass(eq=False)
# class StartQR:
#     text: str
#     reference: np.ndarray                # PATCH x PATCH image for the fallback matcher
#     hit: QRHit


# # --------------------------------------------------------------------------- geometry
# def to_gray(frame: np.ndarray) -> np.ndarray:
#     return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# def order_quad(pts) -> np.ndarray:
#     """4 corners in a consistent clockwise order (start point arbitrary; matching tries 4 rotations)."""
#     p = np.asarray(pts, np.float32).reshape(-1, 2)
#     if len(p) != 4:
#         p = cv2.boxPoints(cv2.minAreaRect(p))
#     c = p.mean(axis=0)
#     ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
#     return p[np.argsort(ang)].astype(np.float32)


# def expand_quad(q: np.ndarray, f: float) -> np.ndarray:
#     c = q.mean(axis=0)
#     return (c + (q - c) * f).astype(np.float32)


# def warp_patch(gray: np.ndarray, quad: np.ndarray, size: int) -> np.ndarray:
#     dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
#     m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
#     return cv2.warpPerspective(gray, m, (size, size), flags=cv2.INTER_LINEAR)


# def match_score(reference: np.ndarray, wide: np.ndarray) -> float:
#     """Best normalised correlation of the reference inside `wide`, over 4 rotations."""
#     best = -1.0
#     for k in range(4):
#         r = np.ascontiguousarray(np.rot90(reference, k))
#         res = cv2.matchTemplate(wide, r, cv2.TM_CCOEFF_NORMED)
#         v = float(np.nanmax(res)) if res.size else -1.0
#         if np.isfinite(v) and v > best:
#             best = v
#     return best


# def ground_offset_m(hit: QRHit, altitude_m: float, cfg: QRConfig) -> Tuple[float, float]:
#     """(dx, dy) metres of the QR centre from the image centre, in IMAGE axes
#     (x right, y down). Map to body/NED axes according to how the camera is mounted."""
#     w, _ = hit.frame_wh
#     gsd = altitude_m * cfg.sensor_w_mm / (cfg.focal_mm * w)      # metres per pixel
#     dx, dy = hit.offset_px
#     return dx * gsd, dy * gsd


# # --------------------------------------------------------------------------- pipeline
# class QRPipeline:
#     def __init__(self, cfg: Optional[QRConfig] = None):
#         self.cfg = cfg or QRConfig()
#         self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#         self._n = 0
#         self._expected_box_px: Optional[float] = None
#         self.last_hits: List[QRHit] = []
#         self.start_votes: Deque[Optional[QRHit]] = deque(maxlen=self.cfg.vote_window)
#         self.target_votes: Deque[bool] = deque(maxlen=self.cfg.vote_window)
#         self.target_text: Optional[str] = None
#         self.reference: Optional[np.ndarray] = None
#         self.last_target_hit: Optional[QRHit] = None

#     def set_altitude(self, altitude_m: Optional[float]) -> None:
#         """Tell the box finder how big a QR box should look (None = accept any size)."""
#         if not altitude_m:
#             self._expected_box_px = None
#             return
#         c = self.cfg
#         box_m = c.qr_size_m * (1.0 + 2.0 * c.border_frac)
#         footprint_m = altitude_m * c.sensor_w_mm / c.focal_mm
#         self._expected_box_px = box_m / footprint_m * c.width

#     # ---- low level -------------------------------------------------------------
#     @staticmethod
#     def _zbar(img: np.ndarray):
#         return zbar_decode(img, symbols=[ZBarSymbol.QRCODE])

#     def _hits_from_zbar(self, results, method: str, gray_full: np.ndarray,
#                         scale: float = 1.0, ox: float = 0.0, oy: float = 0.0) -> List[QRHit]:
#         h, w = gray_full.shape[:2]
#         hits = []
#         for r in results:
#             poly = np.array([[p.x, p.y] for p in r.polygon], np.float32)
#             if len(poly) < 3:
#                 continue
#             quad = (order_quad(poly) / scale + np.array([ox, oy], np.float32)).astype(np.float32)
#             hits.append(QRHit(
#                 data=r.data.decode("utf-8", errors="replace"),
#                 quad=quad, method=method, frame_wh=(w, h),
#                 patch=warp_patch(gray_full, quad, PATCH)))
#         return hits

#     def _direct(self, gray: np.ndarray) -> List[QRHit]:
#         """pyzbar on the whole frame (the safety net)."""
#         return self._hits_from_zbar(self._zbar(gray), "direct", gray)

#     # ---- white-box finder ------------------------------------------------------
#     def _looks_like_qr(self, gray: np.ndarray, box: np.ndarray) -> bool:
#         """Inside the white border there must be a high-contrast, mixed dark/light pattern."""
#         c = self.cfg
#         inner = expand_quad(box, 1.0 / (1.0 + 2.0 * c.border_frac))
#         h, w = gray.shape[:2]
#         x0, y0 = np.maximum(inner.min(axis=0).astype(int), 0)
#         x1, y1 = np.minimum(inner.max(axis=0).astype(int), [w, h])
#         crop = gray[y0:y1:2, x0:x1:2]
#         if crop.size < 100:
#             return False
#         lo, hi = np.percentile(crop, (5, 95))
#         if hi - lo < c.inner_contrast_min:
#             return False
#         dark = float((crop < (lo + hi) / 2.0).mean())
#         return c.inner_dark_range[0] <= dark <= c.inner_dark_range[1]

#     def _white_boxes(self, gray: np.ndarray) -> List[np.ndarray]:
#         """Outer quads (full-frame px) of white-bordered, QR-looking squares."""
#         c = self.cfg
#         s = c.box_scale
#         small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
#         otsu, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         mask = cv2.threshold(small, max(otsu, c.white_min), 255, cv2.THRESH_BINARY)[1]
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         exp = self._expected_box_px
#         boxes = []
#         for cnt in cnts:
#             if len(cnt) < 4:
#                 continue
#             rect = cv2.minAreaRect(cnt)
#             bw, bh = rect[1]
#             if min(bw, bh) < 2:
#                 continue
#             side = 0.5 * (bw + bh) / s
#             if exp:
#                 if abs(side - exp) > c.box_size_tol * exp:
#                     continue
#             elif not (c.box_min_px <= side <= c.box_max_px):
#                 continue
#             if min(bw, bh) / max(bw, bh) < c.box_squareness:
#                 continue
#             if cv2.contourArea(cnt) / (bw * bh) < c.box_fill_min:
#                 continue
#             quad = order_quad(cv2.boxPoints(rect) / s)
#             if self._looks_like_qr(gray, quad):
#                 boxes.append(quad)
#         return boxes

#     # ---- decoding inside a box -------------------------------------------------
#     def _decode_crop(self, gray: np.ndarray, quad: np.ndarray) -> List[QRHit]:
#         """Plain decode of a small crop, then filtered decodes of that same crop."""
#         h, w = gray.shape[:2]
#         x0, y0 = quad.min(axis=0)
#         x1, y1 = quad.max(axis=0)
#         m = self.cfg.crop_margin * max(x1 - x0, y1 - y0)
#         x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
#         x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
#         crop = gray[y0:y1, x0:x1]
#         if crop.size == 0:
#             return []
#         up = 2.0 if max(crop.shape) < 240 else 1.0
#         if up != 1.0:
#             crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
#         res = self._zbar(crop)
#         if res:
#             return self._hits_from_zbar(res, "crop", gray, scale=up, ox=x0, oy=y0)
#         c = self._clahe.apply(crop)
#         sharp = cv2.addWeighted(c, 1.6, cv2.GaussianBlur(c, (0, 0), 2.0), -0.6, 0)
#         res = self._zbar(sharp)
#         if not res:
#             res = self._zbar(cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
#         if res:
#             return self._hits_from_zbar(res, "rescue", gray, scale=up, ox=x0, oy=y0)
#         return []

#     @staticmethod
#     def _duplicate(center: np.ndarray, size: float, hits: List[QRHit]) -> bool:
#         return any(np.linalg.norm(center - h.center) < 0.5 * max(h.size_px, size, 1.0) for h in hits)

#     def _scan(self, gray: np.ndarray, full: Optional[bool] = None) -> List[QRHit]:
#         """One frame: white boxes -> decode inside -> (undecoded boxes kept for matching),
#         plus the whole-frame safety net every `fullframe_every` frames (or when full=True)."""
#         self._n += 1
#         h, w = gray.shape[:2]
#         hits: List[QRHit] = []
#         b = self.cfg.border_frac
#         for box in self._white_boxes(gray):
#             qr = expand_quad(box, 1.0 / (1.0 + 2.0 * b))
#             size = float(np.mean(np.linalg.norm(qr - np.roll(qr, -1, axis=0), axis=1)))
#             if self._duplicate(qr.mean(axis=0), size, hits):
#                 continue
#             decoded = self._decode_crop(gray, qr)
#             if decoded:
#                 hits += [d for d in decoded if not self._duplicate(d.center, d.size_px, hits)]
#             elif self.reference is not None:
#                 hits.append(QRHit(None, qr, "box", (w, h),
#                                   wide=warp_patch(gray, expand_quad(qr, WIDE_EXPAND), WIDE)))
#         if full is None:
#             full = self._n % self.cfg.fullframe_every == 0
#         if full:
#             hits += [d for d in self._direct(gray) if not self._duplicate(d.center, d.size_px, hits)]
#         self.last_hits = hits
#         return hits

#     def detect_all(self, gray: np.ndarray, full: bool = True) -> List[QRHit]:
#         """Every QR in the frame (for testing / custom logic)."""
#         return self._scan(gray, full=full)

#     # ---- mission helpers -------------------------------------------------------
#     def update_start_scan(self, gray: np.ndarray) -> Optional[StartQR]:
#         """Call once per frame at ~5 m. Returns StartQR once the same text is seen in
#         `vote_needed` of the last `vote_window` frames."""
#         decoded = [h for h in self._scan(gray) if h.data]
#         self.start_votes.append(max(decoded, key=lambda h: h.size_px) if decoded else None)
#         texts = Counter(h.data.strip() for h in self.start_votes if h)
#         if texts:
#             text, cnt = texts.most_common(1)[0]
#             if cnt >= self.cfg.vote_needed:
#                 latest = next(h for h in reversed(self.start_votes) if h and h.data.strip() == text)
#                 return StartQR(text, latest.patch, latest)
#         return None

#     def set_target(self, text: str, reference: Optional[np.ndarray] = None) -> None:
#         self.target_text = text.strip()
#         self.reference = reference
#         self.target_votes.clear()
#         self.last_target_hit = None

#     def _text_match(self, h: QRHit) -> bool:
#         """Decoded code: the exact text decides (never overridden by the image matcher)."""
#         return self.target_text is not None and h.data is not None and h.data.strip() == self.target_text

#     def update_target_search(self, gray: np.ndarray) -> Optional[QRHit]:
#         """Call once per frame at ~10 m. Returns the target QRHit once it is confirmed
#         in `vote_needed` of the last `vote_window` frames (check hit.t for freshness)."""
#         hits = self._scan(gray)
#         matches = [h for h in hits if self._text_match(h)]
#         for h in matches:
#             h.score = 1.0
#         if not matches and self.reference is not None:
#             # image-match fallback, only among codes that would not decode
#             cands = [h for h in hits if h.data is None and h.wide is not None]
#             for h in cands:
#                 h.score = match_score(self.reference, h.wide)
#             cands.sort(key=lambda h: -h.score)
#             if cands and cands[0].score >= self.cfg.match_threshold:
#                 runner_up = cands[1].score if len(cands) > 1 else -1.0
#                 if cands[0].score - runner_up >= self.cfg.match_margin:
#                     matches = [cands[0]]

#         self.target_votes.append(bool(matches))
#         if matches:
#             self.last_target_hit = max(matches, key=lambda h: (h.score, h.size_px))
#         if sum(self.target_votes) >= self.cfg.vote_needed:
#             return self.last_target_hit
#         return None


# # --------------------------------------------------------------------------- frame sources
# class FrameGrabber:
#     """Reads the camera on a background thread and always returns the NEWEST frame, so a
#     slow detector never works on stale frames. Forces 1280x800 MJPG (OpenCV often
#     defaults to 640x480 on USB cameras)."""

#     def __init__(self, src=0, cfg: Optional[QRConfig] = None):
#         cfg = cfg or QRConfig()
#         backend = cv2.CAP_V4L2 if (isinstance(src, int) and sys.platform.startswith("linux")) else cv2.CAP_ANY
#         self.cap = cv2.VideoCapture(src, backend)
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Cannot open camera {src}")
#         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # set before size
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
#         self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         if cfg.exposure_abs is not None:
#             self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)                       # 1 = manual (V4L2)
#             self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure_abs)
#         self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         if self.size != (cfg.width, cfg.height):
#             print(f"[WARN] camera delivers {self.size[0]}x{self.size[1]}, wanted {cfg.width}x{cfg.height}",
#                   file=sys.stderr)
#         self._lock = threading.Lock()
#         self._new = threading.Event()
#         self._stop = threading.Event()
#         self._frame: Optional[np.ndarray] = None
#         self._idx = 0
#         self._thread = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()

#     def _loop(self) -> None:
#         while not self._stop.is_set():
#             ok, frame = self.cap.read()
#             if not ok:
#                 time.sleep(0.01)
#                 continue
#             with self._lock:
#                 self._frame, self._idx = frame, self._idx + 1
#             self._new.set()

#     def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
#         if not self._new.wait(timeout):
#             return None
#         self._new.clear()
#         with self._lock:
#             return self._idx, self._frame

#     def close(self) -> None:
#         self._stop.set()
#         self._thread.join(timeout=1.0)
#         self.cap.release()

#     def __enter__(self):
#         return self

#     def __exit__(self, *exc):
#         self.close()


# def camera_frames(grabber: FrameGrabber) -> Iterator[Tuple[int, np.ndarray]]:
#     stalls = 0
#     while stalls < 5:
#         item = grabber.read()
#         if item is None:
#             stalls += 1
#             continue
#         stalls = 0
#         yield item


# def video_frames(path: str) -> Iterator[Tuple[int, np.ndarray]]:
#     """Full-resolution frames from a recording (no resizing)."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video {path}")
#     idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         idx += 1
#         yield idx, frame
#     cap.release()


# # --------------------------------------------------------------------------- test harness
# def _draw(frame: np.ndarray, hits: List[QRHit]) -> np.ndarray:
#     img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     for h in hits:
#         col = (0, 200, 0) if h.data is not None else (0, 200, 255)
#         cv2.polylines(img, [h.quad.astype(np.int32)], True, col, 2)
#         label = f"{h.method} {h.data[:18] if h.data else f'{h.score:.2f}'}"
#         cv2.putText(img, label, tuple(h.quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
#     return img


# def run(args) -> None:
#     cfg = QRConfig(exposure_abs=args.exposure)
#     pipe = QRPipeline(cfg)
#     pipe.set_altitude(args.altitude)
#     if args.mode == "target":
#         if not args.target:
#             sys.exit("--mode target needs --target TEXT")
#         pipe.set_target(args.target)

#     grabber = FrameGrabber(args.camera, cfg) if args.camera is not None else None
#     frames = camera_frames(grabber) if grabber else video_frames(args.video)

#     times, n, with_hit, methods = [], 0, 0, Counter()
#     t_begin = time.time()
#     try:
#         for idx, frame in frames:
#             t0 = time.perf_counter()
#             gray = to_gray(frame)
#             result = None
#             if args.mode == "raw":
#                 hits = pipe.detect_all(gray)
#             elif args.mode == "start":
#                 result = pipe.update_start_scan(gray)
#                 hits = pipe.last_hits
#             else:
#                 result = pipe.update_target_search(gray)
#                 hits = pipe.last_hits
#             times.append((time.perf_counter() - t0) * 1000.0)
#             n += 1
#             if hits:
#                 with_hit += 1
#                 methods.update(h.method for h in hits)
#                 if args.verbose:
#                     for h in hits:
#                         print(f"[{idx}] {h.method:9s} size={h.size_px:5.0f}px data={h.data!r} score={h.score:.2f}")
#             if result is not None:
#                 if args.mode == "start":
#                     print(f"START QR CONFIRMED at frame {idx}: {result.text!r}")
#                 else:
#                     dx, dy = ground_offset_m(result, args.altitude, cfg)
#                     print(f"TARGET CONFIRMED at frame {idx}: offset {dx:+.2f} m, {dy:+.2f} m (image axes)")
#                 break
#             if args.show:
#                 cv2.imshow("qr", cv2.resize(_draw(frame, hits), None, fx=0.5, fy=0.5))
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break
#             if args.max_seconds and time.time() - t_begin > args.max_seconds:
#                 break
#     finally:
#         if grabber:
#             grabber.close()
#         if args.show:
#             cv2.destroyAllWindows()

#     if n:
#         t = np.array(times)
#         print(f"\nframes={n}  frames_with_QR={with_hit} ({100 * with_hit / n:.1f}%)  methods={dict(methods)}")
#         print(f"ms/frame: p50={np.percentile(t, 50):.1f}  p95={np.percentile(t, 95):.1f}  max={t.max():.1f}"
#               f"   ~{1000.0 / max(t.mean(), 1e-6):.1f} FPS")


# def main() -> None:
#     ap = argparse.ArgumentParser(description="AeroTHON QR detection test harness")
#     src = ap.add_mutually_exclusive_group(required=True)
#     src.add_argument("--camera", type=int, help="camera index")
#     src.add_argument("--video", help="recorded video (full resolution, no resizing)")
#     ap.add_argument("--mode", choices=["raw", "start", "target"], default="raw",
#                     help="raw: all hits | start: 3-of-5 start-QR scan | target: find --target text")
#     ap.add_argument("--target", help="target QR text for --mode target")
#     ap.add_argument("--altitude", type=float, default=10.0, help="metres: expected-size hint + offset printout")
#     ap.add_argument("--exposure", type=int, default=None, help="manual exposure (V4L2 units)")
#     ap.add_argument("--max-seconds", type=float, default=0.0)
#     ap.add_argument("--show", action="store_true", help="preview window (needs a display)")
#     ap.add_argument("--verbose", action="store_true")
#     run(ap.parse_args())


# if __name__ == "__main__":
#     main()