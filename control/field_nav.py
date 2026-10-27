"""
control/field_nav.py
--------------------
Field sweep planner for SEARCH_QR. Pure logic: no MAVLink, no camera.

Merges the GPS-fence lawnmower planner (l_sweep_fov.py) with the FSM sweep:
  * The field is 4 GPS corners (config.FIELD_FENCE_FILE or FIELD_CORNERS_LL), in any orientation.
    With no corners it falls back to a FIELD_WIDTH_M x FIELD_HEIGHT_M rectangle at FIELD_START_NE_M.
  * The drone can be ANYWHERE when the sweep starts: it flies to the first waypoint and starts there.
    The home corner is the fence corner nearest the drone.
  * Lane spacing comes from the QR camera footprint at sweep altitude and SWEEP_OVERLAP, rounded
    UP to a whole number of lanes (never wider than the footprint allows). Lanes sit in the middle
    of equal strips, so the field edges are covered.
  * One heading is held for the whole sweep (nose along the lanes) so the camera's across-track axis
    stays across the lanes on every lane.

NE = (north, east) metres in the vehicle's local NED frame.
"""
import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config

log = logging.getLogger("field_nav")

NE = Tuple[float, float]
LL = Tuple[float, float]
EARTH_R = 6378137.0


# ------------------------------------------------------------------ geo helpers
def ll_to_ne(lat, lon, lat0, lon0) -> NE:
    return (math.radians(lat - lat0) * EARTH_R,
            math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)))


def ne_to_ll(n, e, lat0, lon0) -> LL:
    return (lat0 + math.degrees(n / EARTH_R),
            lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0)))))


def parse_fence(text: str) -> List[LL]:
    """4 corners from 'lat lon' / 'lat,lon' lines or ';'-separated pairs (same format as l_sweep.py,
    so its --record-corners file works here)."""
    pts, decimals = [], []
    for line in text.replace(";", "\n").splitlines():
        line = line.split("#")[0].strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(nums) != 2:
            continue
        lat, lon = float(nums[0]), float(nums[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat or lon):
            pts.append((lat, lon))
            decimals += [len(x.split(".")[1]) if "." in x else 0 for x in nums]
    if len(pts) == 5 and math.dist(pts[0], pts[-1]) < 3e-6:        # closed polygon
        pts.pop()
    if len(pts) != 4:
        raise ValueError(f"Field fence needs exactly 4 corners, got {len(pts)}")
    if min(decimals) < 6:
        err = 0.5 * 10 ** -min(decimals) * math.pi / 180 * EARTH_R
        log.warning("corners have only %d decimal places: each can be off by up to %.1f m. "
                    "Use 7 decimals.", min(decimals), err)
    return pts


def load_field_corners() -> Optional[List[LL]]:
    """Corners from config: FIELD_FENCE_FILE first, then FIELD_CORNERS_LL, else None (rectangle fallback)."""
    path = getattr(config, "FIELD_FENCE_FILE", None)
    if path:
        with open(path) as f:
            return parse_fence(f.read())
    corners = getattr(config, "FIELD_CORNERS_LL", None)
    if corners:
        return parse_fence("\n".join(f"{a:.8f} {b:.8f}" for a, b in corners))
    return None


def inside(poly: List[NE], p: NE, tol: float = 0.05) -> bool:
    """Point in polygon (ray casting); points within `tol` m of an edge count as inside."""
    x, y = p
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        if math.hypot(x - ax - t * dx, y - ay - t * dy) <= tol:
            return True
    hit = False
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            hit = not hit
    return hit


# ------------------------------------------------------------------ camera footprint
def footprint_m(alt_m: float, across: Optional[str] = None) -> Tuple[float, float]:
    """(across-track, along-track) ground footprint of the QR camera looking straight down.
    across='h': the image WIDTH (long side) points across the flight direction; 'v': the image height does."""
    across = across or config.SWEEP_CAM_ACROSS
    w, h = config.QR_CAMERA_WIDTH, config.QR_CAMERA_HEIGHT
    f = (w / 2.0) / math.tan(math.radians(config.QR_HFOV_DEG) / 2.0)
    fw, fh = alt_m * w / f, alt_m * h / f
    return (fw, fh) if across == "h" else (fh, fw)


def lane_spacing_m(alt_m: Optional[float] = None) -> float:
    if config.SWEEP_SPACING_M is not None:
        return config.SWEEP_SPACING_M
    alt = config.DELIVERY_ZONE_ALTITUDE_M if alt_m is None else alt_m
    return footprint_m(alt)[0] * (1.0 - config.SWEEP_OVERLAP)


# ------------------------------------------------------------------ planner
@dataclass
class SweepPlan:
    waypoints: List[NE]
    yaw: float                     # heading held for the whole sweep (rad, NED, 0 = north)
    lanes: int
    lane_len: float
    spacing: float
    home_corner: int
    notes: List[str] = field(default_factory=list)


def rect_polygon(origin_ne: NE, width_e: float, height_n: float, offset_ne: NE = (0.0, 0.0)) -> List[NE]:
    n0, e0 = origin_ne[0] + offset_ne[0], origin_ne[1] + offset_ne[1]
    return [(n0, e0), (n0, e0 + width_e), (n0 + height_n, e0 + width_e), (n0 + height_n, e0)]


def plan_sweep(poly: List[NE], from_ne: NE, spacing: float, margin: float = 0.5,
               lane_edge: str = "long", max_start_dist: Optional[float] = None) -> SweepPlan:
    if len(poly) != 4:
        raise ValueError("the field needs 4 corners")
    if spacing < 0.5:
        raise ValueError("lane spacing must be at least 0.5 m")
    notes: List[str] = []
    dists = [math.dist(c, from_ne) for c in poly]
    h = dists.index(min(dists))
    if max_start_dist is not None and dists[h] > max_start_dist:
        raise ValueError(f"The drone is {dists[h]:.0f} m from the nearest field corner "
                         f"(limit {max_start_dist:.0f} m): check the corner coordinates.")
    H, A, B, C = poly[h], poly[(h + 1) % 4], poly[(h - 1) % 4], poly[(h + 2) % 4]
    va, vb = (A[0] - H[0], A[1] - H[1]), (B[0] - H[0], B[1] - H[1])
    la, lb = math.hypot(*va), math.hypot(*vb)
    if min(la, lb) < 2.0:
        raise ValueError("A field edge is shorter than 2 m: are the corners in order around the field?")
    cosang = (va[0] * vb[0] + va[1] * vb[1]) / (la * lb)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    gap = math.dist((H[0] + va[0] + vb[0], H[1] + va[1] + vb[1]), C)
    if abs(angle - 90) > 15 or gap > max(1.0, 0.1 * min(la, lb)):
        notes.append(f"corners are not a rectangle (home angle {angle:.0f} deg, 4th corner off by {gap:.1f} m)")

    if lane_edge == "short":
        along, across = (va, vb) if la < lb else (vb, va)
    else:
        along, across = (va, vb) if la >= lb else (vb, va)
    L, W = math.hypot(*along), math.hypot(*across)
    if 2 * margin >= L - 1.0:
        raise ValueError(f"margin {margin} m is too large for {L:.1f} m lanes")

    lanes = max(1, math.ceil(W / spacing - 1e-6))      # round UP: coverage is never lost
    eff = W / lanes
    s_near, s_far = margin / L, 1 - margin / L
    t = [(i + 0.5) / lanes for i in range(lanes)]

    def P(s, tt):
        return (H[0] + s * along[0] + tt * across[0], H[1] + s * along[1] + tt * across[1])

    pts = [P(s_near, t[0]), P(s_far, t[0])]
    end_s = s_far
    for i in range(1, lanes):
        pts.append(P(end_s, t[i]))
        end_s = s_near if end_s == s_far else s_far
        pts.append(P(end_s, t[i]))

    for p in pts:
        if not inside(poly, p):
            raise ValueError(f"planned waypoint N{p[0]:+.1f} E{p[1]:+.1f} is outside the field")
    yaw = math.atan2(along[1], along[0])
    return SweepPlan(pts, yaw, lanes, L - 2 * margin, eff, h, notes)


def sweep_speed(red_centred: bool) -> float:
    """One rule: red centred in the Pi camera view -> fast, otherwise normal."""
    return config.SWEEP_SPEED_BOOST_MS if red_centred else config.SWEEP_SPEED_NORMAL_MS


def path_length_m(points: List[NE], from_ne: Optional[NE] = None) -> float:
    pts = ([from_ne] if from_ne else []) + list(points)
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


class FieldSweep:
    """Plans on construction (raises ValueError if the plan is unusable) and walks the waypoints."""

    def __init__(self, poly: List[NE], from_ne: NE, arrival_radius: Optional[float] = None,
                 lane_edge: Optional[str] = None):
        self.poly = poly
        self.plan = plan_sweep(poly, from_ne, lane_spacing_m(), config.SWEEP_MARGIN_M,
                               lane_edge or config.SWEEP_LANE_EDGE, config.FIELD_MAX_START_DIST_M)
        self.waypoints = self.plan.waypoints
        self.yaw = self.plan.yaw
        self.radius = config.SWEEP_ARRIVAL_RADIUS_M if arrival_radius is None else arrival_radius
        self.index = 0
        p = self.plan
        log.info("sweep: %d lanes %.1f m apart, %.1f m long, home corner %d, %d waypoints, %.0f m path "
                 "from the drone, heading %.0f deg", p.lanes, p.spacing, p.lane_len, p.home_corner + 1,
                 len(self.waypoints), path_length_m(self.waypoints, from_ne), math.degrees(self.yaw))
        for n in p.notes:
            log.warning("sweep: %s", n)

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def target(self) -> Optional[NE]:
        return None if self.done else self.waypoints[self.index]

    def update(self, pos_ne: NE) -> Optional[NE]:
        """Advance past reached waypoints; return the current target, or None when finished."""
        while not self.done and math.dist(pos_ne, self.waypoints[self.index]) <= self.radius:
            self.index += 1
        return self.target()

    def contains(self, pos_ne: NE, margin: float = 0.0) -> bool:
        """True if pos_ne is inside the field (within `margin` m of the fence counts). The IN_ZONE check."""
        return inside(self.poly, pos_ne, tol=margin)