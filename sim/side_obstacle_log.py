"""side_obstacle_log.py

Replays a logged LEFT/RIGHT side-sensor printout that contains obstacles, e.g. lines like
    Left Obstacle: None cm  |        Right Obstacle: 10.5 cm
("None" = nothing detected on that side). Save the printout as a .txt file and point
config.SIDE_OBSTACLE_LOG_PATH at it, then set config.SENSOR_SOURCE = "obslog".

How the log is used (this is open-loop: the log does not react to where the drone goes):
  * walls come from the corridor geometry in sensor_sim (so they change as SITL moves the drone)
  * each logged value is an obstacle at that distance (cm) from that side sensor; the sensor
    reports whichever is NEARER, the wall or the obstacle. "None" = no obstacle = just the wall.
  * logged values are treated as RAW sensor readings: the calibration bias is added, then the
    mount offset, so left_m / right_m are drone-centre-to-obstacle like everywhere else.
  * front + obstacle_ahead still come from the simulated geometry (config.SIM_OBSTACLES).
  * OPTIONAL front lidar log (config.FRONT_LIDAR_LOG_PATH): if set, front_m / obstacle_ahead come from
    that log instead of the simulated geometry, through the same bias + median-filter pipeline the real
    lidar path uses. In SITL the navigator then injects those values to the FC, so BendyRuler reacts to
    the logged lidar obstacles.
One log sample is consumed per read() call (= one control tick, 10 Hz by default).
"""
import re
import config
from sensor_sim import SimulatedCorridorSensors
from drivers.tof_interface import RangeSensor, SensorConfig

_PAT = re.compile(
    r"Left Obstacle:\s*(None|-?[\d.]+)\s*cm\s*\|\s*Right Obstacle:\s*(None|-?[\d.]+)\s*cm")


def load_side_log(path):
    """Returns (left_m_list, right_m_list); an entry is None when the log said 'None'."""
    left, right = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _PAT.search(line)
            if not m:
                continue                       # skips 'Press Ctrl+C', blank lines, '^C'
            l, r = m.groups()
            left.append(None if l == "None" else float(l) / 100.0)
            right.append(None if r == "None" else float(r) / 100.0)
    return left, right


_FRONT_LABEL = re.compile(r"(?:Front|Lidar)[^:\d]*:\s*(None|-?\d+(?:\.\d+)?)", re.I)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_UNIT_TO_M = {"cm": 0.01, "mm": 0.001, "m": 1.0}


def load_front_log(path, unit="cm"):
    """Front lidar log -> list of metres (None = no return / nothing in range).
    Accepted line formats:  'Front Obstacle: 120.5 cm' | 'Lidar: 120.5' | '120.5' | 'None' | 'time, 120.5'
    Lines that only hold left/right values are skipped, so this can also read the combined side log.
    0, negative and >= 65535 (the TFmini 'no return' code) count as None."""
    scale = _UNIT_TO_M[unit]
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = _FRONT_LABEL.search(s)
            if m:
                tok = m.group(1)
            elif re.search(r"left|right", s, re.I):
                continue                                   # side-sensor line without a front value
            elif re.search(r"none", s, re.I):
                tok = "None"
            else:
                nums = _NUM.findall(s)
                if not nums:
                    continue                               # header / '^C' / text
                tok = nums[-1] if "," in s else nums[0]
            if tok.lower() == "none":
                out.append(None)
            else:
                v = float(tok)
                out.append(None if v <= 0 or v >= 65535 else v * scale)
    return out


class SideObstacleLogSensors:
    def __init__(self, path=None, loop=None, front_path=None):
        self.left_log, self.right_log = load_side_log(
            path or getattr(config, "SIDE_OBSTACLE_LOG_PATH", "side_obstacle_log.txt"))
        if not self.left_log:
            raise ValueError("No 'Left Obstacle: ... | Right Obstacle: ...' lines found in the log file.")
        self.n = len(self.left_log)
        self.loop = getattr(config, "SIDE_OBSTACLE_LOG_LOOP", False) if loop is None else loop
        self._i = 0
        fp = front_path if front_path is not None else getattr(config, "FRONT_LIDAR_LOG_PATH", None)
        self.front_log = (load_front_log(fp, getattr(config, "FRONT_LIDAR_LOG_UNIT", "cm"))
                          if fp else None)
        if fp and not self.front_log:
            raise ValueError(f"No lidar values found in {fp}")
        self._fi = 0
        # the lidar log is separate from the side log: by default it plays ONCE (a looping obstacle
        # would make BendyRuler dodge the same "new" obstacle over and over)
        self.front_loop = getattr(config, "FRONT_LIDAR_LOG_LOOP", False) if loop is None else loop
        self._front_raw_mm = None
        # same filter/bias pipeline as the real lidar path (real_sensors.py)
        self.front = RangeSensor(lambda: self._front_raw_mm, SensorConfig(
            offset_mm=config.FRONT_SENSOR_BIAS_MM, min_mm=1, max_mm=1_000_000, max_jump_mm=1_000_000))
        self.geo = SimulatedCorridorSensors()
        self.geo.noise_std = config.LEFT_SENSOR_NOISE_STD_MM / 1000.0   # real-sensor-sized wall noise

    def read(self, along_corridor_m, lateral_offset_m):
        g = self.geo.read(along_corridor_m, lateral_offset_m)

        i = self._i
        self._i += 1
        if i >= self.n:
            i = (i % self.n) if self.loop else None      # log finished: no more obstacles

        # wall distance as a sensor on the drone would see it (sim values are centre -> wall)
        wall_l = g["left_m"] - config.SENSOR_OFFSET_LEFT_M
        wall_r = g["right_m"] - config.SENSOR_OFFSET_RIGHT_M

        obs_l = obs_r = None
        if i is not None:
            if self.left_log[i] is not None:
                obs_l = self.left_log[i] + config.LEFT_SENSOR_BIAS_MM / 1000.0
            if self.right_log[i] is not None:
                obs_r = self.right_log[i] + config.RIGHT_SENSOR_BIAS_MM / 1000.0

        sensor_l = min(wall_l, obs_l) if obs_l is not None else wall_l
        sensor_r = min(wall_r, obs_r) if obs_r is not None else wall_r

        front_m, obstacle_ahead = g["front_m"], g["obstacle_ahead"]      # default: simulated geometry
        if self.front_log is not None:
            front_m, obstacle_ahead = self._front_from_log(along_corridor_m)

        return {
            "left_m": round(sensor_l + config.SENSOR_OFFSET_LEFT_M, 3),
            "right_m": round(sensor_r + config.SENSOR_OFFSET_RIGHT_M, 3),
            "front_m": round(front_m, 3),
            "obstacle_ahead": obstacle_ahead,
        }

    def _front_from_log(self, along_corridor_m):
        """Next lidar sample -> (front_m, obstacle_ahead), the same way real_sensors.py derives them."""
        j = self._fi
        self._fi += 1
        n = len(self.front_log)
        if j >= n:
            j = (j % n) if self.front_loop else None       # log finished: nothing ahead any more
        value_m = None if j is None else self.front_log[j]
        max_m = config.FRONT_LIDAR_MAX_RANGE_M
        if value_m is None:                                # no return = nothing in range = clear
            self._front_raw_mm = None
            return max_m, False
        self._front_raw_mm = value_m * 1000.0
        f = self.front.read()                              # + calibration bias, median filter
        front_m = min(f.range_mm / 1000.0, max_m) if f.valid else config.OBSTACLE_STOP_DISTANCE_M + 0.01
        # an obstacle only if it is meaningfully nearer than the corridor exit (position from the FC)
        exit_dist = config.CORRIDOR_LENGTH_M - along_corridor_m
        return front_m, (front_m < max_m and front_m < exit_dist - 0.5)