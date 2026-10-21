"""replay_sensors.py

Hardware-free test source for Mission Planner SITL. Uses the SAME pipeline as real_sensors
(bias correction -> median filter -> mount offset), but fed by:
  - corridor geometry from sensor_sim (so SITL motion changes the readings)
  - the real noise from your logged ToF data (L_/R_ docx logs), mean removed
  - raw readings are made to read LOW by the calibration bias, like the real sensors,
    so the correction code is exercised too.
Falls back to Gaussian noise (config *_NOISE_STD_MM) if a log can't be loaded.
Set config.SENSOR_SOURCE = "replay".
"""
import random
import statistics
import config
from sensor_sim import SimulatedCorridorSensors
from drivers.tof_interface import RangeSensor, SensorConfig


def _load_noise_mm(path):
    try:
        from logged_sensor import _extract_distances
        v = _extract_distances(path)
        if len(v) > 5:
            m = statistics.mean(v)
            return [x - m for x in v]
    except Exception as e:
        print(f"[replay] couldn't load {path} ({e}); using Gaussian noise instead.")
    return None


class _Raw:
    """Callable raw reader: true distance (set each tick) - bias + replayed noise, in mm."""
    def __init__(self, noise, std_mm, bias_mm):
        self.noise, self.std, self.bias = noise, std_mm, bias_mm
        self.true_mm, self.i = 0.0, 0

    def __call__(self):
        if self.noise:
            n = self.noise[self.i % len(self.noise)]
            self.i += 1
        else:
            n = random.gauss(0, self.std)
        return int(round(self.true_mm - self.bias + n))


class ReplaySensors:
    def __init__(self):
        self.geo = SimulatedCorridorSensors()
        self.geo.noise_std = 0.0          # noise comes from the replay instead
        try:
            import logged_sensor as ls
            l_path, r_path = ls.LOG_L_PATH, ls.LOG_R_PATH
        except Exception:
            l_path = r_path = None
        self._l = _Raw(_load_noise_mm(l_path) if l_path else None,
                       config.LEFT_SENSOR_NOISE_STD_MM, config.LEFT_SENSOR_BIAS_MM)
        self._r = _Raw(_load_noise_mm(r_path) if r_path else None,
                       config.RIGHT_SENSOR_NOISE_STD_MM, config.RIGHT_SENSOR_BIAS_MM)
        self._f = _Raw(None, config.FRONT_SENSOR_NOISE_STD_MM, config.FRONT_SENSOR_BIAS_MM)
        self.left = RangeSensor(self._l, SensorConfig(offset_mm=config.LEFT_SENSOR_BIAS_MM))
        self.right = RangeSensor(self._r, SensorConfig(offset_mm=config.RIGHT_SENSOR_BIAS_MM))
        self.front = RangeSensor(self._f, SensorConfig(
            offset_mm=config.FRONT_SENSOR_BIAS_MM, min_mm=100,
            max_mm=1_000_000, max_jump_mm=1_000_000))

    def read(self, along_corridor_m, lateral_offset_m):
        g = self.geo.read(along_corridor_m, lateral_offset_m)
        # sim values are center->wall; a real sensor reports sensor->wall
        self._l.true_mm = (g["left_m"] - config.SENSOR_OFFSET_LEFT_M) * 1000
        self._r.true_mm = (g["right_m"] - config.SENSOR_OFFSET_RIGHT_M) * 1000
        self._f.true_mm = g["front_m"] * 1000
        l, r, f = self.left.read(), self.right.read(), self.front.read()
        left_m = (l.range_mm / 1000 + config.SENSOR_OFFSET_LEFT_M) if l.valid else 0.0
        right_m = (r.range_mm / 1000 + config.SENSOR_OFFSET_RIGHT_M) if r.valid else 0.0
        front_m = min(f.range_mm / 1000, config.FRONT_LIDAR_MAX_RANGE_M) if f.valid \
            else config.OBSTACLE_STOP_DISTANCE_M + 0.01
        return {"left_m": round(left_m, 3), "right_m": round(right_m, 3),
                "front_m": round(front_m, 3), "obstacle_ahead": g["obstacle_ahead"]}