"""real_sensors.py

Drop-in replacement for SimulatedCorridorSensors / ReplaySensors on the real drone.
Same interface: read(along_corridor_m, lateral_offset_m) ->
    {left_m, right_m, front_m, obstacle_ahead, front_valid}

  left / right : 2x VL53L1X on the Pi (I2C, separate addresses via XSHUT)
  front        : lidar wired to the FC, read from its DISTANCE_SENSOR MAVLink messages
  If config.STREAM_SIDE_TOF_TO_FC is on, every read() also streams the filtered left/right values
  to the FC as DISTANCE_SENSOR (orientation YAW_270 / YAW_90), so they show up in Mission Planner.
All values are CENTER-referenced (sensor reading + calibration bias + mount offset), like sensor_sim.

front_valid is False while the lidar has produced nothing usable; the navigator aborts if that
lasts longer than config.FRONT_LIDAR_SILENT_ABORT_S.
"""
import time
import config
from tof_interface import (RangeSensor, SensorConfig,
                           make_vl53l1x_pair, make_adafruit_vl53l1x_pair)

DISTANCE_SENSOR_MSG_ID = 132


class RealCorridorSensors:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        if config.TOF_LIBRARY == "adafruit":
            left_raw, right_raw = make_adafruit_vl53l1x_pair(
                config.TOF_LEFT_XSHUT_GPIO, config.TOF_RIGHT_XSHUT_GPIO,
                config.TOF_LEFT_ADDR, config.TOF_RIGHT_ADDR,
                mode=config.TOF_DISTANCE_MODE, timing_ms=config.TOF_TIMING_BUDGET_MS,
                i2c_bus=config.TOF_I2C_BUS)
        else:
            left_raw, right_raw = make_vl53l1x_pair(
                config.TOF_LEFT_XSHUT_GPIO, config.TOF_RIGHT_XSHUT_GPIO,
                config.TOF_LEFT_ADDR, config.TOF_RIGHT_ADDR, config.TOF_I2C_BUS,
                mode=config.TOF_DISTANCE_MODE)
        self.left = RangeSensor(left_raw, SensorConfig(
            offset_mm=config.LEFT_SENSOR_BIAS_MM, min_mm=20, max_mm=4000))
        self.right = RangeSensor(right_raw, SensorConfig(
            offset_mm=config.RIGHT_SENSOR_BIAS_MM, min_mm=20, max_mm=4000))
        # Front: max-range ("nothing ahead") must pass through as a valid clear reading,
        # and big jumps are legitimate (obstacle appears / corridor exit), so loosen both.
        self.front = RangeSensor(self._front_raw, SensorConfig(
            offset_mm=config.FRONT_SENSOR_BIAS_MM, min_mm=100,
            max_mm=1_000_000, max_jump_mm=1_000_000))
        self._last_stream = 0.0
        # The FC reports each rangefinder instance (front + the two MAVLink side ones) in DISTANCE_SENSOR;
        # ask for a higher rate when it carries three, so the front lidar keeps ~10 Hz of its own.
        stream = getattr(config, "STREAM_SIDE_TOF_TO_FC", False)
        vehicle.request_message_interval(DISTANCE_SENSOR_MSG_ID, 30 if stream else 10)

    def _stream_sides(self, l, r):
        """Send the filtered, bias-corrected side readings (sensor-referenced, cm) to the FC."""
        if not getattr(config, "STREAM_SIDE_TOF_TO_FC", False):
            return
        now = time.monotonic()
        if now - self._last_stream < 1.0 / config.STREAM_SIDE_TOF_HZ:
            return
        self._last_stream = now
        for rd, orient, sid in ((l, config.TOF_LEFT_ORIENTATION, config.TOF_LEFT_SENSOR_ID),
                                (r, config.TOF_RIGHT_ORIENTATION, config.TOF_RIGHT_SENSOR_ID)):
            if rd.valid:                      # no reading -> send nothing; the FC times the sensor out
                self.vehicle.send_distance_sensor(
                    rd.range_mm / 10.0, orient, sid,
                    config.TOF_STREAM_MIN_CM, config.TOF_STREAM_MAX_CM)

    def service(self):
        """Read + stream once. Called while main.py waits (ENTER prompt, arming, takeoff, descent) so
        the side data keeps flowing to the FC even when the navigator isn't running."""
        self.read(0.0, 0.0)

    def _front_raw(self):
        return self.vehicle.get_distance_sensor_mm(config.FRONT_LIDAR_ORIENTATION)

    def _side_m(self, reading, mount_offset_m):
        """Centre-to-wall distance for one side, applying the no-reading policy."""
        if reading.valid:
            return reading.range_mm / 1000.0 + mount_offset_m
        if config.TOF_NONE_MEANS_CLEAR:
            return config.TOF_CLEAR_RANGE_M + mount_offset_m     # nothing in range = open side
        return 0.0       # lost sensor -> 0 so the MIN_WALL_CLEARANCE check aborts and holds (fail-safe)

    def read(self, along_corridor_m, lateral_offset_m):
        l, r, f = self.left.read(), self.right.read(), self.front.read()
        self._stream_sides(l, r)
        left_m = self._side_m(l, config.SENSOR_OFFSET_LEFT_M)
        right_m = self._side_m(r, config.SENSOR_OFFSET_RIGHT_M)

        if f.valid:
            front_m = min(f.range_mm / 1000.0, config.FRONT_LIDAR_MAX_RANGE_M)
        else:
            # Lidar silent: don't assume clear. Report just above the stop distance so the
            # speed ramp drops to crawl speed instead of cruising blind (or freezing forever).
            front_m = config.OBSTACLE_STOP_DISTANCE_M + 0.01

        # Only treat it as an obstacle if it's meaningfully closer than the corridor exit
        # (the sim knows this from geometry; here we use the distance to the exit from position).
        exit_dist = config.CORRIDOR_LENGTH_M - along_corridor_m
        obstacle_ahead = (f.valid and front_m < config.FRONT_LIDAR_MAX_RANGE_M
                          and front_m < exit_dist - 0.5)

        return {
            "left_m": round(left_m, 3),
            "right_m": round(right_m, 3),
            "front_m": round(front_m, 3),
            "obstacle_ahead": obstacle_ahead,
            "front_valid": f.valid,
            "left_valid": l.valid,
            "right_valid": r.valid,
        }