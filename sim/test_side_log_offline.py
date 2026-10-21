"""test_side_log_offline.py -- run the navigator against a side-obstacle log, no MAVLink needed.

    python test_side_log_offline.py side_obstacle_log.txt [front_lidar_log.txt]

Uses velocity mode (the lateral PID pushes away from whichever side has the nearer obstacle).
The fake vehicle just integrates the commands; it does not emulate ArduPilot.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math, sys
import config
config.USE_POSITION_TARGET_NAV = False
config.CUSTOM_SIDESTEP_STEERING_ENABLED = False
config.SIM_OBSTACLES = []                       # front is clear; only the side log matters here
from control.corridor_navigator import CorridorNavigator
from side_obstacle_log import SideObstacleLogSensors, load_side_log



class FakeVehicle:
    DT = 1.0 / config.CONTROL_LOOP_HZ           # fixed step = the navigator's virtual clock

    def __init__(self):
        self.x = self.y = self.z = 0.0; self.yaw = 0.0
        self.vx = self.vy = 0.0

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0): self.vx, self.vy = vx, vy
    def hold_position(self): self.vx = self.vy = 0.0
    def send_fake_distance_sensor(self, *a, **k): pass
    def get_yaw(self): return self.yaw

    def get_local_position(self):
        self.x += self.vx * self.DT; self.y += self.vy * self.DT
        return (self.x, self.y, self.z)


class TracingLogger:
    def __init__(self): self.rows = []
    def log(self, **k): self.rows.append(k)
    def close(self): pass


path = sys.argv[1] if len(sys.argv) > 1 else config.SIDE_OBSTACLE_LOG_PATH
front_path = sys.argv[2] if len(sys.argv) > 2 else getattr(config, "FRONT_LIDAR_LOG_PATH", None)
left, right = load_side_log(path)
fmt = lambda v: f"{min(x for x in v if x is not None)*100:.1f} cm"
print(f"{len(left)} samples | left None {sum(x is None for x in left)}, min {fmt(left)} | "
      f"right None {sum(x is None for x in right)}, min {fmt(right)}")

if front_path:
    from side_obstacle_log import load_front_log
    fl = load_front_log(front_path, getattr(config, "FRONT_LIDAR_LOG_UNIT", "cm"))
    vals = [x for x in fl if x is not None]
    print(f"lidar log: {len(fl)} samples | None {len(fl) - len(vals)} | min {min(vals)*100:.1f} cm" if vals
          else f"lidar log: {len(fl)} samples, all None")

lg = TracingLogger()
nav = CorridorNavigator(FakeVehicle(), SideObstacleLogSensors(path, loop=False, front_path=front_path), logger=lg)
result = nav.run(real_time=False)
print(f"result: {result} | along {nav._along_corridor_m:.2f} m | lateral {nav._lateral_offset_m:+.2f} m | ticks {len(lg.rows)}")
worst = min(lg.rows, key=lambda r: min(r['left'], r['right']))
print(f"closest approach (centre to obstacle/wall): {min(worst['left'], worst['right']):.3f} m at t={worst['t']:.1f}s")
print("max |vy| commanded: %.2f m/s" % max(abs(r['vy']) for r in lg.rows))
print("slowest forward speed: %.2f m/s | min front reading: %.2f m" % (
    min(r['vx'] for r in lg.rows), min(r['front'] for r in lg.rows)))
sys.exit(0 if result == "success" else 1)