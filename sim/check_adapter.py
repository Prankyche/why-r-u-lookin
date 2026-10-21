"""
sim/check_adapter.py
Self-contained check: runs the existing corridor navigator through the adapter
using a minimal fake vehicle. Does not import test_navigator_offline.py.
"""
import math
import config

# Same offline settings the offline test uses
config.USE_POSITION_TARGET_NAV = False
config.CUSTOM_SIDESTEP_STEERING_ENABLED = True

from sim.sensor_sim import SimulatedCorridorSensors
from control.corridor_navigator import CorridorNavigator
from adapters.corridor_adapter import run_corridor


class FakeVehicle:
    """Minimal stand-in for Vehicle: integrates velocity commands, no MAVLink."""

    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self.vx = self.vy = self.vz = self.yaw_rate = 0.0

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        self.vx, self.vy, self.vz, self.yaw_rate = vx, vy, vz, yaw_rate

    def hold_position(self):
        self.vx = self.vy = self.vz = self.yaw_rate = 0.0

    def get_local_position(self):
        dt = 1.0 / config.CONTROL_LOOP_HZ
        self.yaw += self.yaw_rate * dt
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * c - self.vy * s) * dt
        self.y += (self.vx * s + self.vy * c) * dt
        self.z += self.vz * dt
        return (self.x, self.y, self.z)

    def get_yaw(self):
        return self.yaw

    def send_fake_distance_sensor(self, distance_cm, orientation, sensor_id):
        pass


if __name__ == "__main__":
    v = FakeVehicle()
    nav = CorridorNavigator(v, SimulatedCorridorSensors())
    print(run_corridor(nav, v, real_time=False))