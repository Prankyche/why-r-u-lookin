"""
test_navigator_offline.py
Complete working version with FakeVehicle that actually integrates velocities.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import time
import config
# This fake vehicle can't emulate ArduPilot's BendyRuler, so test the Python-side logic only:
config.USE_POSITION_TARGET_NAV = False
config.CUSTOM_SIDESTEP_STEERING_ENABLED = True
from sensor_sim import SimulatedCorridorSensors
from control.corridor_navigator import CorridorNavigator, CorridorResult
from flight_logs.flight_logger import FlightLogger



class FakeVehicle:
    """Mimics Vehicle for offline testing (no MAVLink)."""
    
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.yaw = 0.0
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0
        self.last_update_time = time.time()

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        """Store velocity command."""
        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz
        self.last_yaw_rate = yaw_rate

    def hold_position(self):
        """Stop all motion."""
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0

    def get_local_position(self):
        """Integrate velocities into position (body frame -> NED)."""
        # fixed step = the navigator's virtual clock when run(real_time=False); using wall-clock
        # time here made the vehicle barely move, so the old test always timed out at 0.00 m
        dt = 1.0 / config.CONTROL_LOOP_HZ
        
        self.yaw += self.last_yaw_rate * dt
        
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        vx_ned = self.last_vx * cos_yaw - self.last_vy * sin_yaw
        vy_ned = self.last_vx * sin_yaw + self.last_vy * cos_yaw
        
        self.x += vx_ned * dt
        self.y += vy_ned * dt
        self.z += self.last_vz * dt
        
        return (self.x, self.y, self.z)

    def get_yaw(self):
        """Return current heading."""
        return self.yaw

    def send_fake_distance_sensor(self, distance_cm, orientation, sensor_id):
        """No-op in offline mode."""
        pass

    def set_mode(self, mode_name):
        """No-op."""
        pass

    def arm(self):
        """No-op; always return True."""
        return True

    def disarm(self):
        """No-op."""
        pass

    def takeoff(self, altitude_m):
        """Fake takeoff (just move to altitude)."""
        self.z = -altitude_m

    def get_relative_altitude(self, apply_sensor_bias=False):
        """Return altitude (absolute value of Z)."""
        return abs(self.z)


def main():
    print("\n" + "="*70)
    print("OFFLINE CORRIDOR NAVIGATION TEST")
    print("="*70)
    
    vehicle = FakeVehicle()
    sensors = SimulatedCorridorSensors()
    logger = FlightLogger("offline_test_log.csv")

    navigator = CorridorNavigator(vehicle, sensors, logger=logger)
    result = navigator.run(real_time=False)
    logger.close()

    print(f"\n=== RESULT: {result} ===")
    print(f"Final distance: {navigator._along_corridor_m:.2f} m")
    print(f"Final lateral offset: {navigator._lateral_offset_m:.2f} m")
    
    try_plot("offline_test_log.csv")


def try_plot(csv_path):
    """Plot the flight results if matplotlib is available."""
    try:
        import csv as csv_module
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; pip install matplotlib to enable plots)")
        return

    print("\nReading log file...")
    t, along, lateral, left, right, front, vx, vy = [], [], [], [], [], [], [], []
    
    try:
        with open(csv_path, newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                t.append(float(row["t"]))
                along.append(float(row["along_m"]))
                lateral.append(float(row["lateral_m"]))
                left.append(float(row["left_m"]))
                right.append(float(row["right_m"]))
                front.append(float(row["front_m"]))
                vx.append(float(row["vx"]))
                vy.append(float(row["vy"]))
    except Exception as e:
        print(f"Error reading log: {e}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(along, lateral, 'b-', linewidth=2, label="Drone lateral offset")
    axes[0].axhline(config.CORRIDOR_WIDTH_M / 2, color="r", linestyle="--", alpha=0.7)
    axes[0].axhline(-config.CORRIDOR_WIDTH_M / 2, color="r", linestyle="--", alpha=0.7)
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Lateral offset (m)", fontsize=11)
    axes[0].set_title("Corridor Navigation — Lateral Control", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, front, 'r-', linewidth=2, label="Front")
    axes[1].plot(t, left, 'g-', linewidth=1.5, label="Left", alpha=0.7)
    axes[1].plot(t, right, 'b-', linewidth=1.5, label="Right", alpha=0.7)
    axes[1].axhline(config.OBSTACLE_STOP_DISTANCE_M, color="orange", linestyle="--")
    axes[1].set_ylabel("Distance (m)", fontsize=11)
    axes[1].set_title("Sensor Readings — Obstacle Detection", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 4)

    axes[2].plot(t, vx, 'g-', linewidth=2, label="Forward velocity (vx)")
    axes[2].plot(t, vy, 'b-', linewidth=2, label="Lateral velocity (vy)")
    axes[2].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Velocity (m/s)", fontsize=11)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_title("Control Commands — Velocity Profile", fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("offline_test_plot.png", dpi=150, bbox_inches='tight')
    print("✓ Plot saved to offline_test_plot.png")
    
    if t:
        print(f"\n=== FLIGHT SUMMARY ===")
        print(f"Duration: {t[-1]:.1f} s")
        print(f"Distance: {along[-1]:.2f} m / {config.CORRIDOR_LENGTH_M} m")
        print(f"Max lateral offset: {max(abs(min(lateral)), abs(max(lateral))):.3f} m")
        print(f"Avg speed: {along[-1] / t[-1]:.3f} m/s")


if __name__ == "__main__":
    main()