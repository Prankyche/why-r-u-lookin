"""
logged_sensor.py
----------------
Provides left and right distance sensor readings from logged TOF sensor data
(stored in .docx files). Front sensor and obstacle detection are still simulated
using the same geometry as sensor_sim.SimulatedCorridorSensors.
"""

import re
import time
import random
import config
from typing import Dict

# Paths to the logged sensor data (adjust if needed)
LOG_L_PATH = r"C:\Users\Shripriya Sriram\Downloads\L_Sensor Dataset.docx"
LOG_R_PATH = r"C:\Users\Shripriya Sriram\Downloads\R_sensor dataset.docx"

def _extract_distances(docx_path: str) -> list[float]:
    """Extract measured distances (in mm) from a .docx file."""
    import docx
    doc = docx.Document(docx_path)
    distances = []
    for para in doc.paragraphs:
        m = re.search(r'Measured Distance:\s*([\d.]+)\s*mm', para.text)
        if m:
            distances.append(float(m.group(1)))
    return distances

class LoggedFrontLRSensor:
    """
    Supplies left and right distance readings from logs.
    Front distance and obstacle_ahead are computed via simulated corridor
    geometry (same as SimulatedCorridorSensors).
    """
    def __init__(self, dt: float = 0.1):
        """
        dt: time interval between consecutive log entries (seconds).
            Default 0.1 s matches the 10 Hz control loop.
        """
        self.left_mm = _extract_distances(LOG_L_PATH)
        self.right_mm = _extract_distances(LOG_R_PATH)
        # Use the shorter length to avoid index errors
        self.n_samples = min(len(self.left_mm), len(self.right_mm))
        self.left_mm = self.left_mm[:self.n_samples]
        self.right_mm = self.right_mm[:self.n_samples]
        self.dt = dt
        self._start_time = time.time()
        # Simulated front sensor delegate (we only need its front/obstacle logic)
        self._front_sim = None  # lazy init to avoid needing corridor width etc at import

    def _init_front_sim(self):
        if self._front_sim is None:
            # Import here to avoid circular issues
            from sensor_sim import SimulatedCorridorSensors
            self._front_sim = SimulatedCorridorSensors(
                corridor_width_m=config.CORRIDOR_WIDTH_M,
                obstacles=config.SIM_OBSTACLES
            )
            # Note: SimulatedCorridorSensors uses its own noise; we will not use its left/right.

    def _get_next_lr(self) -> tuple[float, float]:
        """Return next left, right reading in metres, based on elapsed time."""
        elapsed = time.time() - self._start_time
        idx = int(elapsed / self.dt)
        if idx >= self.n_samples:
            # Loop the data
            idx = idx % self.n_samples
        left_m = self.left_mm[idx] / 1000.0
        right_m = self.right_mm[idx] / 1000.0
        return left_m, right_m

    def read(self, along_corridor_m: float, lateral_offset_m: float) -> Dict:
        """
        Returns a dict with left_m, right_m, front_m, obstacle_ahead.
        Left/right come from logs (looped over time).
        Front and obstacle_ahead come from simulated corridor geometry.
        """
        self._init_front_sim()
        left_m, right_m = self._get_next_lr()
        # Get simulated front reading (uses along_corridor_m and lateral_offset_m)
        front_reading = self._front_sim.read(along_corridor_m, lateral_offset_m)
        front_m = front_reading["front_m"]
        obstacle_ahead = front_reading["obstacle_ahead"]
        return {
            "left_m": round(left_m, 3),
            "right_m": round(right_m, 3),
            "front_m": round(front_m, 3),
            "obstacle_ahead": obstacle_ahead,
        }