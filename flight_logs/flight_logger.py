"""
flight_logger.py
------------------
Minimal CSV logger so you can plot corridor-run performance afterward
(centering error over time, speed profile, etc).
"""

import csv
import os


class FlightLogger:
    def __init__(self, filepath="corridor_run_log.csv"):
        self.filepath = filepath
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
        ["t", "along_m", "lateral_m", "left_m", "right_m", "front_m",
         "vx", "vy", "yaw_rad", "yaw_error_rad"]
        )

    def log(self, t, along, lateral, left, right, front, vx, vy,
         yaw=None, yaw_error=None):
        self._writer.writerow(
            [f"{t:.3f}", f"{along:.3f}", f"{lateral:.3f}",
            f"{left:.3f}", f"{right:.3f}", f"{front:.3f}",
            f"{vx:.3f}", f"{vy:.3f}",
            f"{yaw:.3f}" if yaw is not None else "",
            f"{yaw_error:.3f}" if yaw_error is not None else ""]
        )

    def close(self):
        self._file.close()
        print(f"[logger] Log saved to {os.path.abspath(self.filepath)}")