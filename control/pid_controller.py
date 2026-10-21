"""
pid_controller.py
------------------
Small, dependency-free PID controller used for lateral centering.
"""

import time


class PID:
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit  # (min, max) or a single +/- float

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error):
        now = time.time()
        if self._prev_time is None:
            dt = 0.0
        else:
            dt = now - self._prev_time
        self._prev_time = now

        self._integral += error * dt
        derivative = 0.0 if dt <= 0 else (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limit is not None:
            limit = self.output_limit
            if isinstance(limit, tuple):
                output = max(limit[0], min(limit[1], output))
            else:
                output = max(-limit, min(limit, output))

        return output