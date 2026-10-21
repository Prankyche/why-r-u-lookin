"""
corridor_navigator.py
-----------------------
The actual corridor-navigation control loop:
  - reads left/right/front distances every tick
  - computes a lateral centering correction (PID on left-right error)
  - computes a forward speed (slows/stops based on front obstacle distance)
  - sends body-frame velocity setpoints (or a corridor-end position target) to the flight controller
  - exits when it has traveled the corridor length, or aborts on timeout / unsafe centering error

Real-drone additions:
  - PILOT_TAKEOVER: if the FC leaves GUIDED (RC mode switch) the loop stops commanding and returns
    without landing, so the script never fights the pilot.
  - LINK_LOST:      no FC heartbeat for config.LINK_LOST_TIMEOUT_S.
  - SENSOR_LOST:    front lidar silent for config.FRONT_LIDAR_SILENT_ABORT_S.
  - Hard stop in position-target mode: speed is set by the FC there (WP_SPD), so the
    slow/stop logic below would otherwise have no effect. An obstacle at or inside
    OBSTACLE_STOP_DISTANCE_M now holds position until it has been clear for CLEAR_STREAK_TO_RELEASE ticks.

This module is sensor-source-agnostic: pass it anything with
a .read(along_corridor_m, lateral_offset_m) -> {"left_m","right_m","front_m"[, "obstacle_ahead", "front_valid"]}
method.
"""

import time
import math
from pymavlink import mavutil
import config
from control.pid_controller import PID


class CorridorResult:
    SUCCESS = "success"
    TIMEOUT = "timeout"
    UNSAFE_ABORT = "unsafe_abort"
    PILOT_TAKEOVER = "pilot_takeover"
    LINK_LOST = "link_lost"
    SENSOR_LOST = "sensor_lost"


class CorridorNavigator:
    def __init__(self, vehicle, sensors, logger=None):
        self.vehicle = vehicle
        self.sensors = sensors
        self.logger = logger

        self.lateral_pid = PID(
            kp=config.LATERAL_KP,
            ki=config.LATERAL_KI,
            kd=config.LATERAL_KD,
            output_limit=config.LATERAL_MAX_MPS,
        )

        # Internal estimate of how far we've traveled into the corridor.
        # In sim this is tracked from commanded velocity * dt (dead reckoning).
        # On real hardware it comes from the FC's local position (LOCAL_POSITION_NED).
        self._along_corridor_m = 0.0
        self._lateral_offset_m = 0.0
        self._avoid_direction = None  # locked +1 (right) / -1 (left) while sidestepping an obstacle
        self._clear_streak = 0        # consecutive ticks with front distance clear of obstacle
        self._last_target_t = 0.0     # last time a position target was sent (position-target mode)
        self._stop_hold = False       # hard-stop latch (position-target mode)
        self._release_streak = 0      # consecutive clear ticks while the hard stop is latched
        self._front_lost_since = None # when the front lidar last went invalid

    def run(self, real_time=True):
        """
        Blocking call: navigates the full corridor length. Returns CorridorResult.

        real_time=True  -> sleeps to match CONTROL_LOOP_HZ (use for SITL/real hardware,
                            where the vehicle's actual motion is happening in wall-clock time).
        real_time=False -> runs as fast as possible, using a virtual clock (use for the
                            offline/no-MAVLink logic test, where nothing physically moves).
        """
        self.lateral_pid.reset()
        period = 1.0 / config.CONTROL_LOOP_HZ
        start_time = time.time()
        virtual_time = 0.0

        # Optional vehicle features (the offline test's FakeVehicle may not have them)
        get_mode = getattr(self.vehicle, "get_mode", None)
        hb_age = getattr(self.vehicle, "heartbeat_age_s", None)

        print("[corridor] Starting corridor navigation...")

        # Track progress from REAL telemetry, not dead-reckoning from our own
        # commanded velocity. BendyRuler is primary: if the FC steers the real vehicle around an
        # obstacle, that has to be reflected here, or this loop would never know it was cleared.
        start_pos = self.vehicle.get_local_position()
        for _ in range(10):
            if start_pos is not None:
                break
            time.sleep(0.2)
            start_pos = self.vehicle.get_local_position()
        if start_pos is None:
            print("[corridor] WARNING: couldn't read starting position, falling back to dead-reckoning.")
        start_x, start_y = (start_pos[0], start_pos[1]) if start_pos else (0.0, 0.0)

        # Lock yaw to whatever heading the corridor is actually aligned to at the start of this run.
        # BendyRuler's own avoidance maneuvers can reorient the vehicle -- once that happens,
        # body-frame vy stops matching the corridor's real-world left/right.
        target_yaw = self.vehicle.get_yaw()
        if target_yaw is None:
            target_yaw = 0.0
            print("[corridor] WARNING: couldn't read starting yaw, assuming 0 (North).")

        while True:
            loop_start = time.time()
            elapsed_time = (time.time() - start_time) if real_time else virtual_time

            # ---- pilot / link supervision (real drone) ----
            if config.PILOT_TAKEOVER_CHECK and get_mode is not None:
                mode = get_mode()
                if mode is not None and mode != config.GUIDED_MODE_NAME:
                    print(f"[corridor] FC mode is {mode}, not {config.GUIDED_MODE_NAME} -- "
                          f"pilot took over, script stops commanding.")
                    return CorridorResult.PILOT_TAKEOVER
            if hb_age is not None:
                age = hb_age()
                if age is not None and age > config.LINK_LOST_TIMEOUT_S:
                    print(f"[corridor] No FC heartbeat for {age:.1f}s -- link lost, aborting.")
                    return CorridorResult.LINK_LOST

            if elapsed_time > config.CORRIDOR_TIMEOUT_S:
                print("[corridor] TIMEOUT -- aborting, holding position.")
                self.vehicle.hold_position()
                return CorridorResult.TIMEOUT

            # Pull real position every tick. NED local frame. Falls back to the last dead-reckoned
            # estimate on a dropped read so a single missed telemetry packet doesn't stall the loop.
            pos = self.vehicle.get_local_position()
            if pos is not None:
                if config.USE_POSITION_TARGET_NAV:
                    # project onto the corridor axes (along = target_yaw direction, lateral = its right)
                    dx, dy = pos[0] - start_x, pos[1] - start_y
                    self._along_corridor_m = dx * math.cos(target_yaw) + dy * math.sin(target_yaw)
                    self._lateral_offset_m = -dx * math.sin(target_yaw) + dy * math.cos(target_yaw)
                else:
                    self._along_corridor_m = pos[0] - start_x
                    self._lateral_offset_m = pos[1] - start_y

            reading = self.sensors.read(self._along_corridor_m, self._lateral_offset_m)
            left, right, front = reading["left_m"], reading["right_m"], reading["front_m"]
            obstacle_ahead = reading.get("obstacle_ahead", front < config.OBSTACLE_SLOW_DISTANCE_M)

            # ---- front lidar silent for too long -> don't fly blind ----
            if reading.get("front_valid", True):
                self._front_lost_since = None
            elif self._front_lost_since is None:
                self._front_lost_since = elapsed_time
            elif elapsed_time - self._front_lost_since > config.FRONT_LIDAR_SILENT_ABORT_S:
                print(f"[corridor] Front lidar silent for more than "
                      f"{config.FRONT_LIDAR_SILENT_ABORT_S:.1f}s -- aborting, holding position.")
                self.vehicle.hold_position()
                return CorridorResult.SENSOR_LOST

            # PRIMARY avoidance input for BendyRuler (SITL only): FRONT sensor only.
            # Deliberately NOT sending left/right -- BendyRuler dodges discrete obstacles in open
            # space; feeding it the corridor walls makes it think it is boxed in and fight our
            # lateral PID. Wall-centering stays entirely our own job.
            # On the real drone FEED_DISTANCE_SENSOR_TO_FC is False: the FC reads the real lidar.
            if config.FEED_DISTANCE_SENSOR_TO_FC:
                # The FC treats this as a beam straight ahead, so ONLY report real obstacles; send
                # max range (800 cm = nothing) otherwise, or BendyRuler sees a phantom wall ahead.
                dist_cm = front * 100 if obstacle_ahead else 800
                self.vehicle.send_fake_distance_sensor(
                    dist_cm, mavutil.mavlink.MAV_SENSOR_ROTATION_NONE, sensor_id=0)

            centering_error = left - right
            # Sign convention: vy > 0 means move RIGHT (body frame).
            # If left > right the drone is closer to the right wall, so it should move LEFT
            # (vy negative). error = left - right is positive then, PID returns kp*error, so negate.
            #
            # Safety check is based on actual wall clearance, not the left/right DIFFERENCE --
            # a large difference is expected while deliberately sidestepping an obstacle.
            min_clearance = min(left, right)
            if min_clearance < config.MIN_WALL_CLEARANCE_M:
                print(f"[corridor] UNSAFE wall clearance {min_clearance:.2f} m -- aborting.")
                self.vehicle.hold_position()
                return CorridorResult.UNSAFE_ABORT

            lateral_correction = self.lateral_pid.update(centering_error)
            vy = -lateral_correction  # negate per sign convention above

            vx = self._compute_forward_speed(front, obstacle_ahead)

            # Actively correct back toward the corridor's original heading (P-controller on yaw error).
            yaw_rt = 0.0
            current_yaw = None
            yaw_error = None
            if config.YAW_LOCKED:
                current_yaw = self.vehicle.get_yaw()
                if current_yaw is not None:
                    yaw_error = target_yaw - current_yaw
                    # normalize to [-pi, pi] so we always turn the short way
                    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                    yaw_rt = max(-config.YAW_LOCK_MAX_RATE_RAD_S,
                                 min(config.YAW_LOCK_MAX_RATE_RAD_S, config.YAW_LOCK_KP * yaw_error))

            # --- reactive obstacle sidestep (BACKUP steering, off by default) ---
            # Gated behind CUSTOM_SIDESTEP_STEERING_ENABLED so it can't fight BendyRuler.
            if config.CUSTOM_SIDESTEP_STEERING_ENABLED and obstacle_ahead and front < config.OBSTACLE_SLOW_DISTANCE_M:
                self._clear_streak = 0
                # Lock the avoidance direction the FIRST time an obstacle is detected, based on
                # which side has more clearance, so sensor noise can't cause flip-flopping.
                if self._avoid_direction is None:
                    self._avoid_direction = 1 if (right - left) > 0 else -1
                avoid_vy = config.LATERAL_MAX_MPS * self._avoid_direction

                # blend: closer the obstacle, the more we prioritize avoidance over centering
                closeness = max(0.0, min(1.0,
                    (config.OBSTACLE_SLOW_DISTANCE_M - front) / config.OBSTACLE_SLOW_DISTANCE_M))
                vy = (1 - closeness) * vy + closeness * avoid_vy
                # never allow the nudge to push us within MIN_WALL_CLEARANCE_M (+ buffer)
                buffer = config.MIN_WALL_CLEARANCE_M + 0.15
                if right - buffer < 0 and vy > 0:
                    vy = 0.0
                if left - buffer < 0 and vy < 0:
                    vy = 0.0
                # keep a small forward creep instead of a hard 0 while sidestepping
                if front > config.OBSTACLE_STOP_DISTANCE_M:
                    vx = max(vx, config.FORWARD_SPEED_MIN_MPS)
            else:
                # Release the direction lock only after several CONSECUTIVE clear readings
                # (avoids the "sawtooth" re-lock/re-trigger cycling).
                self._clear_streak += 1
                if self._clear_streak >= config.CLEAR_STREAK_TO_RELEASE:
                    self._avoid_direction = None
                    self._clear_streak = 0

            # --- hard-stop latch (used in position-target mode, where the FC sets the speed) ---
            hard_blocked = obstacle_ahead and front <= config.OBSTACLE_STOP_DISTANCE_M
            if hard_blocked:
                if not self._stop_hold:
                    print(f"[corridor] Obstacle at {front:.2f} m -- holding position until clear.")
                self._stop_hold = True
                self._release_streak = 0
            elif self._stop_hold:
                self._release_streak += 1
                if self._release_streak >= config.CLEAR_STREAK_TO_RELEASE:
                    print("[corridor] Obstacle cleared -- resuming.")
                    self._stop_hold = False
                    self._last_target_t = 0.0      # re-send the destination immediately

            if config.USE_POSITION_TARGET_NAV and pos is not None:
                if self._stop_hold:
                    # zero-velocity setpoint overrides the standing position target
                    self.vehicle.hold_position()
                elif time.time() - self._last_target_t >= config.POSITION_TARGET_PERIOD_S:
                    # Send the corridor END as a destination (re-aimed at the corridor centre from
                    # the side sensors) and let ArduPilot fly it -- BendyRuler can steer around obstacles.
                    shift = max(-1.0, min(1.0, -(left - right) / 2.0))   # +ve = move right
                    lat_center = self._lateral_offset_m + shift
                    ahead = config.CORRIDOR_LENGTH_M + config.POSITION_TARGET_EXIT_MARGIN_M
                    cy, sy = math.cos(target_yaw), math.sin(target_yaw)
                    dest_x = start_x + ahead * cy - lat_center * sy
                    dest_y = start_y + ahead * sy + lat_center * cy
                    self.vehicle.send_position_target_local(dest_x, dest_y, pos[2], yaw=target_yaw)
                    self._last_target_t = time.time()
            else:
                self.vehicle.send_velocity_body(vx=vx, vy=vy, vz=0.0, yaw_rate=yaw_rt)

            # Fallback only: if this tick's telemetry read failed, keep the loop moving using
            # dead-reckoning rather than freezing at a stale position.
            if pos is None:
                self._along_corridor_m += vx * period
                self._lateral_offset_m += vy * period

            if self.logger:
                self.logger.log(
                    t=elapsed_time,
                    along=self._along_corridor_m,
                    lateral=self._lateral_offset_m,
                    left=left, right=right, front=front,
                    vx=vx, vy=vy,
                    yaw=current_yaw, yaw_error=yaw_error,
                )

            if self._along_corridor_m >= config.CORRIDOR_LENGTH_M:
                print("[corridor] Reached corridor end.")
                self.vehicle.hold_position()
                return CorridorResult.SUCCESS

            # maintain loop rate (real hardware/SITL) or advance virtual clock (offline test)
            if real_time:
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, period - elapsed)
                time.sleep(sleep_time)
            else:
                virtual_time += period

    def _compute_forward_speed(self, front_distance_m, obstacle_ahead=False):
        # Only a REAL obstacle should bring us to a full stop. The corridor exit shrinking below
        # the stop threshold is not a reason to halt -- the drone would freeze short of the finish.
        if obstacle_ahead and front_distance_m <= config.OBSTACLE_STOP_DISTANCE_M:
            return 0.0
        if front_distance_m >= config.OBSTACLE_SLOW_DISTANCE_M:
            return config.FORWARD_SPEED_MAX_MPS
        # linear ramp between stop distance and slow distance
        span = config.OBSTACLE_SLOW_DISTANCE_M - config.OBSTACLE_STOP_DISTANCE_M
        frac = (front_distance_m - config.OBSTACLE_STOP_DISTANCE_M) / span
        speed = config.FORWARD_SPEED_MIN_MPS + frac * (
            config.FORWARD_SPEED_MAX_MPS - config.FORWARD_SPEED_MIN_MPS
        )
        return max(config.FORWARD_SPEED_MIN_MPS, min(config.FORWARD_SPEED_MAX_MPS, speed))