"""
mavlink_connection.py
---------------------
Thin wrapper around pymavlink for corridor navigation.

Changes in this version (real-drone port):
  * connect() passes the serial baud rate and autoreconnect (config.MAVLINK_BAUD / MAVLINK_AUTORECONNECT)
  * get_mode() / heartbeat_age_s(): lets the navigator notice a pilot takeover or a dead link
  * DISTANCE_SENSOR is cached PER ORIENTATION, so a second rangefinder (e.g. a downward one) can't
    overwrite the forward lidar reading
  * HEARTBEATs from other components / ground stations are ignored for mode tracking
  * FakeVehicle gained the same methods for offline tests; old commented-out copies removed
"""

import time
from pymavlink import mavutil
import config


class Vehicle:
    def __init__(self, connection_string=None, sensor_bias_mm=None, baud=None):
        self.connection_string = connection_string or config.MAVLINK_CONNECTION_STRING
        self.baud = baud or getattr(config, "MAVLINK_BAUD", 115200)
        self.master = None

        # Sensor calibration bias (mm, added to raw reading to correct)
        self.sensor_bias_mm = sensor_bias_mm or {
            'front': getattr(config, 'FRONT_SENSOR_BIAS_MM', 17.4),
            'left': getattr(config, 'LEFT_SENSOR_BIAS_MM', 7.2),
            'right': getattr(config, 'RIGHT_SENSOR_BIAS_MM', 7.2),
        }

        self._t0 = time.time()          # for DISTANCE_SENSOR time_boot_ms
        self._idle_hooks = []           # callables run while we wait (arm / takeoff / land / mode change)

        # Newest message of each type, and when it arrived
        self._telemetry_cache = {}
        self._cache_timestamps = {}

    # ============= Idle hooks =============
    def add_idle_hook(self, fn):
        """Register fn() to be called ~10-20 times a second while this class waits for the FC
        (arming, takeoff, landing, mode change). Used to keep streaming the side ToF readings so a
        MAVLink proximity sensor on the FC doesn't go stale (PreArm: PRXn: No Data)."""
        self._idle_hooks.append(fn)

    def _idle(self):
        for fn in self._idle_hooks:
            try:
                fn()
            except Exception as e:
                print(f"[mavlink] idle hook failed ({e}); removing it")
                self._idle_hooks.remove(fn)

    # ============= Connection =============
    def connect(self, timeout_s=30):
        print(f"[mavlink] Connecting to {self.connection_string} (baud {self.baud}) ...")
        self.master = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud,
            autoreconnect=getattr(config, "MAVLINK_AUTORECONNECT", True),
        )
        hb = self.master.wait_heartbeat(timeout=timeout_s)
        if hb is None:
            raise RuntimeError(f"No MAVLink heartbeat on {self.connection_string} within {timeout_s}s "
                               f"(check wiring, SERIAL2_PROTOCOL/BAUD and the port name)")
        self._telemetry_cache["HEARTBEAT"] = hb
        self._cache_timestamps["HEARTBEAT"] = time.time()
        print(f"[mavlink] Heartbeat received (sysid={self.master.target_system}, "
              f"compid={self.master.target_component})")
        self._request_data_streams()
        return self.master

    def _request_data_streams(self, rate_hz=10):
        """Request telemetry streams from autopilot (the SRn_* params on the FC also cover this)."""
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate_hz,
            1,  # start streaming
        )
        time.sleep(0.5)

    # ============= Arming / Mode =============
    def set_mode(self, mode_name, wait=True):
        """Switch flight mode. wait=False sends the command and returns at once (use it for RTL/LAND failsafes: with a dead
        link the confirmation never arrives and waiting would freeze the caller for 10 s)."""
        mode_id = self.master.mode_mapping()[mode_name]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        return self._wait_for_mode(mode_name) if wait else True

    def _wait_for_mode(self, mode_name, timeout_s=10):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self._idle()
            self._pump()
            if self.get_mode() == mode_name:
                print(f"[mavlink] Mode set to {mode_name}")
                return True
            time.sleep(0.05)
        print(f"[mavlink] WARNING: mode {mode_name} not confirmed within timeout")
        return False

    def arm(self, wait_armed=True, timeout_s=45, retry_interval_s=3):
        """Sends arm command and retries until armed or timeout. FC messages (PreArm reasons) are
        printed as they arrive."""
        print("  [mavlink] Arming (will retry until armed or timeout)...")
        t0 = time.time()
        last_attempt = 0

        while time.time() - t0 < timeout_s:
            if time.time() - last_attempt >= retry_interval_s:
                self.master.mav.command_long_send(
                    self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 0, 0, 0, 0, 0, 0,
                )
                last_attempt = time.time()

            if not wait_armed:
                return True

            self._idle()
            self._pump()
            if self.is_armed():
                print("  [mavlink] Armed.")
                return True
            time.sleep(0.05)

        print(f"[mavlink] WARNING: arm not confirmed within {timeout_s}s timeout")
        return False

    def disarm(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        print("[mavlink] Disarm command sent.")

    def land(self, timeout_s=90):
        """Switch to LAND and wait until the FC reports disarmed (it auto-disarms on touchdown).
        A plain disarm() is refused while airborne, so always use this to finish a flight."""
        print("[mavlink] Landing (LAND mode)...")
        self.set_mode("LAND")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self._idle()
            self._pump()
            if self.is_armed() is False:
                print("[mavlink] Landed and disarmed.")
                return True
            time.sleep(0.05)
        print("[mavlink] WARNING: no disarm confirmation after landing timeout")
        return False

    # ============= Takeoff =============
    def takeoff(self, altitude_m):
        print(f"[mavlink] Taking off to {altitude_m} m ...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )
        self._wait_for_altitude(altitude_m)

    def _wait_for_altitude(self, target_altitude_m, tolerance_m=0.3, timeout_s=30):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            alt = self.get_relative_altitude(apply_sensor_bias=False)  # altitude doesn't need bias
            if alt is not None and abs(alt - target_altitude_m) <= tolerance_m:
                print(f"[mavlink] Reached altitude {alt:.2f} m")
                return True
            self._idle()
            time.sleep(0.1)
        print("[mavlink] WARNING: target altitude not confirmed within timeout")
        return False

    # ============= Telemetry (with message caching) =============
    def _pump(self):
        """Drain EVERY queued MAVLink message and keep the newest of each type.
        DISTANCE_SENSOR is additionally stored per orientation; HEARTBEAT only from the flight
        controller itself (not from ground stations or other components)."""
        now = time.time()
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "BAD_DATA":
                continue
            if t == "STATUSTEXT":
                print(f"[FC] {msg.text}")
            if t == "HEARTBEAT":
                if (msg.get_srcSystem() != self.master.target_system
                        or msg.type == mavutil.mavlink.MAV_TYPE_GCS):
                    continue
            self._telemetry_cache[t] = msg
            self._cache_timestamps[t] = now
            if t == "DISTANCE_SENSOR":
                key = f"DISTANCE_SENSOR_{msg.orientation}"
                self._telemetry_cache[key] = msg
                self._cache_timestamps[key] = now

    def _get_cached_message(self, msg_type, max_age_s=0.2):
        """Newest message of this type (or None). max_age_s kept for compatibility."""
        self._pump()
        return self._telemetry_cache.get(msg_type)

    def get_relative_altitude(self, apply_sensor_bias=False):
        """Returns altitude in meters, or None."""
        msg = self._get_cached_message("GLOBAL_POSITION_INT")
        if msg is None:
            return None
        return msg.relative_alt / 1000.0  # mm -> m

    def get_local_position(self):
        """Returns (x, y, z) in local NED frame (meters), or None."""
        msg = self._get_cached_message("LOCAL_POSITION_NED")
        if msg is None:
            return None
        return (msg.x, msg.y, msg.z)

    def get_global_latlon(self):
        """(lat, lon) in degrees from the newest GLOBAL_POSITION_INT, or None."""
        msg = self._get_cached_message("GLOBAL_POSITION_INT")
        if msg is None or not (msg.lat or msg.lon):
            return None
        return (msg.lat / 1e7, msg.lon / 1e7)

    def get_battery(self, max_age_s=3.0):
        """(remaining_percent or None, volts or None) from the newest SYS_STATUS, or (None, None) if stale/absent.
        remaining is None when the FC has no capacity configured (it reports -1)."""
        msg = self._get_cached_message("SYS_STATUS")
        t = self._cache_timestamps.get("SYS_STATUS")
        if msg is None or t is None or time.time() - t > max_age_s:
            return (None, None)
        pct = None if msg.battery_remaining is None or msg.battery_remaining < 0 else float(msg.battery_remaining)
        volts = None if msg.voltage_battery in (None, 0, 65535) else msg.voltage_battery / 1000.0
        return (pct, volts)

    def get_latlon(self):
        """(lat, lon) in degrees from the FC's fused position estimate, or None."""
        msg = self._get_cached_message("GLOBAL_POSITION_INT")
        if msg is None or not (msg.lat or msg.lon):
            return None
        return (msg.lat / 1e7, msg.lon / 1e7)

    def get_yaw(self):
        """Returns current yaw in radians (NED, 0 = North), or None."""
        msg = self._get_cached_message("ATTITUDE")
        if msg is None:
            return None
        return msg.yaw

    def get_mode(self):
        """Current flight mode name from the FC's newest heartbeat (e.g. 'GUIDED'), or None."""
        msg = self._get_cached_message("HEARTBEAT")
        if msg is None:
            return None
        return mavutil.mode_string_v10(msg)

    def is_armed(self):
        """True/False from the newest FC heartbeat, or None if none has been seen."""
        msg = self._get_cached_message("HEARTBEAT")
        if msg is None:
            return None
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def heartbeat_age_s(self):
        """Seconds since the last FC heartbeat, or None if none has been seen."""
        self._pump()
        t = self._cache_timestamps.get("HEARTBEAT")
        return None if t is None else time.time() - t

    # ============= Guided velocity / position control =============
    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        """Send velocity setpoint in BODY frame (forward, right, down), m/s."""
        type_mask = 0b0000_0111_1100_0111
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, yaw_rate,
        )

    def send_velocity_local(self, vx_north, vy_east, vz, yaw_rate=0.0):
        """Velocity setpoint in a FIXED local frame (North, East, Down), m/s."""
        type_mask = 0b0000_0111_1100_0111
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
            0, 0, 0, vx_north, vy_east, vz, 0, 0, 0, 0, yaw_rate,
        )

    def send_position_target_local(self, x, y, z, yaw=None):
        """Position setpoint in LOCAL_NED (m, z negative = up), optional yaw (rad).
        Unlike velocity setpoints, ArduPilot's object avoidance (BendyRuler, OA_TYPE) acts
        on position/waypoint targets in Guided mode (needs GUID_OPTIONS=64), so this is what lets
        it steer around obstacles. Speed to the target is set by WP_SPD / WPNAV_SPEED."""
        type_mask = 0x09F8 if yaw is not None else 0x0DF8   # position (+yaw) only
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
            x, y, z, 0, 0, 0, 0, 0, 0, yaw if yaw is not None else 0.0, 0.0,
        )

    def hold_position(self):
        self.send_velocity_body(0, 0, 0, 0)

    def set_speed(self, speed_mps):
        """Guided ground speed for position targets (MAV_CMD_DO_CHANGE_SPEED, type 1 = ground)."""
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, float(speed_mps), -1, 0, 0, 0, 0,
        )
        name = getattr(config, "SPEED_PARAM_NAME", None)
        if name:        # GUID_OPTIONS bit 6: the FC ignores DO_CHANGE_SPEED and uses WP_SPD / WPNAV_SPEED
            self.master.mav.param_set_send(
                self.master.target_system, self.master.target_component, name.encode(),
                float(speed_mps) * getattr(config, "SPEED_PARAM_SCALE", 1.0),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    # ============= Fake distance sensor injection (SITL only) =============
    def send_fake_distance_sensor(self, distance_cm, orientation, sensor_id):
        """
        Inject a fake distance sensor reading into the autopilot.
        orientation: MAV_SENSOR_ROTATION_* (e.g. NONE=front, YAW_90=right, YAW_270=left)
        Never used on the real drone (the FC reads the real lidar itself).
        """
        self.master.mav.distance_sensor_send(
            0,                      # time_boot_ms
            10,                     # min_distance (cm)
            800,                    # max_distance (cm)
            int(distance_cm),
            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
            sensor_id,
            orientation,
            0,                      # covariance
        )

    def send_distance_sensor(self, distance_cm, orientation, sensor_id, min_cm=10, max_cm=400):
        """Send one REAL measurement to the FC as DISTANCE_SENSOR (distance clamped to min..max cm).
        Used to stream the Pi's left/right VL53L1X to ArduPilot / Mission Planner."""
        cur = int(min(max(distance_cm, min_cm), max_cm))
        self.master.mav.distance_sensor_send(
            int((time.time() - self._t0) * 1000) & 0xFFFFFFFF,   # time_boot_ms
            int(min_cm), int(max_cm), cur,
            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
            sensor_id, orientation, 0)                            # covariance unknown

    # ============= Real DISTANCE_SENSOR (lidar wired to the FC) =============
    def request_message_interval(self, msg_id, hz):
        """Ask the FC to stream a MAVLink message at `hz` (e.g. DISTANCE_SENSOR = 132)."""
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def get_distance_sensor_mm(self, orientation=0, max_age_s=0.5):
        """Newest DISTANCE_SENSOR with this orientation in mm, or None if never seen / stale."""
        self._pump()
        key = f"DISTANCE_SENSOR_{orientation}"
        msg = self._telemetry_cache.get(key)
        if msg is None:
            return None
        if time.time() - self._cache_timestamps.get(key, 0) > max_age_s:
            return None
        return msg.current_distance * 10   # cm -> mm


class FakeVehicle:
    """
    Mimics Vehicle for offline testing (no MAVLink).
    Integrates velocity commands into a position and orientation.
    """
    def __init__(self):
        self.x, self.y, self.z = 0.0, 0.0, 0.0  # local NED position
        self.yaw = 0.0                          # heading (rad)
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0
        self.last_update_time = time.time()
        self.mode = "GUIDED"

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        """Store velocity command; integration happens in get_local_position()."""
        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz
        self.last_yaw_rate = yaw_rate

    def hold_position(self):
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0
        self.last_yaw_rate = 0.0

    def get_local_position(self):
        """Integrate velocities into position (body frame)."""
        import math
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        if dt > 0.5:  # Protect against large time jumps
            dt = 0.05

        # Integrate yaw first (affects body frame)
        self.yaw += self.last_yaw_rate * dt

        # Transform body-frame velocity to NED (rotate by yaw)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        vx_ned = self.last_vx * cos_yaw - self.last_vy * sin_yaw
        vy_ned = self.last_vx * sin_yaw + self.last_vy * cos_yaw

        # Integrate into position
        self.x += vx_ned * dt
        self.y += vy_ned * dt
        self.z += self.last_vz * dt

        return (self.x, self.y, self.z)

    def get_yaw(self):
        """Return current heading."""
        return self.yaw

    def get_mode(self):
        return self.mode

    def heartbeat_age_s(self):
        return 0.0

    def is_armed(self):
        return True

    def add_idle_hook(self, fn):
        pass

    def send_fake_distance_sensor(self, distance_cm, orientation, sensor_id):
        """No-op in offline mode."""
        pass

    def request_message_interval(self, msg_id, hz):
        pass

    def send_distance_sensor(self, distance_cm, orientation, sensor_id, min_cm=10, max_cm=400):
        pass

    def get_distance_sensor_mm(self, orientation=0, max_age_s=0.5):
        return None

    def set_mode(self, mode_name, wait=True):
        self.mode = mode_name
        return True

    def arm(self):
        """No-op; always returns True."""
        return True

    def disarm(self):
        """No-op."""
        pass

    def takeoff(self, altitude_m):
        """No-op; just move to altitude."""
        self.z = -altitude_m  # NED: negative is up

    def get_relative_altitude(self, apply_sensor_bias=False):
        """Return absolute value of Z (NED up is negative)."""
        return abs(self.z)

    def land(self):
        self.z = 0.0
        return True

    def send_position_target_local(self, x, y, z, yaw=None):
        """Offline stub: ArduPilot path planning is not emulated."""
        pass

    def set_speed(self, speed_mps):
        """Offline stub."""
        self.speed = speed_mps

    def get_global_latlon(self):
        """Offline stub: no GPS."""
        return None

    def get_latlon(self):
        """Offline stub: no GPS."""
        return None

    def get_battery(self, max_age_s=3.0):
        """Offline stub: no battery data."""
        return (None, None)