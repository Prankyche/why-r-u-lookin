"""
sim/fake_fc.py
--------------
A small ArduCopter-like MAVLink endpoint over UDP, so the REAL driver (drivers/mavlink_connection.Vehicle) and main.py can
be exercised end to end without hardware or SITL. It is NOT ArduPilot: it models only what the mission uses.

  heartbeat / mode (COMMAND_LONG DO_SET_MODE and SET_MODE) / arm / takeoff / position targets (yaw honoured only when the
  type_mask says so) / body- and local-frame velocity targets (3 s timeout like ArduPilot) / DO_CHANGE_SPEED (reset to
  WP_SPD on every new position target, like ArduPilot) / PARAM_SET / LAND / RTL / battery (SYS_STATUS) / RC channels.

Use:  fc = FakeFC(14590); fc.start(); Vehicle("udpin:127.0.0.1:14590").connect() ...
"""
import math
import threading
import time

from pymavlink import mavutil

mav = mavutil.mavlink
LAT0, LON0, EARTH_R = 12.9716510, 80.0442180, 6378137.0
MODES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL", 9: "LAND", 17: "BRAKE"}
YAW_IGNORE, GUID_TIMEOUT_S = 0x0400, 3.0


class FakeFC(threading.Thread):
    def __init__(self, port, yaw0=0.0, wp_spd=1.0, honour_change_speed=True, reset_speed_on_target=True,
                 accel=2.0, time_scale=1.0, start_mode=5):
        super().__init__(daemon=True)
        self.m = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1, source_component=1)
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = 0.0
        self.yaw, self.tgt_yaw = yaw0, None
        self.mode, self.armed = start_mode, False
        self.params = {"GUID_OPTIONS": 64.0, "WP_SPD": float(wp_spd)}
        self.speed_limit, self.honour, self.reset_on_target = float(wp_spd), honour_change_speed, reset_speed_on_target
        self.accel, self.scale = accel, time_scale
        self.tgt = None                                   # (n, e, d)
        self.vel_cmd = None                               # (frame, vx, vy, vz, t)
        self.takeoff_d = None
        self.rc = {i: 1500 for i in range(1, 19)}; self.rc[7] = 1000
        self.battery_pct, self.battery_v = 90, 16.0
        self.silent = False
        self.n_targets, self.n_vel, self.targets, self.speed_cmds = 0, 0, [], []
        self.mode_log, self.path = [], []
        self.t0 = time.time()
        self._stop = threading.Event()

    # ---- helpers
    def stop(self): self._stop.set()

    def ll(self):
        return (LAT0 + math.degrees(self.x / EARTH_R),
                LON0 + math.degrees(self.y / (EARTH_R * math.cos(math.radians(LAT0)))))

    def _set_mode(self, mode_id):
        self.mode = int(mode_id)
        self.mode_log.append((time.time() - self.t0, MODES.get(self.mode, str(mode_id))))

    def _ack(self, cmd, res=0):
        if not self.silent:
            self.m.mav.command_ack_send(cmd, res)

    # ---- inbound
    def _handle(self, msg):
        t = msg.get_type()
        if t == "COMMAND_LONG":
            c = msg.command
            if c == mav.MAV_CMD_DO_SET_MODE:
                self._set_mode(msg.param2)
            elif c == mav.MAV_CMD_COMPONENT_ARM_DISARM:
                self.armed = bool(msg.param1)
            elif c == mav.MAV_CMD_NAV_TAKEOFF and self.armed and self.mode == 4:
                self.takeoff_d = -msg.param7
            elif c == mav.MAV_CMD_DO_CHANGE_SPEED:
                self.speed_cmds.append((time.time() - self.t0, msg.param2))
                if self.honour:
                    self.speed_limit = msg.param2
            self._ack(c)
        elif t == "SET_MODE":
            self._set_mode(msg.custom_mode)
        elif t == "PARAM_SET":
            name = (msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()).rstrip("\x00")
            self.params[name] = float(msg.param_value)
            if name == "WP_SPD":
                self.speed_limit = float(msg.param_value)
        elif t == "SET_POSITION_TARGET_LOCAL_NED" and self.mode == 4:
            if msg.type_mask & 0b111 == 0b111 and not (msg.type_mask & 0b111000):       # velocity only
                self.vel_cmd, self.tgt, self.n_vel = (msg.coordinate_frame, msg.vx, msg.vy, msg.vz, time.time()), None, self.n_vel + 1
            else:                                                                          # position
                self.tgt, self.vel_cmd = (msg.x, msg.y, msg.z), None
                self.n_targets += 1
                use_yaw = not (msg.type_mask & YAW_IGNORE)
                self.tgt_yaw = msg.yaw if use_yaw else self.tgt_yaw
                self.targets.append((time.time() - self.t0, msg.x, msg.y, msg.z, msg.yaw if use_yaw else None))
                if self.reset_on_target:
                    self.speed_limit = self.params["WP_SPD"]
        elif t == "PARAM_REQUEST_READ":
            name = (msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()).rstrip("\x00")
            if name in self.params and not self.silent:
                self.m.mav.param_value_send(name.encode(), self.params[name], 9, len(self.params), 0)

    # ---- physics
    def _move_to(self, tx, ty, dt, limit):
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        want = min(limit, math.sqrt(2 * self.accel * d)) if d > 1e-6 else 0.0
        wvx, wvy = (dx / d * want, dy / d * want) if d > 1e-6 else (0.0, 0.0)
        dvx, dvy = wvx - self.vx, wvy - self.vy
        dv, cap = math.hypot(dvx, dvy), self.accel * dt
        if dv > cap:
            dvx, dvy = dvx / dv * cap, dvy / dv * cap
        self.vx, self.vy = self.vx + dvx, self.vy + dvy
        if math.hypot(self.vx, self.vy) * dt >= d and d < 0.05 + math.hypot(self.vx, self.vy) * dt:
            self.x, self.y, self.vx, self.vy = tx, ty, 0.0, 0.0
        else:
            self.x += self.vx * dt; self.y += self.vy * dt

    def _physics(self, dt):
        if not self.armed:
            return
        if self.mode == 9:                                          # LAND
            self.vx = self.vy = 0.0
            self.z = min(0.0, self.z + 1.0 * dt)
            if self.z >= 0.0:
                self.armed = False
            return
        if self.mode == 6:                                          # RTL: home, then down
            self._move_to(0.0, 0.0, dt, 3.0)
            if math.hypot(self.x, self.y) < 0.5:
                self.z = min(0.0, self.z + 1.0 * dt)
                if self.z >= 0.0:
                    self.armed = False
            return
        if self.mode != 4:                                          # LOITER / BRAKE / ...: hold
            self.vx = self.vy = 0.0
            return
        if self.takeoff_d is not None:
            self.z = max(self.takeoff_d, self.z - 2.5 * dt)
            if self.z <= self.takeoff_d + 1e-6:
                self.takeoff_d, self.tgt = None, (self.x, self.y, self.z)
            return
        if self.vel_cmd and time.time() - self.vel_cmd[4] <= GUID_TIMEOUT_S * 1.0:
            frame, vx, vy, vz, _ = self.vel_cmd
            if frame == mav.MAV_FRAME_BODY_OFFSET_NED:
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                vx, vy = vx * c - vy * s, vx * s + vy * c
            self.vx, self.vy = vx, vy
            self.x += vx * dt; self.y += vy * dt; self.z += vz * dt
            return
        self.vel_cmd = None
        if self.tgt is None:
            self.vx = self.vy = 0.0
            return
        tx, ty, tz = self.tgt
        self._move_to(tx, ty, dt, self.speed_limit)
        self.z += max(-1.5 * dt, min(1.5 * dt, tz - self.z))
        if self.tgt_yaw is not None:
            err = (self.tgt_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += max(-1.0 * dt, min(1.0 * dt, err))

    # ---- outbound
    def _stream(self, k):
        ms = int((time.time() - self.t0) * 1000)
        lat, lon = self.ll()
        self.m.mav.global_position_int_send(ms, int(lat * 1e7), int(lon * 1e7), int((10 - self.z) * 1000),
                                            int(-self.z * 1000), int(self.vx * 100), int(self.vy * 100), 0, 0)
        self.m.mav.local_position_ned_send(ms, self.x, self.y, self.z, self.vx, self.vy, 0)
        self.m.mav.attitude_send(ms, 0, 0, (self.yaw + math.pi) % (2 * math.pi) - math.pi, 0, 0, 0)     # ArduPilot reports +-pi
        self.m.mav.sys_status_send(0, 0, 0, 0, int(self.battery_v * 1000), 0, int(self.battery_pct), 0, 0, 0, 0, 0, 0)
        self.m.mav.gps_raw_int_send(ms * 1000, 3, int(lat * 1e7), int(lon * 1e7), int((10 - self.z) * 1000), 100, 100, 0, 0, 12)
        if k % 5 == 0:
            self.m.mav.rc_channels_send(ms, 18, *[self.rc[i] for i in range(1, 19)], 255)

    def run(self):
        dt, k, last_hb, nxt = 0.02, 0, 0.0, time.time()
        while not self._stop.is_set():
            now = time.time()
            if not self.silent and now - last_hb >= 0.5:
                base = mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (mav.MAV_MODE_FLAG_SAFETY_ARMED if self.armed else 0)
                self.m.mav.heartbeat_send(mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_ARDUPILOTMEGA, base, self.mode, 4)
                last_hb = now
            while True:
                msg = self.m.recv_match(blocking=False)
                if msg is None:
                    break
                self._handle(msg)
            self._physics(dt * self.scale)
            if k % 5 == 0:
                if not self.silent:
                    self._stream(k // 5)
                self.path.append((now - self.t0, self.x, self.y, -self.z, self.yaw, math.hypot(self.vx, self.vy)))
            k += 1
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))