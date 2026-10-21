"""
config.py
---------
All tunable parameters for corridor navigation.

This version is set up for the REAL DRONE:
  * Raspberry Pi 5 <-> ArduPilot FC over a serial link (MAVLink2)
  * 2x VL53L1X (left/right) on the Pi, I2C bus 1, Adafruit driver, XSHUT on GPIO20/GPIO21
  * front lidar (TFmini) wired to the FC; the FC reports it back as DISTANCE_SENSOR

For SITL / Mission Planner testing, change the three lines marked  <-- SITL.
"""

import os

# ---- MAVLink connection ----
# Real drone, Pi 5 GPIO UART (TELEM2 on the FC):  "/dev/ttyAMA0"   (or "/dev/serial0")
# Real drone, USB cable FC -> Pi:                 "/dev/ttyACM0"
# Real drone via MAVProxy on the Pi (so Mission Planner can share the link):  "udp:127.0.0.1:14550"
#   (only ONE program may own the serial port -- run main.py OR a separate script on that UDP port, not both)
# SITL / Mission Planner:                         "tcp:127.0.0.1:5762"                    <-- SITL
MAVLINK_CONNECTION_STRING = "/dev/ttyAMA0"
MAVLINK_BAUD = 921600               # must match SERIAL2_BAUD=921 on the FC (ignored for tcp / USB)
MAVLINK_AUTORECONNECT = True        # survive a brief serial glitch instead of crashing the script

# ---- Corridor geometry ----
CORRIDOR_WIDTH_M = 3.5
CORRIDOR_LENGTH_M = 10.0
CORRIDOR_ALTITUDE_M = 3.0
SCAN_ALTITUDE_M = 5.0
DELIVERY_ZONE_ALTITUDE_M = 10.0

# ---- Sensor mounting ----
# Distance from each side sensor to the drone's CENTRE. MEASURE THESE on the real airframe.
SENSOR_OFFSET_LEFT_M = 0.15
SENSOR_OFFSET_RIGHT_M = 0.15

# ---- Sensor calibration corrections (from real hardware test, Sept 2026) ----
# Both sensors calibrated at ~300 mm ground truth, Sept 2026.
# These are POSITIVE offsets (add to raw reading to correct toward ground truth).
FRONT_SENSOR_BIAS_MM = 12.9        # TFmini, 1 cm resolution so this is rough
FRONT_SENSOR_NOISE_STD_MM = 4.56
LEFT_SENSOR_BIAS_MM = 7.2          # VL53L1X left
LEFT_SENSOR_NOISE_STD_MM = 1.97
RIGHT_SENSOR_BIAS_MM = 17.3        # VL53L1X right (reads lower than left)
RIGHT_SENSOR_NOISE_STD_MM = 2.34

# ---- Front LiDAR (TFmini) ----
FRONT_LIDAR_MAX_RANGE_M = 8.0       # keep equal to RNGFND1_MAX_CM / 100 on the FC
OBSTACLE_STOP_DISTANCE_M = 1.0      # stop/hold if obstacle closer than this
OBSTACLE_SLOW_DISTANCE_M = 2.5      # start slowing down within this range
CLEAR_STREAK_TO_RELEASE = 8         # debounce for clearing obstacle / releasing the hard stop
FRONT_LIDAR_SILENT_ABORT_S = 1.0    # real lidar: no valid reading for this long -> hold + abort

# ---- Avoidance architecture ----
# PRIMARY: feed live distance sensor data to BendyRuler on the FC
FEED_DISTANCE_SENSOR_TO_FC = True

# BACKUP: companion-computer-side sidestep steering (off by default)
CUSTOM_SIDESTEP_STEERING_ENABLED = False

# ---- Control loop ----
CONTROL_LOOP_HZ = 10.0              # 10 Hz update rate
FORWARD_SPEED_MAX_MPS = 0.5         # cruise speed inside corridor
FORWARD_SPEED_MIN_MPS = 0.1         # crawl speed when near obstacle
LATERAL_KP = 0.8     # Up from 0.6
LATERAL_KI = 0.03    # Up from 0.02
LATERAL_KD = 0.2     # Up from 0.15
LATERAL_MAX_MPS = 0.3               # clamp lateral correction speed
YAW_LOCKED = True                   # keep yaw fixed
YAW_LOCK_KP = 2.0                   # P-gain: yaw error (rad) -> yaw_rate
YAW_LOCK_MAX_RATE_RAD_S = 0.5       # clamp on corrective yaw rate

# ---- Safety ----
MIN_WALL_CLEARANCE_M = 0.15         # abort if closer to either wall
CORRIDOR_TIMEOUT_S = 90             # abort if mission takes longer
GUIDED_MODE_NAME = "GUIDED"         # the mode this script flies in
PILOT_TAKEOVER_CHECK = True         # stop commanding (and do NOT land) if the FC leaves GUIDED
LINK_LOST_TIMEOUT_S = 3.0           # no FC heartbeat for this long -> abort the corridor run

# ---- Simulated sensors (for offline testing without Gazebo) ----
# Only used by sensor_sim.py in real_time=False mode
SIM_OBSTACLES = []   # [] = no simulated front obstacles (with SENSOR_SOURCE = "obslog" + a lidar log, the front comes from the log)
SIM_WALL_NOISE_STD_M = 0.03   # Use real sensor noise levels instead:
# SIM_WALL_NOISE_STD_M = (FRONT_SENSOR_NOISE_STD_MM / 1000.0)  # ~0.002 m

# ---- Sensor source ----
#   "sim"    : everything simulated (sensor_sim.py)
#   "logged" : left/right from the L_/R_ calibration .docx logs, front simulated
#   "replay" : simulated corridor + the real noise from those calibration logs
#   "obslog" : left/right from SIDE_OBSTACLE_LOG_PATH, front from FRONT_LIDAR_LOG_PATH if that file exists   <- Mission Planner test
#   "real"   : VL53L1X on the Pi + lidar on the FC                                                         <- the drone
SENSOR_SOURCE = "real"                              # <-- SITL: "obslog"
SIDE_OBSTACLE_LOG_PATH = "side_obstacle_log.txt"   # 'Left Obstacle: .. cm | Right Obstacle: .. cm' lines
SIDE_OBSTACLE_LOG_LOOP = True                      # loop the logs when they end (for continuous testing)
# Lidar log: used only if the file exists, so a missing file can't crash the run after takeoff.
FRONT_LIDAR_LOG_PATH = "front_lidar_log.txt" if os.path.exists("front_lidar_log.txt") else None
FRONT_LIDAR_LOG_UNIT = "cm"                        # TFmini prints centimetres
FRONT_LIDAR_LOG_LOOP = False                       # False: the lidar log plays once, then the front is clear

# ---- Pi hardware (SENSOR_SOURCE = "real") ----
TOF_I2C_BUS = 1
TOF_LEFT_XSHUT_GPIO = 20            # BCM pin wired to left VL53L1X XSHUT
TOF_RIGHT_XSHUT_GPIO = 21           # BCM pin wired to right VL53L1X XSHUT
TOF_LEFT_ADDR = 0x2A                # left is moved to 0x2A at start-up; right stays at the default 0x29
TOF_RIGHT_ADDR = 0x29               # (start-up scans the bus, so addresses left over from a previous run are fine)
TOF_LIBRARY = "adafruit"            # "adafruit" (adafruit_vl53l1x) | "vl53l1x" (pip VL53L1X package)
TOF_DISTANCE_MODE = 2               # adafruit: 1 short / 2 long;  vl53l1x pkg: 1 short / 2 medium / 3 long
TOF_TIMING_BUDGET_MS = 50           # adafruit driver only
TOF_NONE_MEANS_CLEAR = False        # False = a side sensor with no reading is a FAULT (reads 0.0 -> abort/hold)
TOF_CLEAR_RANGE_M = 4.0             # only used if TOF_NONE_MEANS_CLEAR is True
FRONT_LIDAR_ORIENTATION = 0         # MAV_SENSOR_ROTATION_NONE = forward (matches RNGFND1_ORIENT)

# ---- Stream the Pi's left/right ToF to the FC as DISTANCE_SENSOR (shows in Mission Planner) ----
# Same packets your standalone script sent. On the FC they arrive as MAVLink rangefinders
# (RNGFND2 = left, RNGFND3 = right -- see corridor_params_real.txt) and join the proximity picture.
STREAM_SIDE_TOF_TO_FC = True
STREAM_SIDE_TOF_HZ = 12
TOF_LEFT_ORIENTATION = 6            # MAV_SENSOR_ROTATION_YAW_270 (left)
TOF_RIGHT_ORIENTATION = 2           # MAV_SENSOR_ROTATION_YAW_90  (right)
TOF_LEFT_SENSOR_ID = 0
TOF_RIGHT_SENSOR_ID = 1
TOF_STREAM_MIN_CM = 10              # clamp range sent to the FC (matches RNGFND2/3_MIN_CM / MAX_CM)
TOF_STREAM_MAX_CM = 400

# ---- Real-drone start-up safety ----
PREFLIGHT_SENSOR_CHECK = (SENSOR_SOURCE == "real")   # all three sensors must read OK before arming
CONFIRM_BEFORE_ARM = (SENSOR_SOURCE == "real")       # wait for ENTER before arming (props-on safety)

if SENSOR_SOURCE == "real":
    # On the real drone the FC reads the lidar itself; injecting fake DISTANCE_SENSOR would conflict with it.
    # (Every other source -- sim, replay, obslog -- needs the injection so BendyRuler gets obstacle data.)
    FEED_DISTANCE_SENSOR_TO_FC = False

# ---- Position-target navigation (lets ArduPilot's BendyRuler steer around obstacles) ----
USE_POSITION_TARGET_NAV = True    # True = send corridor-end destination instead of velocities
POSITION_TARGET_PERIOD_S = 0.5      # how often the destination is re-sent
POSITION_TARGET_EXIT_MARGIN_M = 1.0 # aim this far past the corridor end so the drone doesn't stop early
# In this mode the script ALSO enforces a hard stop: obstacle <= OBSTACLE_STOP_DISTANCE_M -> hold until clear.
# FC params: see corridor_drone.param
#   OA_TYPE=1, OA_BR_TYPE=1, OA_BR_LOOKAHEAD=4, OA_MARGIN_MAX=0.6, OA_DB_SIZE=100, OA_DB_EXPIRE=3,
#   GUID_OPTIONS=64, WP_SPD=0.5 (m/s; older firmware: WPNAV_SPEED=50 cm/s), AVOID_ENABLE=0,
#   PRX1_TYPE=2 for Mission Planner SITL / 4 (RangeFinder) for the real lidar on the FC

# ---- Altitude control (for descent phase before corridor entry) ----
DESCENT_RATE_MPS = 0.5              # how fast to descend from scan altitude