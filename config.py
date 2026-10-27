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

# ---- Sensor source ----                                                        <- the drone
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
    FEED_DISTANCE_SENSOR_TO_FC = False

# ---- Position-target navigation (lets ArduPilot's BendyRuler steer around obstacles) ----
USE_POSITION_TARGET_NAV = True    # True = send corridor-end destination instead of velocities
POSITION_TARGET_PERIOD_S = 0.5      # how often the destination is re-sent
POSITION_TARGET_EXIT_MARGIN_M = 1.0

# ---- Altitude control (for descent phase before corridor entry) ----
DESCENT_RATE_MPS = 0.5              # how fast to descend from scan altitude
QR_CALIB_PATH = "calib.npz"          # if this file exists, its focal length and centre are used
QR_CAMERA_WIDTH = 1280
QR_CAMERA_HEIGHT = 800
QR_CAMERA_FPS = 30
QR_EXPOSURE_ABS = None               # manual exposure in V4L2 units (100 us); None = auto
QR_HFOV_DEG = 70.0                   # datasheet value; ignored when calib.npz is present
QR_LATENCY_S = 0.05                  # exposure + USB + decode delay, measure it
QR_SIZE_M = 1.0                      # black QR modules only
QR_BORDER_M = 0.0762                 # 3 inch white border on each side
QR_FULLFRAME_EVERY = 4               # whole-frame decode every N frames (safety net)
QR_START_WINDOW = 5                  # start code: N frames ...
QR_START_NEEDED = 3                  # ... of which this many must agree
QR_CONFIRM_FRAMES = 3                # target: seen this many times ...
QR_CONFIRM_WINDOW_S = 1.0            # ... within this many seconds
QR_MATCH_THRESHOLD = 0.80            # undecoded box vs start code
QR_CONFLICT_THRESHOLD = 0.85         # decoded, different text, but pattern matches
QR_MATCH_MARGIN = 0.15               # best candidate must beat the runner-up by this