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
MAVLINK_CONNECTION_STRING = "udp:127.0.0.1:14550"   # MAVProxy on the Pi; MAVProxy owns the serial port
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

# ---- QR vision (vision/qr/aerothon_qr.py) ----
# Plain values only. adapters/qr_adapter.py turns these into aerothon_qr.QRConfig.
# Calibrate the Arducam B0332 with calibrate_camera.py and save calib.npz before flying.
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

# ---- Red zone (Pi camera, adapters/redzone_adapter.py) ----
RED_ZONE_TRIGGER_RADIUS = 0.20       # fraction of frame half-width: red centred in view = boost speed
RED_ZONE_MIN_AREA_PX = 5000          # ignore red blobs smaller than this (noise)
RED_HSV_LOW1 = (0, 120, 70)          # red wraps around hue 0 in HSV, so two ranges
RED_HSV_HIGH1 = (10, 255, 255)
RED_HSV_LOW2 = (170, 120, 70)
RED_HSV_HIGH2 = (180, 255, 255)

# ---- Field sweep (SEARCH_QR state, control/field_nav.py) ----
# The field is 4 GPS corners IN ORDER around it (same file format as l_sweep.py --record-corners):
#   FIELD_FENCE_FILE = "fence.txt"                   or   FIELD_CORNERS_LL = [(lat, lon), x4]
# With neither set, the sweep falls back to the FIELD_WIDTH_M x FIELD_HEIGHT_M rectangle below, placed at
# FIELD_START_NE_M from where the drone is when SEARCH_QR starts.
# The drone may be anywhere: it flies to the first waypoint (nearest corner) and starts from there.
FIELD_FENCE_FILE = None
FIELD_CORNERS_LL = None
FIELD_MAX_START_DIST_M = 150.0       # refuse (RTL) if the nearest field corner is further than this: bad coordinates
FIELD_WIDTH_M = 30.0                 # fallback rectangle: extent east of the start corner (30 x 40 per team figure)
FIELD_HEIGHT_M = 40.0                # fallback rectangle: extent north of the start corner
FIELD_START_NE_M = (0.0, 0.0)        # fallback rectangle: (north, east) from the drone's SEARCH_QR start to the field corner
SWEEP_CAM_ACROSS = "h"               # QR camera mounting: "h" = image WIDTH (long side) points across the lanes, "v" = image height.
                                     # VERIFY on the bench: wrong = lanes too far apart = missed QR. "v" is the safe (narrower) choice.
SWEEP_OVERLAP = 0.275                # side overlap between lanes (25-30 %). Must exceed the QR size (~1.2 m) or a QR can fall between lanes.
SWEEP_SPACING_M = None               # None = from the camera footprint and SWEEP_OVERLAP. A number overrides it.
SWEEP_MARGIN_M = 0.5                 # lane ends stay this far inside the fence
SWEEP_LANE_EDGE = "long"             # lanes run along the "long" or "short" field edge
SWEEP_ARRIVAL_RADIUS_M = 1.5         # within this distance = waypoint reached, send the next
SWEEP_SPEED_NORMAL_MS = 1.5
SWEEP_SPEED_BOOST_MS = 2.5           # used while red is centred in view (2.5-3.0 m/s)

# ---- Entry banner (ALIGN_BANNER, adapters/banner_adapter.py) ----
# Exact banner colour #00B140 = OpenCV HSV (71, 255, 177). The camera will not reproduce it
# exactly under field light, so sample the banner in the preview and adjust if needed.
BANNER_HSV = (71, 255, 177)
BANNER_HUE_TOL = 8                   # +/- on hue only
BANNER_S_MIN = 120
BANNER_V_MIN = 60
BANNER_MIN_AREA_PX = 1200            # at 640x480
BANNER_ALIGN_TOL = 0.10              # normalised image offset counted as centred
BANNER_ALIGNED_FRAMES = 10           # consecutive centred frames before BANNER_FOUND
BANNER_KP = 0.10                     # m/s per unit of normalised offset
BANNER_MAX_SPEED_MS = 0.08
BANNER_RIGHT_SIGN = 1.0              # verify on the bench (props off)
BANNER_FORWARD_SIGN = -1.0

# ---- Exit banner (blue, ALIGN_EXIT_BANNER) and return turn (TURN_RETURN) ----
# Exact exit banner colour #0000FF = OpenCV HSV (120, 255, 255).
EXIT_BANNER_HSV = (120, 255, 255)
EXIT_BANNER_MIN_AREA_PX = 1000       # at 640x480, banner seen from 10 m
RETURN_TURN_TOL_DEG = 5.0            # yaw within this of the reverse heading = turn done

# ---- Mission runner (main.py) ----
MAIN_LOOP_HZ = 10.0
ALT_TOLERANCE_M = 0.4                # |altitude - state target| within this = ALT_REACHED
ALT_CHANGE_TIMEOUT_S = 20            # blocking climb/descent before a corridor run
QR_CAMERA_INDEX = 0                  # /dev/video index of the Arducam OV9281
PI_CAMERA_INDEX = 1                  # Pi Camera V2 (banners + red zone). On a Pi 5 the CSI camera is read via picamera2 if installed
PI_CAMERA_WIDTH = 640
PI_CAMERA_HEIGHT = 480
# Payload motor (N20 through a motor driver). SET THESE before flying; None = not wired -> main.py refuses a real flight.
PAYLOAD_GPIO_FORWARD = None          # BCM pin that drives the release direction
PAYLOAD_GPIO_REVERSE = None          # BCM pin for the other direction (None if one direction only)
PAYLOAD_RUN_S = 2.0                  # PLACEHOLDER: time the motor runs to release. Measure on the bench.
EXIT_BANNER_NE_M = None              # (north, east) of the blue exit banner from the field origin. None = not known:
                                     # the drone then only centres on a banner that is already in view. MEASURE IT.
EXIT_BANNER_ARRIVE_M = 0.5           # fly to EXIT_BANNER_NE_M by position until this close, then centre on the banner by camera

# ---- Guided position-target handling ----
# With GUID_OPTIONS bit 6 (64) set, ArduPilot flies position targets at WP_SPD and ignores DO_CHANGE_SPEED.
# Then name the parameter here so set_speed() also writes it: "WP_SPD" (m/s, firmware 4.7+) or "WPNAV_SPEED" (cm/s, scale 100).
# None = DO_CHANGE_SPEED only. NOTE: a parameter write is saved on the FC; restore WP_SPD after the flight.
SPEED_PARAM_NAME = None
SPEED_PARAM_SCALE = 1.0
TARGET_RESEND_S = 3.0                # re-send a position target only if the drone has stopped short of it for this long

# ---- Delivery: fly over the QR before descending ----
DROP_XY_TOL_M = 1.0                  # DESCEND_DROP only reports ALT_REACHED once the drone is this close (N/E) to the QR
# Mapping from qr_adapter.target_offset_m() = (dx, dy) in image axes (x right, y down) to body axes. With the QR camera
# mounted looking straight down and image-up = nose: forward = -dy, right = +dx. VERIFY on the bench (props off): hold a
# QR ahead of the nose and check the drone would move toward it.
QR_OFFSET_FORWARD_SIGN = -1.0
QR_OFFSET_RIGHT_SIGN = 1.0

# ---- Failsafes the runner raises (main.py) ----
# Battery comes from the FC (SYS_STATUS). The FC only reports a percentage if BATT_CAPACITY is set; otherwise set a voltage.
BATTERY_LOW_PCT = 25.0               # LOW_BATT (-> RTL) below this percentage...
BATTERY_LOW_VOLTS = None             # ...or below this pack voltage (None = voltage check off). SET THIS if the FC has no BATT_CAPACITY.
BATTERY_DEBOUNCE_S = 3.0             # must stay low this long (ignores current-sag spikes)
GEOFENCE_MAX_ALT_M = 13.0            # GEOFENCE_BREACH (-> RTL) above this height (sweep altitude is 10 m)
GEOFENCE_RADIUS_M = None             # horizontal limit from the take-off point, m (None = off). Set it to the arena size.
GEOFENCE_DEBOUNCE_S = 1.0
# Failsafes are NOT raised during LAND (RTL would interrupt the landing) or RELEASE.

# ---- Landing and search passes ----
LAND_AT_TAKEOFF = True               # after the return corridor: climb, fly back over the take-off point, then land (rulebook)
HOME_TOL_M = 1.0                     # RETURN_HOME is done when within this (N/E) of the take-off point
SWEEP_PASSES = 2                     # 1 = a single sweep. 2 = a second, cross-hatched sweep (lanes along the other field edge) if no QR was found