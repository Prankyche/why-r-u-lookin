"""
sensor_calibration.py
---------------------
Sensor bias correction and validation for Aerothon 2026.

Calibration data (from real hardware test at ~300 mm):
Numbers below are the READINGS' error (negative = sensor reads low). config.py stores the
opposite sign (a positive bias that is ADDED to the raw reading).
- Front (TFmini):  measured 287.1 mm, error -12.9 mm (1 cm resolution, so rough)
- Left (VL53L1X):  measured 292.8 mm, error  -7.2 ± 1.97 mm
- Right (VL53L1X): measured 282.7 mm, error -17.3 ± 2.34 mm
"""

import config


def correct_distance(raw_mm, sensor_type='front', use_bias=True):
    """
    Apply calibration bias to raw sensor reading.
    
    Args:
        raw_mm: Raw distance in mm from sensor
        sensor_type: 'front', 'left', or 'right'
        use_bias: If False, return raw reading (for debugging/logging)
    
    Returns:
        Corrected distance in meters (float)
    
    Example:
        >>> raw_front_mm = 283
        >>> corrected_m = correct_distance(raw_front_mm, 'front')
        >>> print(f"{corrected_m:.3f} m")  # 0.300 m (approx ground truth)
    """
    bias_map = {
        'front': getattr(config, 'FRONT_SENSOR_BIAS_MM', 12.9),
        'left': getattr(config, 'LEFT_SENSOR_BIAS_MM', 7.2),
        'right': getattr(config, 'RIGHT_SENSOR_BIAS_MM', 17.3),
    }
    
    if sensor_type not in bias_map:
        raise ValueError(f"Unknown sensor_type: {sensor_type}")
    
    if use_bias:
        corrected_mm = raw_mm + bias_map[sensor_type]
    else:
        corrected_mm = raw_mm
    
    return corrected_mm / 1000.0


def validate_bias_model(measurements_mm, expected_truth_mm, sensor_type='front'):
    """
    Validate calibration model against a known distance.
    
    Args:
        measurements_mm: List of raw measurements (mm)
        expected_truth_mm: Ground truth distance (mm)
        sensor_type: 'front', 'left', or 'right'
    
    Returns:
        dict with 'mean_error_mm', 'stdev_mm', 'corrected_mean_m', 'corrected_truth_m'
    """
    import statistics
    
    if not measurements_mm:
        return None
    
    # Compute error
    errors = [m - expected_truth_mm for m in measurements_mm]
    mean_error = statistics.mean(errors)
    stdev = statistics.stdev(errors) if len(measurements_mm) > 1 else 0
    
    # Corrected values
    corrected = [correct_distance(m, sensor_type, use_bias=True) for m in measurements_mm]
    corrected_mean = statistics.mean(corrected)
    corrected_truth = expected_truth_mm / 1000.0
    
    residual = abs(corrected_mean - corrected_truth)
    
    return {
        'raw_mean_mm': statistics.mean(measurements_mm),
        'mean_error_mm': mean_error,
        'stdev_mm': stdev,
        'corrected_mean_m': corrected_mean,
        'expected_truth_m': corrected_truth,
        'residual_m': residual,
        'samples': len(measurements_mm),
    }


# Calibration constants (from real test, Sept 2026)
CALIBRATION_DATA = {
    'front': {
        'ground_truth_mm': 300,
        'mean_measured_mm': 287.1,
        'mean_error_mm': -12.9,
        'stdev_mm': 4.56,
        'samples': 56,
        'sensor_model': 'TFmini LiDAR',
        'test_date': '2026-09-30',
    },
    'left': {
        'ground_truth_mm': 300,
        'mean_measured_mm': 292.8,
        'mean_error_mm': -7.2,
        'stdev_mm': 1.97,
        'samples': 53,
        'sensor_model': 'VL53L1X ToF',
        'test_date': '2026-09-30',
    },
    'right': {
        'ground_truth_mm': 300,
        'mean_measured_mm': 282.7,
        'mean_error_mm': -17.3,
        'stdev_mm': 2.34,
        'samples': 53,
        'sensor_model': 'VL53L1X ToF',
        'test_date': '2026-09-30',
    },
}


def print_calibration_summary():
    """Print a human-readable summary of sensor calibration."""
    print("\n" + "="*70)
    print("AEROTHON 2026 — SENSOR CALIBRATION SUMMARY")
    print("="*70)
    
    for sensor, data in CALIBRATION_DATA.items():
        print(f"\n{sensor.upper()} Sensor ({data['sensor_model']})")
        print(f"  Tested: {data['test_date']}")
        print(f"  Samples: {data['samples']}")
        print(f"  Ground truth: {data['ground_truth_mm']} mm")
        print(f"  Measured: {data['mean_measured_mm']:.1f} mm")
        print(f"  Bias: {data['mean_error_mm']:.1f} mm ± {data['stdev_mm']:.2f} mm (95% CI)")
        print(f"  Impact at 1 m: ±{data['stdev_mm']*3.33:.1f} mm (estimated)")
    
    print("\n" + "="*70)