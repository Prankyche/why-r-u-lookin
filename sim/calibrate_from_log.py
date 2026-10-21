"""calibrate_from_log.py  --  turn logged sensor values into config.py numbers.

Usage (from the project folder):
    python calibrate_from_log.py "L_Sensor Dataset.docx" --ref 300 --sensor left
    python calibrate_from_log.py "TF_mini_lidar.docx"      --ref 300 --sensor front --unit cm

Accepts .txt / .csv / .log / .docx (use --unit cm for TFmini logs, which print centimetres). Reads either lines like
    Measured Distance: 290 mm | Error: -10 mm
or a plain list of numbers (one per line / comma separated, in mm).
--ref is the TRUE distance in mm the sensor was held at during the log.
Prints the bias (add this to raw readings) and noise std, plus lines to paste into config.py.
"""
import argparse
import re
import statistics
import sys


def read_text(path):
    if path.lower().endswith(".docx"):
        import docx
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" ".join(c.text for c in row.cells))
        return "\n".join(parts)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_mm(text):
    vals = [float(x) for x in re.findall(r"Measured Distance:\s*(-?\d+(?:\.\d+)?)", text)]
    if vals:
        return vals
    # fall back: first number on each line (handles "290", "290 mm", "12.3, 290")
    for line in text.splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if nums:
            vals.append(float(nums[-1] if "," in line else nums[0]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--ref", type=float, required=True, help="true distance in mm")
    ap.add_argument("--sensor", choices=["front", "left", "right"], default="front")
    ap.add_argument("--max-range", type=float, default=None, help="drop readings >= this (mm), e.g. no-return values")
    ap.add_argument("--unit", choices=["mm", "cm"], default="mm", help="unit of the logged numbers (TFmini prints cm)")
    a = ap.parse_args()

    vals = extract_mm(read_text(a.file))
    if a.unit == "cm":
        vals = [v * 10 for v in vals]
    if a.max_range:
        vals = [v for v in vals if v < a.max_range]
    # drop wild outliers (>50 mm off the median) so one bad sample doesn't wreck the bias
    if len(vals) >= 5:
        med = statistics.median(vals)
        kept = [v for v in vals if abs(v - med) <= 50]
        dropped = len(vals) - len(kept)
        vals = kept
    else:
        dropped = 0
    if len(vals) < 5:
        sys.exit(f"Only {len(vals)} usable values found -- check the file format.")

    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    bias = a.ref - mean
    name = a.sensor.upper()
    print(f"{a.sensor}: {len(vals)} samples ({dropped} outliers dropped)")
    print(f"  mean reading {mean:.1f} mm vs true {a.ref:.0f} mm  ->  bias {bias:+.1f} mm,  noise std {std:.2f} mm")
    print(f"  min {min(vals):.0f}  max {max(vals):.0f}")
    print("\nPaste into config.py:")
    print(f"{name}_SENSOR_BIAS_MM = {bias:.1f}")
    print(f"{name}_SENSOR_NOISE_STD_MM = {std:.2f}")
    print("\nNote: one reference distance can't separate a fixed offset from a scale error;")
    print("log at 2-3 distances (e.g. 100, 500, 1000 mm) and compare the biases.")


if __name__ == "__main__":
    main()