"""
sim/make_test_videos.py
Makes two synthetic videos with a QR code, sized for the camera at 5 m (start)
and 10 m (target). Run from the project root:
  python -m sim.make_test_videos
"""
import cv2
import numpy as np
import qrcode
from PIL import Image

W, H, FPS, FRAMES = 1280, 800, 30, 60
START_TEXT = "AEROTHON_START_CODE_HERE"   # replace with the real start code
BG = 70                                   # dark background so the white border stands out


def make_video(path, text, box_px):
    qr_img = qrcode.make(text, border=4).convert("L")
    qr = np.array(qr_img.resize((box_px, box_px), Image.NEAREST))
    canvas = np.full((H, W), BG, np.uint8)
    y0, x0 = (H - box_px) // 2, (W - box_px) // 2
    canvas[y0:y0 + box_px, x0:x0 + box_px] = qr
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(FRAMES):
        writer.write(frame)
    writer.release()
    print(f"wrote {path} ({box_px} px QR)")


if __name__ == "__main__":
    # Sizes come from the adapter's own formula: about 461 px at 5 m, 230 px at 10 m.
    make_video("sim/test_start_5m.mp4", START_TEXT, 461)
    make_video("sim/test_target_10m.mp4", START_TEXT, 230)