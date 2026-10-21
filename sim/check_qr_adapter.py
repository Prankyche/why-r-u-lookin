"""
sim/check_qr_adapter.py
Runs a recorded video through QRAdapter twice:
  pass 1 in SCAN_START_QR (looks for the start code)
  pass 2 in SEARCH_QR    (looks for the same code again)
Usage (from project root):
  python -m sim.check_qr_adapter path\to\recording.mp4
"""
import sys
from config import DELIVERY_ZONE_ALTITUDE_M as DELIVERY_ALT
from fsm.mission_fsm import State
from adapters.qr_adapter import QRAdapter
from vision.qr.qr_whiteborder import video_frames


# def main():
#     if len(sys.argv) < 2:
#         sys.exit("usage: python -m sim.check_qr_adapter <video>")
#     path = sys.argv[1]
#     adapter = QRAdapter()

#     print("--- pass 1: SCAN_START_QR ---")
#     for idx, frame in video_frames(path):
#         events = adapter.poll(State.SCAN_START_QR, frame)
#         if events:
#             print(f"frame {idx}: {events}")
#             break
#     else:
#         print("no start code confirmed in this video")
#         return
#     print(f"start text: {adapter.start_text!r}")

#     print("--- pass 2: SEARCH_QR ---")
#     for idx, frame in video_frames(path):
#         events = adapter.poll(State.SEARCH_QR, frame)
#         if any(e.name == "QR_MATCH" for e in events):
#             print(f"frame {idx}: {events}")
#             print("offset (image axes, m):", adapter.target_offset_m(config_alt()))
#             break
#     else:
#         print("target not confirmed in this video")

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python -m sim.check_qr_adapter <start_video> <target_video>")
    start_video, target_video = sys.argv[1], sys.argv[2]
    adapter = QRAdapter()

    print("--- pass 1: SCAN_START_QR (5 m) ---")
    for idx, frame in video_frames(start_video):
        events = adapter.poll(State.SCAN_START_QR, frame)
        if events:
            print(f"frame {idx}: {events}")
            break
    else:
        print("no start code confirmed")
        return
    print(f"start text: {adapter.start_text!r}")

    print("--- pass 2: SEARCH_QR (10 m) ---")
    for idx, frame in video_frames(target_video):
        events = adapter.poll(State.SEARCH_QR, frame)
        if any(e.name == "QR_MATCH" for e in events):
            print(f"frame {idx}: {events}")
            print("offset (image axes, m):", adapter.target_offset_m(DELIVERY_ALT))
            break
    else:
        print("target not confirmed")

def config_alt():
    import config
    return config.DELIVERY_ZONE_ALTITUDE_M


if __name__ == "__main__":
    main()