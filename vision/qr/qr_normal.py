import cv2
import numpy as np
from pyzbar.pyzbar import decode
from typing import Optional, Dict
import time

class QRDetectionPipeline_Pi_Optimized:
    """
    Optimized for Raspberry Pi 5 at 10m altitude.
    Priority: Speed + Accuracy (skip expensive YOLO).
    """
    
    def __init__(self):
        self.frame_count = 0
        self.last_valid_qr = None
        self.consecutive_misses = 0
        self.max_misses_before_preprocess = 5  # After 5 direct fails, try preprocessing
    
    def direct_decode(self, image: np.ndarray) -> Optional[Dict]:
        """Fast pyzbar decode on original frame."""
        try:
            results = decode(image)
            if results:
                qr_data = results[0]
                return {
                    "data": qr_data.data.decode("utf-8"),
                    "type": qr_data.type,
                    "rect": qr_data.rect,
                    "attempt": 1,
                    "success": True,
                    "timestamp": time.time()
                }
        except:
            pass
        return None
    
    def enhanced_decode(self, image: np.ndarray) -> Optional[Dict]:
        """
        Enhanced detection for 10m altitude (small QR codes).
        Uses contrast + sharpening without expensive rotation/deskewing.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 1.0
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            results = decode(sharpened)
            if results:
                qr_data = results[0]
                return {
                    "data": qr_data.data.decode("utf-8"),
                    "type": qr_data.type,
                    "rect": qr_data.rect,
                    "attempt": 2,
                    "success": True,
                    "preprocessing": "clahe+sharpen",
                    "timestamp": time.time()
                }
        except:
            pass
        return None
    
    def detect_and_decode(self, image: np.ndarray, force_enhanced: bool = False) -> Optional[Dict]:
        self.frame_count += 1
        
        result = self.direct_decode(image)
        if result:
            self.consecutive_misses = 0
            self.last_valid_qr = result
            return result
        
        self.consecutive_misses += 1
        
        if force_enhanced or self.consecutive_misses >= self.max_misses_before_preprocess:
            result = self.enhanced_decode(image)
            if result:
                self.consecutive_misses = 0
                self.last_valid_qr = result
                return result
        
        return None
    
    def process_frame_stream(self, frame_generator, max_frames: int = None, 
                            callback_on_detect=None):
        start_time = time.time()
        frame_num = 0
        last_fps_calc = start_time
        fps = 0
        
        for frame in frame_generator:
            if max_frames and frame_num >= max_frames:
                break
            
            frame_num += 1
            
            result = self.detect_and_decode(frame)
            
            if result and callback_on_detect:
                callback_on_detect(frame_num, result)
            
            if frame_num % 30 == 0:
                elapsed = time.time() - last_fps_calc
                fps = 30 / elapsed if elapsed > 0 else 0
                last_fps_calc = time.time()
            
            yield frame_num, result, fps


def camera_frame_generator(camera_id=0, resize_width=None):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize_width:
            h = int(frame.shape[0] * resize_width / frame.shape[1])
            frame = cv2.resize(frame, (resize_width, h))
        
        yield frame
    
    cap.release()


def video_frame_generator(video_path, resize_width=None, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        if resize_width:
            h = int(frame.shape[0] * resize_width / frame.shape[1])
            frame = cv2.resize(frame, (resize_width, h))
        
        yield frame
    
    cap.release()


# ============================================
# USAGE EXAMPLES FOR AEROTHON
# ============================================

def example_live_camera_detection():
    """Real-time detection from camera (simulates drone onboard Pi)."""
    print("=" * 60)
    print("QR DETECTION - LIVE CAMERA (10m altitude simulation)")
    print("=" * 60)
    
    pipeline = QRDetectionPipeline_Pi_Optimized()
    
    def on_detect(frame_num, result):
        print(f"\n✓ QR DETECTED at frame {frame_num}")
        print(f"  Data: {result['data']}")
        print(f"  Attempt: {result['attempt']}")
        print(f"  Timestamp: {result['timestamp']}")
    
    try:
        frame_gen = camera_frame_generator(camera_id=0, resize_width=640)
        
        for frame_num, result, fps in pipeline.process_frame_stream(
            frame_gen, 
            callback_on_detect=on_detect
        ):
            if frame_num % 30 == 0:
                status = f"✓ {result['attempt']}" if result else "✗"
                print(f"Frame {frame_num}: {status} | FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")


def example_video_detection(video_path):
    """
    Detect QR in video file.
    Use this to test on recorded drone footage.
    """
    print("=" * 60)
    print("QR DETECTION - VIDEO FILE")
    print("=" * 60)
    print(f"Input: {video_path}\n")
    
    pipeline = QRDetectionPipeline_Pi_Optimized()
    
    stats = {
        "total_frames": 0,
        "detected_frames": 0,
        "attempt1_success": 0,
        "attempt2_success": 0,
        "qr_data": []
    }
    
    def on_detect(frame_num, result):
        stats["detected_frames"] += 1
        if result["attempt"] == 1:
            stats["attempt1_success"] += 1
        else:
            stats["attempt2_success"] += 1
        
        stats["qr_data"].append({
            "frame": frame_num,
            "data": result["data"],
            "attempt": result["attempt"]
        })
        
        print(f"  Frame {frame_num}: ✓ {result['data']} (Attempt {result['attempt']})")
    
    try:
        frame_gen = video_frame_generator(video_path, resize_width=640, skip_frames=1)
        
        for frame_num, result, fps in pipeline.process_frame_stream(
            frame_gen,
            callback_on_detect=on_detect
        ):
            stats["total_frames"] = frame_num
            
            if frame_num % 60 == 0:
                print(f"Processing... Frame {frame_num} | FPS: {fps:.1f}")
        
    except FileNotFoundError:
        print(f"Error: {video_path} not found")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total frames: {stats['total_frames']}")
    print(f"QR detected in {stats['detected_frames']} frames ({100*stats['detected_frames']/stats['total_frames']:.1f}%)")
    print(f"  Attempt 1 (Direct): {stats['attempt1_success']}")
    print(f"  Attempt 2 (Enhanced): {stats['attempt2_success']}")
    
    if stats["qr_data"]:
        print(f"\nDetected QR Codes:")
        unique_qrs = {}
        for det in stats["qr_data"]:
            qr_str = det["data"]
            if qr_str not in unique_qrs:
                unique_qrs[qr_str] = []
            unique_qrs[qr_str].append(det["frame"])
        
        for qr_str, frames in unique_qrs.items():
            print(f"  '{qr_str}': {len(frames)} frames (e.g., frame {frames[0]})")


def example_onboard_pi_mission_simulation():
    """
    Simulate Aerothon mission QR detection on Pi.
    This is what runs on Raspberry Pi 5 during actual flight.
    """
    print("=" * 60)
    print("AEROTHON MISSION - QR DETECTION PHASE")
    print("=" * 60)
    print("Altitude: 10m | Task: Detect, decode, navigate to drop zone\n")
    
    pipeline = QRDetectionPipeline_Pi_Optimized()
    
    # Simulate: drone at 10m, scanning for QR
    qr_location = None
    
    def on_detect(frame_num, result):
        nonlocal qr_location
        qr_location = result["data"]
        print(f"\n{'='*60}")
        print(f"✓✓✓ QR CODE FOUND ✓✓✓")
        print(f"{'='*60}")
        print(f"Data: {qr_location}")
        print(f"Frame: {frame_num} | Attempt: {result['attempt']}")
        print(f"Navigation: Travel to coordinates from QR data")
        print(f"Next: Descend to 5m and drop payload")
        print(f"{'='*60}\n")
    
    # In real flight, this would be from the onboard camera
    # For testing, use video file
    video_path = input("Enter video path (drone footage at 10m): ").strip()
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return
    
    try:
        frame_gen = video_frame_generator(video_path, resize_width=640)
        
        for frame_num, result, fps in pipeline.process_frame_stream(
            frame_gen,
            callback_on_detect=on_detect
        ):
            if frame_num % 30 == 0:
                status = "scanning..."
                if result:
                    status = f"✓ DETECTED!"
                print(f"[{frame_num}] {status} | FPS: {fps:.1f}")
            
            # Stop once QR is found (in real mission, you'd verify + navigate)
            if qr_location:
                print(f"\nMission: Proceed to navigation phase with: {qr_location}")
                break
    
    except KeyboardInterrupt:
        print("\nMission interrupted")


if __name__ == "__main__":
    import os
    
    print("\nQR Detection Pipeline - Optimized for Raspberry Pi 5 @ 10m")
    print("="*60)
    print("1. Live camera detection")
    print("2. Process video file")
    print("3. Aerothon mission simulation")
    
    choice = input("\nSelect mode (1-3): ").strip()
    
    if choice == "1":
        example_live_camera_detection()
    elif choice == "2":
        video = input("Enter video path: ").strip()
        example_video_detection(video)
    elif choice == "3":
        example_onboard_pi_mission_simulation()
    else:
        print("Invalid choice")