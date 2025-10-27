# Eye Blink & Drowsiness Detector
# OpenCV-based implementation with continuous tracking

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional
import threading
import wave
import io
import os
from datetime import datetime

# Constants
DEFAULT_WINDOW_SEC = 60
DEFAULT_LONG_CLOSURE_SEC = 1.0
DEFAULT_ALERT_COOLDOWN_SEC = 3.0
DEFAULT_FPS_SMOOTH = 0.9

# Eye landmark indices for MediaPipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(pts):
    """Calculate Eye Aspect Ratio from 6 landmark points"""
    p1, p2, p3, p4, p5, p6 = pts
    num = euclidean(p2, p6) + euclidean(p3, p5)
    den = 2.0 * euclidean(p1, p4)
    if den == 0:
        return 0.0
    return float(num / den)

@dataclass
class MetricsBuffer:
    window_sec: float = DEFAULT_WINDOW_SEC
    buffer: Deque[Tuple[float, float, bool]] = field(default_factory=lambda: deque(maxlen=10000))

    def add(self, t: float, ear: float, is_closed: bool):
        self.buffer.append((t, ear, is_closed))
        self.prune(t)

    def prune(self, t_now: float):
        while self.buffer and (t_now - self.buffer[0][0] > self.window_sec):
            self.buffer.popleft()

    def perclos(self) -> float:
        if not self.buffer:
            return 0.0
        closed = sum(1 for _, _, c in self.buffer if c)
        return 100.0 * closed / len(self.buffer)

    def df(self) -> pd.DataFrame:
        if not self.buffer:
            return pd.DataFrame(columns=["timestamp", "ear", "is_closed"])
        ts, ear, closed = zip(*self.buffer)
        return pd.DataFrame({"timestamp": ts, "ear": ear, "is_closed": closed})

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Calibration & thresholds
        self.threshold_mode = "auto"
        self.manual_tau = 0.21
        self.tau = self.manual_tau
        
        self.open_samples: List[float] = []
        self.closed_samples: List[float] = []
        
        # Temporal logic
        self.closed_consec_frames = 0
        self.open_consec_frames = 0
        self.blink_count = 0
        self.last_state_closed = False
        
        # For blinks & long-closure
        self.min_blink_frames = 2
        self.min_long_frames = 10
        self.long_closure_sec = DEFAULT_LONG_CLOSURE_SEC
        
        # Rolling metrics
        self.metrics = MetricsBuffer(window_sec=DEFAULT_WINDOW_SEC)
        
        # FPS estimation
        self.last_ts = None
        self.ema_fps = None
        
        # Alerts
        self.last_alert_ts = 0.0
        self.alert_cooldown_sec = DEFAULT_ALERT_COOLDOWN_SEC
        self.should_play_beep = False
        
        # Logging
        self.event_log: List[dict] = []
        
        # Live values
        self.live_values = {
            "ear": 0.0,
            "perclos": 0.0,
            "blink_rate": 0.0,
            "fps": 0.0,
            "is_closed": False,
            "long_closure": False,
        }
        
        # Calibration state
        self.calibrating_open = False
        self.calibrating_closed = False
        self.calibration_start_time = 0
        self.calibration_samples = []

    def _update_fps(self):
        now = time.time()
        if self.last_ts is None:
            self.last_ts = now
            return
        dt = now - self.last_ts
        self.last_ts = now
        if dt > 0:
            inst_fps = 1.0 / dt
            if self.ema_fps is None:
                self.ema_fps = inst_fps
            else:
                self.ema_fps = DEFAULT_FPS_SMOOTH * self.ema_fps + (1 - DEFAULT_FPS_SMOOTH) * inst_fps
        self.live_values["fps"] = round(self.ema_fps or 0.0, 1)
        
        if self.ema_fps:
            self.min_long_frames = max(3, int(self.long_closure_sec * self.ema_fps))

    def _calc_ear(self, landmarks_norm, image_w, image_h):
        def pick(indices):
            return [(landmarks_norm[i].x, landmarks_norm[i].y) for i in indices]
        left_pts = pick(LEFT_EYE)
        right_pts = pick(RIGHT_EYE)
        left_ear = eye_aspect_ratio(left_pts)
        right_ear = eye_aspect_ratio(right_pts)
        return (left_ear + right_ear) / 2.0

    def _compute_threshold(self):
        if self.threshold_mode != "auto":
            self.tau = self.manual_tau
            return
        if self.open_samples and self.closed_samples:
            mu_open = float(np.mean(self.open_samples))
            mu_closed = float(np.mean(self.closed_samples))
            self.tau = (mu_open + mu_closed) / 2.0
        else:
            self.tau = self.manual_tau

    def start_calibration(self, calibration_type: str):
        """Start continuous calibration"""
        if calibration_type == "open":
            self.calibrating_open = True
            self.calibrating_closed = False
            self.calibration_samples = []
            print("Calibrating OPEN eyes - keep your eyes open...")
        elif calibration_type == "closed":
            self.calibrating_open = False
            self.calibrating_closed = True
            self.calibration_samples = []
            print("Calibrating CLOSED eyes - close your eyes...")
        self.calibration_start_time = time.time()

    def stop_calibration(self):
        """Stop calibration and process samples"""
        if self.calibrating_open and len(self.calibration_samples) > 10:
            self.open_samples.extend(self.calibration_samples)
            mu = float(np.mean(self.calibration_samples))
            print(f"Open calibration complete: {len(self.calibration_samples)} samples, mean EAR: {mu:.3f}")
        elif self.calibrating_closed and len(self.calibration_samples) > 10:
            self.closed_samples.extend(self.calibration_samples)
            mu = float(np.mean(self.calibration_samples))
            print(f"Closed calibration complete: {len(self.calibration_samples)} samples, mean EAR: {mu:.3f}")
        
        self.calibrating_open = False
        self.calibrating_closed = False
        self.calibration_samples = []
        self._compute_threshold()
        print(f"New threshold: {self.tau:.3f}")

    def process_frame(self, frame):
        """Process a single frame and return annotated frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self._update_fps()
        
        results = self.face_mesh.process(rgb)
        ear = 0.0
        is_closed = False
        long_closure = False
        
        h, w, _ = frame.shape
        
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            ear = self._calc_ear(lms, w, h)
            self._compute_threshold()
            is_closed = ear < self.tau
            
            # Collect calibration samples during calibration
            if self.calibrating_open or self.calibrating_closed:
                if ear > 0:  # Valid EAR reading
                    self.calibration_samples.append(ear)
            
            # Blink detection
            if is_closed:
                self.closed_consec_frames += 1
                self.open_consec_frames = 0
            else:
                self.open_consec_frames += 1
                if self.last_state_closed and self.closed_consec_frames >= self.min_blink_frames:
                    self.blink_count += 1
                self.closed_consec_frames = 0
            
            self.last_state_closed = is_closed
            
            # Long closure detection
            if self.closed_consec_frames >= self.min_long_frames:
                long_closure = True
            
            tnow = time.time()
            self.metrics.add(tnow, ear, is_closed)
            perclos = self.metrics.perclos()
            
            # Blink rate calculation
            df = self.metrics.df()
            blink_rate = 0.0
            if len(df) > 1:
                duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
                if duration > 10:
                    blink_rate = (self.blink_count / duration) * 60.0
            
            # Alerts
            self.should_play_beep = False
            if long_closure or perclos >= 40.0:  # Default threshold
                if (tnow - self.last_alert_ts) > self.alert_cooldown_sec:
                    self.last_alert_ts = tnow
                    self.should_play_beep = True
            
            # Logging
            self.event_log.append({
                "timestamp": tnow,
                "ear": ear,
                "is_closed": is_closed,
                "perclos": perclos,
                "blink_count": self.blink_count,
                "blink_rate_per_min": blink_rate,
                "fps": self.live_values["fps"],
                "long_closure": long_closure
            })
            
            # Update live values
            self.live_values.update({
                "ear": round(ear, 4),
                "perclos": round(perclos, 1),
                "blink_rate": round(blink_rate, 1),
                "is_closed": is_closed,
                "long_closure": long_closure,
            })
            
            # Draw overlays
            self._draw_overlays(frame, lms, w, h, ear, is_closed, long_closure, perclos)
        
        return frame

    def _draw_overlays(self, frame, landmarks, w, h, ear, is_closed, long_closure, perclos):
        # Background rectangle for text
        cv2.rectangle(frame, (5, 5), (500, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (500, 200), (255, 255, 255), 2)
        
        # Status color
        status_color = (0, 255, 0) if not is_closed else (0, 0, 255)
        alert_color = (0, 0, 255) if long_closure else (255, 255, 0)
        
        # Status text
        status_text = "EYES OPEN" if not is_closed else "EYES CLOSED"
        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # EAR and threshold
        cv2.putText(frame, f"EAR: {ear:.3f} (threshold: {self.tau:.3f})", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blink count
        cv2.putText(frame, f"Blinks: {self.blink_count}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # PERCLOS
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calibration status
        if self.calibrating_open:
            cv2.putText(frame, "CALIBRATING OPEN EYES...", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.calibrating_closed:
            cv2.putText(frame, "CALIBRATING CLOSED EYES...", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Alert message
        if long_closure:
            cv2.putText(frame, "ALERT: DROWSINESS DETECTED!", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        
        # Draw eye landmarks
        for idx in LEFT_EYE:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
        
        for idx in RIGHT_EYE:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
        
        # Draw eye contours
        left_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        
        cv2.polylines(frame, [np.array(left_eye_points)], True, (0, 255, 255), 3)
        cv2.polylines(frame, [np.array(right_eye_points)], True, (255, 0, 255), 3)

    def save_csv(self, filename=None):
        """Save event log to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drowsy_log_{timestamp}.csv"
        
        df = pd.DataFrame(self.event_log)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return filename

def play_beep():
    """Play a beep sound - non-blocking to prevent lag"""
    def beep_thread():
        try:
            # Generate beep sound
            sr = 24000
            dur = 0.3
            t = np.linspace(0, dur, int(sr*dur), endpoint=False)
            sig = (0.3*np.sin(2*np.pi*440*t)).astype(np.float32)
            
            # Save to temporary file and play
            temp_file = "temp_beep.wav"
            with wave.open(temp_file, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                data16 = (sig * 32767).astype(np.int16).tobytes()
                w.writeframes(data16)
            
            # Play using system command (works on macOS) - run in background
            os.system(f"afplay {temp_file} &")
            os.remove(temp_file)
        except Exception as e:
            # Silent fail to avoid interrupting main thread
            pass
    
    # Run beep in separate thread to avoid blocking
    threading.Thread(target=beep_thread, daemon=True).start()

def main():
    """Main application loop"""
    print("Eye Blink & Drowsiness Detector")
    print("Press 'q' to quit, 'o' to calibrate open eyes, 'c' to calibrate closed eyes")
    print("Press 's' to stop calibration, 'd' to download CSV data")
    
    # Initialize detector
    detector = DrowsinessDetector()
    
    # Initialize camera with better error handling
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Try different camera indices if 0 fails
    if not cap.isOpened():
        print("Camera 0 failed, trying camera 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open any camera")
        print("Please check:")
        print("1. Camera permissions are granted to Terminal")
        print("2. No other applications are using the camera")
        print("3. Camera is connected and working")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized. Starting detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('Drowsiness Detector', processed_frame)
            
            # Play beep if alert
            if detector.should_play_beep:
                threading.Thread(target=play_beep, daemon=True).start()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                detector.start_calibration("open")
            elif key == ord('c'):
                detector.start_calibration("closed")
            elif key == ord('s'):
                detector.stop_calibration()
            elif key == ord('d'):
                detector.save_csv()
            
            # Print status every 5 seconds
            if int(time.time()) % 5 == 0:
                live = detector.live_values
                print(f"EAR: {live['ear']:.3f}, PERCLOS: {live['perclos']:.1f}%, Blinks: {detector.blink_count}, FPS: {live['fps']:.1f}")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final data
        if detector.event_log:
            detector.save_csv()
            print(f"Session complete. Total blinks: {detector.blink_count}")

if __name__ == "__main__":
    main()
