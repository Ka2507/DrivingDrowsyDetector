#!/usr/bin/env python3
"""
Mobile-Optimized Drowsiness Detector for Car Mount Usage
Supports Android/iOS deployment with low-light handling
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pandas as pd
import os
import platform
import threading
from datetime import datetime

# For mobile deployment
try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.image import Image
    from kivy.clock import Clock
    from kivy.graphics.texture import Texture
    from kivy.core.window import Window
    from kivy.uix.popup import Popup
    from kivy.uix.gridlayout import GridLayout
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False
    print("Kivy not available. Install with: pip install kivy")

# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.21
CLOSED_EYES_FRAME_THRESHOLD = 15
PERCLOS_WINDOW_SEC = 60
ALERT_COOLDOWN_SEC = 3.0

# Low-light detection thresholds
LOW_LIGHT_THRESHOLD = 50  # Average brightness below this triggers enhancement
BRIGHTNESS_CHECK_INTERVAL = 30  # Check brightness every 30 frames

class LowLightHandler:
    """Handles low-light detection and camera enhancement"""
    
    def __init__(self):
        self.brightness_history = deque(maxlen=10)
        self.is_low_light = False
        self.frame_count = 0
        
    def detect_low_light(self, frame):
        """Detect if the current frame is in low light conditions"""
        self.frame_count += 1
        
        if self.frame_count % BRIGHTNESS_CHECK_INTERVAL == 0:
            # Convert to grayscale and calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            self.brightness_history.append(brightness)
            
            # Use median to avoid flickering
            if len(self.brightness_history) >= 3:
                median_brightness = np.median(list(self.brightness_history))
                self.is_low_light = median_brightness < LOW_LIGHT_THRESHOLD
        
        return self.is_low_light
    
    def enhance_frame(self, frame):
        """Enhance frame for low-light conditions"""
        if not self.is_low_light:
            return frame
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for very dark conditions
        if np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)) < 30:
            gamma = 1.5
            enhanced = np.power(enhanced / 255.0, gamma) * 255
            enhanced = np.uint8(enhanced)
        
        return enhanced

class MobileDrowsinessDetector:
    """Mobile-optimized drowsiness detector"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.low_light_handler = LowLightHandler()
        
        # Calibration data
        self.open_eye_samples = []
        self.closed_eye_samples = []
        self.current_ear_threshold = EAR_THRESHOLD
        self.is_calibrating_open = False
        self.is_calibrating_closed = False
        
        # Metrics
        self.blink_count = 0
        self.frames_closed_consec = 0
        self.last_state_closed = False
        self.last_alert_time = 0
        self.metrics_buffer = deque(maxlen=int(PERCLOS_WINDOW_SEC * 30))
        self.event_log = []
        
        # Mobile-specific
        self.is_running = False
        self.cap = None
        self.current_frame = None
        self.last_update_time = 0
        
    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def eye_aspect_ratio(self, eye_landmarks):
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        A = self.euclidean_distance(p2, p6)
        B = self.euclidean_distance(p3, p5)
        C = self.euclidean_distance(p1, p4)
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    
    def play_beep(self):
        """Cross-platform beep sound - non-blocking"""
        def beep_thread():
            try:
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(440, 200)
                elif platform.system() == "Darwin":  # macOS
                    os.system('afplay /System/Library/Sounds/Ping.aiff &')
                else:  # Linux/Android
                    os.system('play -nq -c1 synth 0.2 sine 440 2>/dev/null &')
            except:
                # Silent fail if audio not available
                pass
        
        # Run beep in separate thread to avoid blocking
        threading.Thread(target=beep_thread, daemon=True).start()
    
    def update_metrics(self, ear, is_closed, t_now):
        """Update drowsiness metrics"""
        self.metrics_buffer.append((t_now, ear, is_closed))
        
        # Blink detection
        if is_closed:
            self.frames_closed_consec += 1
        else:
            if self.last_state_closed and self.frames_closed_consec > 1:
                self.blink_count += 1
            self.frames_closed_consec = 0
        self.last_state_closed = is_closed
        
        # PERCLOS calculation
        closed_frames_in_window = sum(1 for _, _, closed in self.metrics_buffer if closed)
        perclos = (closed_frames_in_window / len(self.metrics_buffer)) * 100 if self.metrics_buffer else 0
        
        # Long closure alert - optimized to reduce lag
        long_closure = self.frames_closed_consec >= CLOSED_EYES_FRAME_THRESHOLD
        if long_closure and (t_now - self.last_alert_time > ALERT_COOLDOWN_SEC):
            # Play beep in background thread to avoid blocking
            self.play_beep()
            self.last_alert_time = t_now
        
        # Log event
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "ear": ear,
            "is_closed": is_closed,
            "perclos": perclos,
            "blink_count": self.blink_count,
            "long_closure_alert": long_closure,
            "low_light_mode": self.low_light_handler.is_low_light
        })
        
        return perclos, long_closure
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        if frame is None:
            return None, {}
        
        # Low-light detection and enhancement
        is_low_light = self.low_light_handler.detect_low_light(frame)
        if is_low_light:
            frame = self.low_light_handler.enhance_frame(frame)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        ear = 0.0
        is_closed = False
        perclos = 0.0
        long_closure = False
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            
            left_ear = self.eye_aspect_ratio(left_eye_pts)
            right_ear = self.eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0
            
            # Calibration logic
            if self.is_calibrating_open:
                self.open_eye_samples.append(ear)
            elif self.is_calibrating_closed:
                self.closed_eye_samples.append(ear)
            else:
                if self.open_eye_samples and self.closed_eye_samples:
                    mu_open = np.mean(self.open_eye_samples)
                    mu_closed = np.mean(self.closed_eye_samples)
                    self.current_ear_threshold = (mu_open + mu_closed) / 2.0
                else:
                    self.current_ear_threshold = EAR_THRESHOLD
                
                is_closed = ear < self.current_ear_threshold
                perclos, long_closure = self.update_metrics(ear, is_closed, time.time())
            
            # Draw overlays
            color = (0, 255, 0) if not is_closed else (0, 0, 255)
            
            # Mobile-optimized UI overlay
            overlay_height = 120
            cv2.rectangle(display_frame, (5, 5), (w-5, overlay_height), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (w-5, overlay_height), (255, 255, 255), 2)
            
            # Status text
            status_text = "EYES OPEN" if not is_closed else "EYES CLOSED"
            cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Metrics
            cv2.putText(display_frame, f"EAR: {ear:.3f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Blinks: {self.blink_count}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"PERCLOS: {perclos:.1f}%", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Low-light indicator
            if is_low_light:
                cv2.putText(display_frame, "LOW LIGHT MODE", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Alert - optimized rendering to reduce lag
            if long_closure:
                # Use simpler text rendering for alerts to reduce lag
                cv2.putText(display_frame, "ALERT!", (w//2-50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw eye landmarks
            for pts in [left_eye_pts, right_eye_pts]:
                cv2.polylines(display_frame, [np.array(pts)], True, (0, 255, 255), 2)
                for (x, y) in pts:
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 255), -1)
        else:
            cv2.putText(display_frame, "No face detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if is_low_light:
                cv2.putText(display_frame, "Low light - ensure good lighting", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return display_frame, {
            'ear': ear,
            'is_closed': is_closed,
            'perclos': perclos,
            'blink_count': self.blink_count,
            'long_closure': long_closure,
            'is_low_light': is_low_light
        }
    
    def start_calibration(self, calibration_type):
        """Start calibration process"""
        if calibration_type == 'open':
            self.is_calibrating_open = True
            self.is_calibrating_closed = False
            self.open_eye_samples.clear()
        elif calibration_type == 'closed':
            self.is_calibrating_closed = True
            self.is_calibrating_open = False
            self.closed_eye_samples.clear()
    
    def stop_calibration(self):
        """Stop calibration and calculate new threshold"""
        self.is_calibrating_open = False
        self.is_calibrating_closed = False
        
        if self.open_eye_samples and self.closed_eye_samples:
            mu_open = np.mean(self.open_eye_samples)
            mu_closed = np.mean(self.closed_eye_samples)
            self.current_ear_threshold = (mu_open + mu_closed) / 2.0
            return f"Calibration complete. Threshold: {self.current_ear_threshold:.3f}"
        else:
            return "Not enough samples for calibration"
    
    def save_csv_log(self):
        """Save session data to CSV"""
        if self.event_log:
            df = pd.DataFrame(self.event_log)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drowsiness_log_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return f"Data saved to {filename}"
        else:
            return "No data to save"

if KIVY_AVAILABLE:
    class MobileDrowsinessApp(App):
        """Kivy-based mobile app"""
        
        def build(self):
            self.detector = MobileDrowsinessDetector()
            self.setup_ui()
            self.setup_camera()
            return self.root
        
        def setup_ui(self):
            """Setup mobile-optimized UI"""
            # Main layout
            self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)
            
            # Video display
            self.video_image = Image(size_hint=(1, 0.7))
            self.root.add_widget(self.video_image)
            
            # Status display
            self.status_label = Label(
                text="Starting camera...",
                size_hint=(1, 0.1),
                font_size=20,
                color=(1, 1, 1, 1)
            )
            self.root.add_widget(self.status_label)
            
            # Control buttons
            controls = GridLayout(cols=3, size_hint=(1, 0.1), spacing=10)
            
            self.calibrate_open_btn = Button(text="Calibrate\nOpen Eyes")
            self.calibrate_open_btn.bind(on_press=self.start_open_calibration)
            
            self.calibrate_closed_btn = Button(text="Calibrate\nClosed Eyes")
            self.calibrate_closed_btn.bind(on_press=self.start_closed_calibration)
            
            self.stop_calibrate_btn = Button(text="Stop\nCalibration")
            self.stop_calibrate_btn.bind(on_press=self.stop_calibration)
            
            controls.add_widget(self.calibrate_open_btn)
            controls.add_widget(self.calibrate_closed_btn)
            controls.add_widget(self.stop_calibrate_btn)
            
            self.root.add_widget(controls)
            
            # Bottom controls
            bottom_controls = GridLayout(cols=2, size_hint=(1, 0.1), spacing=10)
            
            self.save_btn = Button(text="Save Data")
            self.save_btn.bind(on_press=self.save_data)
            
            self.quit_btn = Button(text="Quit")
            self.quit_btn.bind(on_press=self.quit_app)
            
            bottom_controls.add_widget(self.save_btn)
            bottom_controls.add_widget(self.quit_btn)
            
            self.root.add_widget(bottom_controls)
            
            # Set dark theme
            Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        def setup_camera(self):
            """Setup camera and start processing"""
            self.detector.cap = cv2.VideoCapture(0)
            if not self.detector.cap.isOpened():
                self.status_label.text = "Camera not available"
                return
            
            # Mobile-optimized camera settings
            self.detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.detector.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.detector.is_running = True
            Clock.schedule_interval(self.update_frame, 1.0/30.0)  # 30 FPS
        
        def update_frame(self, dt):
            """Update video frame"""
            if not self.detector.is_running:
                return
            
            ret, frame = self.detector.cap.read()
            if not ret:
                return
            
            processed_frame, metrics = self.detector.process_frame(frame)
            if processed_frame is not None:
                # Convert frame to texture
                buf = cv2.flip(processed_frame, 0).tobytes()
                texture = Texture.create(size=(processed_frame.shape[1], processed_frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.video_image.texture = texture
                
                # Update status
                status = f"EAR: {metrics['ear']:.3f} | Blinks: {metrics['blink_count']} | PERCLOS: {metrics['perclos']:.1f}%"
                if metrics['is_low_light']:
                    status += " | LOW LIGHT"
                if metrics['long_closure']:
                    status += " | ALERT!"
                
                self.status_label.text = status
        
        def start_open_calibration(self, instance):
            self.detector.start_calibration('open')
            self.status_label.text = "Calibrating open eyes - keep eyes open"
        
        def start_closed_calibration(self, instance):
            self.detector.start_calibration('closed')
            self.status_label.text = "Calibrating closed eyes - close eyes"
        
        def stop_calibration(self, instance):
            result = self.detector.stop_calibration()
            self.status_label.text = result
        
        def save_data(self, instance):
            result = self.detector.save_csv_log()
            self.status_label.text = result
        
        def quit_app(self, instance):
            self.detector.is_running = False
            if self.detector.cap:
                self.detector.cap.release()
            App.get_running_app().stop()

def main():
    """Main function - choose between desktop and mobile versions"""
    if KIVY_AVAILABLE and platform.system() != "Windows":
        # Run mobile version
        print("Starting mobile version...")
        MobileDrowsinessApp().run()
    else:
        # Run desktop version with mobile optimizations
        print("Starting desktop version with mobile optimizations...")
        
        detector = MobileDrowsinessDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Mobile-optimized settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        cv2.namedWindow("Mobile Drowsiness Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mobile Drowsiness Detector", 800, 600)
        
        print("\n--- Mobile Drowsiness Detector Controls ---")
        print("Press 'q' to quit")
        print("Press 'o' to calibrate open eyes")
        print("Press 'c' to calibrate closed eyes")
        print("Press 's' to stop calibration")
        print("Press 'd' to save CSV data")
        print("Press 'l' to toggle low-light mode info")
        print("------------------------------------------\n")
        
        detector.is_running = True
        
        while detector.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, metrics = detector.process_frame(frame)
            if processed_frame is not None:
                cv2.imshow("Mobile Drowsiness Detector", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                detector.start_calibration('open')
                print("Started calibrating OPEN eyes")
            elif key == ord('c'):
                detector.start_calibration('closed')
                print("Started calibrating CLOSED eyes")
            elif key == ord('s'):
                result = detector.stop_calibration()
                print(result)
            elif key == ord('d'):
                result = detector.save_csv_log()
                print(result)
            elif key == ord('l'):
                print(f"Low light mode: {detector.low_light_handler.is_low_light}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
