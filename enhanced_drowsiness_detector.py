#!/usr/bin/env python3
"""
Enhanced Drowsiness Detector with Improved Head Pose Detection
Fixes false positives and improves eye tracking accuracy
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
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.21
CLOSED_EYES_FRAME_THRESHOLD = 15
PERCLOS_WINDOW_SEC = 60
ALERT_COOLDOWN_SEC = 3.0

# Improved head pose detection thresholds
HEAD_TURN_THRESHOLD = 45  # Increased from 30 - more lenient
HEAD_DOWN_THRESHOLD = 35  # Increased from 20 - more lenient
MIRROR_CHECK_DURATION = 3.0  # Increased from 2.0 - longer grace period
BRIEF_GLANCE_DURATION = 2.0  # Increased from 1.0 - longer grace period

# Additional eye landmarks for better tracking
LEFT_EYE_DETAILED = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_DETAILED = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

class EnhancedHeadPoseDetector:
    """Enhanced head pose detection with better accuracy and fewer false positives"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.7
        )
        
        # Head pose tracking with smoothing
        self.head_turn_start_time = None
        self.head_down_start_time = None
        self.is_looking_away = False
        self.look_away_reason = "none"
        
        # Smoothing buffers for stable detection
        self.yaw_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=5)
        
        # Key points for head pose estimation
        self.nose_tip = 1
        self.chin = 152
        self.left_eye = 33
        self.right_eye = 263
        self.left_ear = 234
        self.right_ear = 454
        self.forehead = 10
        
        # Face orientation tracking
        self.face_orientation_history = deque(maxlen=10)
        self.stable_orientation_count = 0
    
    def calculate_head_pose_enhanced(self, landmarks, frame_shape):
        """Calculate head pose with multiple methods for better accuracy"""
        h, w, _ = frame_shape
        
        # Get key points
        nose = landmarks[self.nose_tip]
        chin = landmarks[self.chin]
        left_eye = landmarks[self.left_eye]
        right_eye = landmarks[self.right_eye]
        left_ear = landmarks[self.left_ear]
        right_ear = landmarks[self.right_ear]
        forehead = landmarks[self.forehead]
        
        # Convert to pixel coordinates
        nose_3d = np.array([nose.x * w, nose.y * h, nose.z * w])
        chin_3d = np.array([chin.x * w, chin.y * h, chin.z * w])
        left_eye_3d = np.array([left_eye.x * w, left_eye.y * h, left_eye.z * w])
        right_eye_3d = np.array([right_eye.x * w, right_eye.y * h, right_eye.z * w])
        left_ear_3d = np.array([left_ear.x * w, left_ear.y * h, left_ear.z * w])
        right_ear_3d = np.array([right_ear.x * w, right_ear.y * h, right_ear.z * w])
        forehead_3d = np.array([forehead.x * w, forehead.y * h, forehead.z * w])
        
        # Method 1: Eye-ear vector for yaw
        eye_center = (left_eye_3d + right_eye_3d) / 2
        ear_center = (left_ear_3d + right_ear_3d) / 2
        yaw_vector = ear_center - eye_center
        yaw_angle = np.arctan2(yaw_vector[1], yaw_vector[0]) * 180 / np.pi
        
        # Method 2: Nose-chin vector for pitch
        nose_chin_vector = chin_3d - nose_3d
        pitch_angle = np.arctan2(nose_chin_vector[2], nose_chin_vector[1]) * 180 / np.pi
        
        # Method 3: Forehead-nose vector for additional pitch validation
        forehead_nose_vector = nose_3d - forehead_3d
        pitch_angle_alt = np.arctan2(forehead_nose_vector[2], forehead_nose_vector[1]) * 180 / np.pi
        
        # Combine methods for better accuracy
        final_pitch = (pitch_angle + pitch_angle_alt) / 2
        
        return yaw_angle, final_pitch
    
    def is_stable_head_position(self, yaw, pitch):
        """Check if head position is stable to avoid false positives"""
        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        
        if len(self.yaw_buffer) < 3:
            return True
        
        # Check if recent values are stable (low variance)
        yaw_variance = np.var(list(self.yaw_buffer))
        pitch_variance = np.var(list(self.pitch_buffer))
        
        # Consider stable if variance is low
        return yaw_variance < 50 and pitch_variance < 50
    
    def update_head_pose_enhanced(self, landmarks, frame_shape):
        """Enhanced head pose update with better false positive prevention"""
        yaw, pitch = self.calculate_head_pose_enhanced(landmarks, frame_shape)
        current_time = time.time()
        
        # Add smoothing
        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        
        # Use smoothed values
        smoothed_yaw = np.mean(list(self.yaw_buffer))
        smoothed_pitch = np.mean(list(self.pitch_buffer))
        
        # Check stability before making decisions
        is_stable = self.is_stable_head_position(smoothed_yaw, smoothed_pitch)
        
        # Only trigger look-away if position is stable and exceeds thresholds
        if not is_stable:
            # Reset timers if position is not stable
            self.head_turn_start_time = None
            self.head_down_start_time = None
            self.is_looking_away = False
            self.look_away_reason = "unstable"
            return smoothed_yaw, smoothed_pitch, False, "unstable"
        
        # Check for head turns (mirror checks) - more lenient
        if abs(smoothed_yaw) > HEAD_TURN_THRESHOLD:
            if self.head_turn_start_time is None:
                self.head_turn_start_time = current_time
                self.look_away_reason = "mirror_check"
            elif current_time - self.head_turn_start_time > MIRROR_CHECK_DURATION:
                self.is_looking_away = True
        else:
            self.head_turn_start_time = None
        
        # Check for head down (looking at phone/instruments) - more lenient
        if smoothed_pitch > HEAD_DOWN_THRESHOLD:
            if self.head_down_start_time is None:
                self.head_down_start_time = current_time
                self.look_away_reason = "looking_down"
            elif current_time - self.head_down_start_time > BRIEF_GLANCE_DURATION:
                self.is_looking_away = True
        else:
            self.head_down_start_time = None
        
        # Reset if looking forward
        if abs(smoothed_yaw) < HEAD_TURN_THRESHOLD and smoothed_pitch < HEAD_DOWN_THRESHOLD:
            self.is_looking_away = False
            self.look_away_reason = "none"
            self.head_turn_start_time = None
            self.head_down_start_time = None
        
        return smoothed_yaw, smoothed_pitch, self.is_looking_away, self.look_away_reason

class EnhancedEyeTracker:
    """Enhanced eye tracking with multiple methods and better accuracy"""
    
    def __init__(self):
        self.eye_landmarks = LEFT_EYE_DETAILED + RIGHT_EYE_DETAILED
        self.blink_buffer = deque(maxlen=10)
        self.ear_history = deque(maxlen=30)
        
    def calculate_ear_enhanced(self, landmarks, frame_shape):
        """Calculate EAR using multiple methods for better accuracy"""
        h, w, _ = frame_shape
        
        # Method 1: Standard EAR calculation
        left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        
        left_ear = self.eye_aspect_ratio(left_eye_pts)
        right_ear = self.eye_aspect_ratio(right_eye_pts)
        standard_ear = (left_ear + right_ear) / 2.0
        
        # Method 2: Enhanced EAR with more landmarks
        left_eye_detailed = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_DETAILED]
        right_eye_detailed = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_DETAILED]
        
        enhanced_left_ear = self.calculate_enhanced_ear(left_eye_detailed)
        enhanced_right_ear = self.calculate_enhanced_ear(right_eye_detailed)
        enhanced_ear = (enhanced_left_ear + enhanced_right_ear) / 2.0
        
        # Method 3: Eye opening ratio
        opening_ratio = self.calculate_eye_opening_ratio(landmarks, frame_shape)
        
        # Combine methods with weights
        combined_ear = (standard_ear * 0.4 + enhanced_ear * 0.4 + opening_ratio * 0.2)
        
        # Add to history for smoothing
        self.ear_history.append(combined_ear)
        
        # Return smoothed EAR
        if len(self.ear_history) >= 5:
            return np.mean(list(self.ear_history)[-5:])
        else:
            return combined_ear
    
    def calculate_enhanced_ear(self, eye_landmarks):
        """Calculate EAR using more detailed landmarks"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Use multiple vertical and horizontal measurements
        vertical_distances = []
        horizontal_distances = []
        
        # Vertical measurements (top to bottom)
        for i in range(0, len(eye_landmarks), 2):
            if i + 1 < len(eye_landmarks):
                dist = self.euclidean_distance(eye_landmarks[i], eye_landmarks[i + 1])
                vertical_distances.append(dist)
        
        # Horizontal measurements (left to right)
        for i in range(1, len(eye_landmarks), 2):
            if i + 1 < len(eye_landmarks):
                dist = self.euclidean_distance(eye_landmarks[i], eye_landmarks[i + 1])
                horizontal_distances.append(dist)
        
        if not vertical_distances or not horizontal_distances:
            return 0.0
        
        avg_vertical = np.mean(vertical_distances)
        avg_horizontal = np.mean(horizontal_distances)
        
        return avg_vertical / avg_horizontal if avg_horizontal != 0 else 0.0
    
    def calculate_eye_opening_ratio(self, landmarks, frame_shape):
        """Calculate eye opening ratio using eye height vs width"""
        h, w, _ = frame_shape
        
        # Get eye corners and center points
        left_eye_corners = [33, 133]  # Left and right corners
        right_eye_corners = [362, 263]  # Left and right corners
        
        # Calculate eye width and height
        left_width = abs(landmarks[left_eye_corners[1]].x - landmarks[left_eye_corners[0]].x) * w
        right_width = abs(landmarks[right_eye_corners[1]].x - landmarks[right_eye_corners[0]].x) * w
        
        # Estimate eye height (simplified)
        left_height = abs(landmarks[160].y - landmarks[144].y) * h
        right_height = abs(landmarks[385].y - landmarks[380].y) * h
        
        # Calculate opening ratio
        left_ratio = left_height / left_width if left_width != 0 else 0
        right_ratio = right_height / right_width if right_width != 0 else 0
        
        return (left_ratio + right_ratio) / 2.0
    
    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def eye_aspect_ratio(self, eye_landmarks):
        """Standard EAR calculation"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        A = self.euclidean_distance(p2, p6)
        B = self.euclidean_distance(p3, p5)
        C = self.euclidean_distance(p1, p4)
        return (A + B) / (2.0 * C) if C != 0 else 0.0

class MLPersonalization:
    """Machine Learning model for personalized drowsiness detection"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.user_data = []
        self.model_file = "personalized_model.pkl"
        self.scaler_file = "scaler.pkl"
        
        # Load existing model if available
        self.load_model()
    
    def extract_features(self, ear, blink_count, perclos, yaw, pitch, is_looking_away, time_of_day):
        """Extract features for ML model"""
        # Time-based features
        hour = time_of_day.hour
        is_night = 1 if hour < 6 or hour > 22 else 0
        is_evening = 1 if 18 <= hour <= 22 else 0
        
        # Behavioral features
        head_movement = abs(yaw) + abs(pitch)
        attention_score = 1 if not is_looking_away else 0
        
        return [
            ear,
            blink_count,
            perclos,
            yaw,
            pitch,
            head_movement,
            attention_score,
            hour,
            is_night,
            is_evening
        ]
    
    def add_training_data(self, features, is_drowsy):
        """Add training data point"""
        self.user_data.append(features + [is_drowsy])
        
        # Retrain model every 50 new samples
        if len(self.user_data) >= 50:
            self.train_model()
    
    def train_model(self):
        """Train the personalized model"""
        if len(self.user_data) < 10:
            return
        
        data = np.array(self.user_data)
        X = data[:, :-1]  # Features
        y = data[:, -1]   # Labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model
        self.save_model()
        print(f"Model retrained with {len(self.user_data)} samples")
    
    def predict_drowsiness(self, features):
        """Predict drowsiness probability"""
        if not self.is_trained:
            return 0.5  # Default probability
        
        features_scaled = self.scaler.transform([features])
        probability = self.model.predict_proba(features_scaled)[0][1]
        return probability
    
    def save_model(self):
        """Save trained model"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load existing model"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("Loaded existing personalized model")
        except Exception as e:
            print(f"Error loading model: {e}")

class EnhancedDrowsinessDetector:
    """Enhanced drowsiness detector with improved accuracy"""
    
    def __init__(self):
        self.head_pose_detector = EnhancedHeadPoseDetector()
        self.eye_tracker = EnhancedEyeTracker()
        self.ml_personalization = MLPersonalization()
        
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
        
        # Session tracking
        self.session_start_time = time.time()
        self.driving_duration = 0
        
        # Simulation mode for testing
        self.simulation_mode = False
        self.simulated_drowsiness = False
        
    def play_beep(self):
        """Non-blocking beep sound"""
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
                pass
        
        threading.Thread(target=beep_thread, daemon=True).start()
    
    def toggle_simulation_mode(self):
        """Toggle simulation mode for testing"""
        self.simulation_mode = not self.simulation_mode
        print(f"Simulation mode: {'ON' if self.simulation_mode else 'OFF'}")
        if self.simulation_mode:
            print("Press 't' to simulate drowsiness, 'n' for normal state")
    
    def simulate_drowsiness(self, drowsy=True):
        """Simulate drowsiness for testing"""
        if self.simulation_mode:
            self.simulated_drowsiness = drowsy
            print(f"Simulated state: {'DROWSY' if drowsy else 'NORMAL'}")
    
    def update_metrics(self, ear, is_closed, t_now, yaw, pitch, is_looking_away):
        """Update drowsiness metrics with enhanced logic"""
        self.metrics_buffer.append((t_now, ear, is_closed, yaw, pitch, is_looking_away))
        
        # Enhanced blink detection
        if is_closed:
            self.frames_closed_consec += 1
        else:
            if self.last_state_closed and self.frames_closed_consec > 1:
                self.blink_count += 1
            self.frames_closed_consec = 0
        self.last_state_closed = is_closed
        
        # PERCLOS calculation
        closed_frames_in_window = sum(1 for _, _, closed, _, _, _ in self.metrics_buffer if closed)
        perclos = (closed_frames_in_window / len(self.metrics_buffer)) * 100 if self.metrics_buffer else 0
        
        # Enhanced alert logic
        long_closure = self.frames_closed_consec >= CLOSED_EYES_FRAME_THRESHOLD
        
        # Don't alert if driver is looking away (mirror check, etc.)
        if is_looking_away:
            long_closure = False
        
        # ML-based drowsiness prediction
        current_time = datetime.now()
        features = self.ml_personalization.extract_features(
            ear, self.blink_count, perclos, yaw, pitch, is_looking_away, current_time
        )
        
        ml_drowsiness_prob = self.ml_personalization.predict_drowsiness(features)
        
        # Combined alert logic
        should_alert = False
        alert_reason = "none"
        
        if self.simulation_mode and self.simulated_drowsiness:
            should_alert = True
            alert_reason = "simulation"
        elif long_closure and not is_looking_away:
            should_alert = True
            alert_reason = "eye_closure"
        elif ml_drowsiness_prob > 0.7 and not is_looking_away:
            should_alert = True
            alert_reason = "ml_prediction"
        
        # Trigger alert
        if should_alert and (t_now - self.last_alert_time > ALERT_COOLDOWN_SEC):
            self.play_beep()
            self.last_alert_time = t_now
            
            # Add to training data
            self.ml_personalization.add_training_data(features, True)
        
        # Log event
        self.event_log.append({
            "timestamp": current_time.isoformat(),
            "ear": ear,
            "is_closed": is_closed,
            "perclos": perclos,
            "blink_count": self.blink_count,
            "yaw_angle": yaw,
            "pitch_angle": pitch,
            "is_looking_away": is_looking_away,
            "look_away_reason": self.head_pose_detector.look_away_reason,
            "ml_drowsiness_prob": ml_drowsiness_prob,
            "alert_triggered": should_alert,
            "alert_reason": alert_reason,
            "simulation_mode": self.simulation_mode
        })
        
        return perclos, long_closure, ml_drowsiness_prob, should_alert, alert_reason
    
    def process_frame(self, frame):
        """Process a single frame for enhanced drowsiness detection"""
        if frame is None:
            return None, {}
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.head_pose_detector.face_mesh.process(rgb_frame)
        
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        ear = 0.0
        is_closed = False
        perclos = 0.0
        long_closure = False
        yaw = 0.0
        pitch = 0.0
        is_looking_away = False
        ml_drowsiness_prob = 0.0
        should_alert = False
        alert_reason = "none"
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Enhanced eye tracking
            ear = self.eye_tracker.calculate_ear_enhanced(landmarks, frame.shape)
            
            # Enhanced head pose detection
            yaw, pitch, is_looking_away, look_away_reason = self.head_pose_detector.update_head_pose_enhanced(landmarks, frame.shape)
            
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
                perclos, long_closure, ml_drowsiness_prob, should_alert, alert_reason = self.update_metrics(
                    ear, is_closed, time.time(), yaw, pitch, is_looking_away
                )
            
            # Enhanced UI overlay
            overlay_height = 180
            cv2.rectangle(display_frame, (5, 5), (w-5, overlay_height), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (w-5, overlay_height), (255, 255, 255), 2)
            
            # Status text
            status_text = "EYES OPEN" if not is_closed else "EYES CLOSED"
            color = (0, 255, 0) if not is_closed else (0, 0, 255)
            cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Head pose info with stability indicator
            pose_text = f"Head: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°"
            stability_color = (0, 255, 0) if look_away_reason == "none" else (0, 255, 255) if look_away_reason == "unstable" else (0, 0, 255)
            cv2.putText(display_frame, pose_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
            
            # Looking away indicator
            if is_looking_away:
                cv2.putText(display_frame, f"Looking: {look_away_reason}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif look_away_reason == "unstable":
                cv2.putText(display_frame, "Head position unstable", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # ML prediction
            ml_text = f"ML Drowsiness: {ml_drowsiness_prob:.2f}"
            ml_color = (0, 255, 0) if ml_drowsiness_prob < 0.5 else (0, 255, 255) if ml_drowsiness_prob < 0.7 else (0, 0, 255)
            cv2.putText(display_frame, ml_text, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ml_color, 2)
            
            # Enhanced metrics
            cv2.putText(display_frame, f"EAR: {ear:.3f} | Blinks: {self.blink_count} | PERCLOS: {perclos:.1f}%", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Threshold info
            cv2.putText(display_frame, f"Threshold: {self.current_ear_threshold:.3f}", (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Alert
            if should_alert:
                cv2.putText(display_frame, f"ALERT: {alert_reason.upper()}!", (w//2-100, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Simulation mode indicator
            if self.simulation_mode:
                cv2.putText(display_frame, "SIMULATION MODE", (w-200, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Draw enhanced eye landmarks
            left_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            
            for pts in [left_eye_pts, right_eye_pts]:
                cv2.polylines(display_frame, [np.array(pts)], True, (0, 255, 255), 2)
                for (x, y) in pts:
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 255), -1)
        else:
            cv2.putText(display_frame, "No face detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return display_frame, {
            'ear': ear,
            'is_closed': is_closed,
            'perclos': perclos,
            'blink_count': self.blink_count,
            'long_closure': long_closure,
            'yaw': yaw,
            'pitch': pitch,
            'is_looking_away': is_looking_away,
            'ml_drowsiness_prob': ml_drowsiness_prob,
            'should_alert': should_alert,
            'alert_reason': alert_reason
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
            filename = f"enhanced_drowsiness_log_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return f"Data saved to {filename}"
        else:
            return "No data to save"

def main():
    """Main application loop with enhanced features"""
    print("Enhanced Drowsiness Detector - Improved Accuracy")
    print("Press 'q' to quit")
    print("Press 'o' to calibrate open eyes")
    print("Press 'c' to calibrate closed eyes")
    print("Press 's' to stop calibration")
    print("Press 'd' to save CSV data")
    print("Press 'm' to toggle simulation mode")
    print("Press 't' to simulate drowsiness (in simulation mode)")
    print("Press 'n' to simulate normal state (in simulation mode)")
    print("------------------------------------------")
    
    detector = EnhancedDrowsinessDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Mobile-optimized settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow("Enhanced Drowsiness Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enhanced Drowsiness Detector", 1000, 700)
    
    detector.is_running = True
    
    while detector.is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, metrics = detector.process_frame(frame)
        if processed_frame is not None:
            cv2.imshow("Enhanced Drowsiness Detector", processed_frame)
        
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
        elif key == ord('m'):
            detector.toggle_simulation_mode()
        elif key == ord('t'):
            detector.simulate_drowsiness(True)
        elif key == ord('n'):
            detector.simulate_drowsiness(False)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
