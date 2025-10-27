# Eye Blink & Drowsiness Detector
# Professional implementation with live tracking

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional
import threading

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from PIL import Image
import io
import wave

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detector",
    page_icon="👁️",
    layout="wide"
)

# Professional CSS styling with better readability
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #2563eb;
    --secondary-color: #64748b;
    --success-color: #059669;
    --warning-color: #d97706;
    --danger-color: #dc2626;
    --background-color: #ffffff;
    --border-color: #e2e8f0;
    --text-color: #1e293b;
    --light-bg: #f8fafc;
    --dark-text: #0f172a;
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* Better metric badges with high contrast */
.metric-badge {
    padding: 12px 20px;
    border-radius: 8px;
    display: inline-block;
    margin-right: 16px;
    margin-bottom: 12px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    border: 2px solid;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.ok {
    background: #22c55e;
    color: white;
    border-color: #16a34a;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

.warn {
    background: #f59e0b;
    color: white;
    border-color: #d97706;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

.alert {
    background: #ef4444;
    color: white;
    border-color: #dc2626;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

/* Better text contrast - WHITE TEXT */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: white !important;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

p, div, span, .stMarkdown, .stText, .stSelectbox label, .stSlider label {
    font-family: 'Inter', sans-serif;
    color: white !important;
    font-size: 16px;
    line-height: 1.6;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

/* Force all Streamlit text to be white */
.stApp {
    background-color: #1a1a1a !important;
}

.stApp > div {
    background-color: #1a1a1a !important;
}

.stMarkdown {
    color: white !important;
}

.stText {
    color: white !important;
}

.stSelectbox label {
    color: white !important;
}

.stSlider label {
    color: white !important;
}

.stCheckbox label {
    color: white !important;
}

.stRadio label {
    color: white !important;
}

/* Better button styling - WHITE TEXT */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    border: 2px solid white;
    background: #333333;
    color: white !important;
    padding: 12px 20px;
    font-size: 16px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

.stButton > button:hover {
    background: #555555;
    border-color: #ffffff;
    color: white !important;
}

/* Status indicators */
.status-box {
    padding: 16px 24px;
    border-radius: 12px;
    margin: 16px 0;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    text-align: center;
    border: 3px solid;
}

.status-success {
    background: #22c55e;
    color: white;
    border-color: #16a34a;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

.status-warning {
    background: #f59e0b;
    color: white;
    border-color: #d97706;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

.status-danger {
    background: #ef4444;
    color: white;
    border-color: #dc2626;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}

/* Help tooltips */
.help-icon {
    display: inline-block;
    width: 20px;
    height: 20px;
    background: var(--primary-color);
    color: white;
    border-radius: 50%;
    text-align: center;
    font-size: 14px;
    font-weight: bold;
    margin-left: 8px;
    cursor: help;
    vertical-align: middle;
}

/* Video container */
.video-container {
    border: 3px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    background: var(--light-bg);
    margin: 16px 0;
}

/* Settings panel */
.settings-panel {
    background: var(--light-bg);
    padding: 20px;
    border-radius: 12px;
    border: 2px solid var(--border-color);
    margin: 16px 0;
}

/* Better slider styling */
.stSlider > div > div > div > div {
    background: var(--primary-color);
}

.stSlider > div > div > div > div > div {
    background: var(--primary-color);
}
</style>
""", unsafe_allow_html=True)

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
            if long_closure or perclos >= st.session_state.get("perclos_alert_threshold", 40.0):
                if (tnow - self.last_alert_ts) > self.alert_cooldown_sec and st.session_state.get("enable_audio", True):
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
        # Background rectangle for text with better contrast
        cv2.rectangle(frame, (5, 5), (450, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 180), (255, 255, 255), 3)
        
        # Status color
        status_color = (0, 255, 0) if not is_closed else (0, 0, 255)
        alert_color = (0, 0, 255) if long_closure else (255, 255, 0)
        
        # Status text with larger font
        status_text = "EYES OPEN" if not is_closed else "EYES CLOSED"
        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # EAR and threshold with better contrast
        cv2.putText(frame, f"EAR: {ear:.3f} (threshold: {self.tau:.3f})", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blink count
        cv2.putText(frame, f"Blinks: {self.blink_count}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # PERCLOS
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Alert message
        if long_closure:
            cv2.putText(frame, "ALERT: DROWSINESS DETECTED!", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        
        # Draw eye landmarks with better visibility
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

# Initialize session state
if "detector" not in st.session_state:
    st.session_state.detector = DrowsinessDetector()

if "perclos_alert_threshold" not in st.session_state:
    st.session_state.perclos_alert_threshold = 40.0

if "enable_audio" not in st.session_state:
    st.session_state.enable_audio = True

if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False

# Main UI
st.title("Eye Blink & Drowsiness Detector")
st.markdown("**Real-time eye tracking and drowsiness detection using MediaPipe FaceMesh**")

# Instructions
with st.expander("How to use this application"):
    st.markdown("""
    ### Quick Start Guide:
    1. **Position yourself**: Sit in front of your camera with good lighting on your face
    2. **Start tracking**: Click "Start Live Tracking" to begin continuous monitoring
    3. **Calibrate (optional)**: 
       - Choose "auto" calibration mode
       - Click "Record Open Eyes" and keep your eyes open for 3-5 seconds
       - Click "Record Closed Eyes" and close your eyes for 2-3 seconds
    4. **Monitor**: Watch the live video feed and metrics
       - **Green "EYES OPEN"** = You're alert
       - **Red "EYES CLOSED"** = Eyes detected as closed
       - **Colored circles** = Eye landmarks being tracked
    5. **Alerts**: Audio alerts will play automatically when drowsiness is detected
    6. **Download data**: Use the CSV download button to save your session data
    """)

# Live tracking controls
st.subheader("Live Tracking Controls")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Start Live Tracking", key="start_tracking", type="primary"):
        st.session_state.tracking_active = True
        st.success("Live tracking started!")

with col2:
    if st.button("Stop Tracking", key="stop_tracking"):
        st.session_state.tracking_active = False
        st.info("Live tracking stopped.")

with col3:
    if st.button("Reset All Data", key="reset_data"):
        detector = st.session_state.detector
        detector.event_log.clear()
        detector.blink_count = 0
        detector.metrics.buffer.clear()
        st.success("All data reset!")

# Calibration section
st.subheader("Calibration Settings")

cal_col1, cal_col2, cal_col3 = st.columns([1, 1, 1])

with cal_col1:
    rec_open = st.button("Record Open Eyes (3s)", key="rec_open")
    st.markdown('<span class="help-icon" title="Click this button and keep your eyes open for 3 seconds to calibrate the system for your open eye state">?</span>', unsafe_allow_html=True)

with cal_col2:
    rec_closed = st.button("Record Closed Eyes (2s)", key="rec_closed")
    st.markdown('<span class="help-icon" title="Click this button and close your eyes for 2 seconds to calibrate the system for your closed eye state">?</span>', unsafe_allow_html=True)

with cal_col3:
    clear_cal = st.button("Clear Calibration", key="clear_cal")
    st.markdown('<span class="help-icon" title="Clear all calibration data and reset to default threshold">?</span>', unsafe_allow_html=True)

# Calibration settings
settings_col1, settings_col2 = st.columns([1, 1])
with settings_col1:
    mode = st.radio("Threshold Mode", ["auto", "manual"], index=0, horizontal=True)
    st.markdown('<span class="help-icon" title="Auto mode uses your calibration data, Manual mode uses a fixed threshold">?</span>', unsafe_allow_html=True)

with settings_col2:
    manual_tau = st.slider("Manual EAR Threshold", 0.10, 0.40, 0.21, 0.005)
    st.markdown('<span class="help-icon" title="Eye Aspect Ratio threshold - lower values mean stricter detection">?</span>', unsafe_allow_html=True)

# Apply calibration settings
detector = st.session_state.detector
detector.threshold_mode = mode
detector.manual_tau = manual_tau

# Calibration actions
def collect_samples(duration_sec: float, label: str):
    st.info(f"Collecting {label} EAR samples for {duration_sec:.0f}s...")
    start = time.time()
    samples = []
    while time.time() - start < duration_sec:
        samples.append(detector.live_values.get("ear", 0.0))
        time.sleep(0.05)
    arr = np.array(samples, dtype=float)
    arr = arr[(arr > 0) & (arr < 1)]
    if len(arr) < 5:
        st.warning("Not enough valid samples collected. Try again with your eyes clearly open/closed.")
        return
    mu = float(np.mean(arr))
    if label == "open":
        detector.open_samples.extend(arr.tolist())
    else:
        detector.closed_samples.extend(arr.tolist())
    detector._compute_threshold()
    st.success(f"Collected {len(arr)} samples. {label.capitalize()} EAR mean={mu:.3f}. New tau={detector.tau:.3f}")

if rec_open:
    collect_samples(3.0, "open")
if rec_closed:
    collect_samples(2.0, "closed")
if clear_cal:
    detector.open_samples.clear()
    detector.closed_samples.clear()
    st.success("Calibration cleared.")

# Live video feed
st.subheader("Live Video Feed")
st.markdown("Position your face in the camera view below to track eye movement and detect drowsiness.")

# Camera input for live tracking
if st.session_state.tracking_active:
    camera_input = st.camera_input("Live Camera Feed", key="live_camera")
    
    if camera_input is not None:
        # Convert to OpenCV format
        image = Image.open(camera_input)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Convert back to RGB for display
        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame
        st.image(display_frame, channels="RGB", use_column_width=True)
        
        # Status indicators with better styling
        if detector.live_values["long_closure"]:
            st.markdown('<div class="status-box status-danger">DROWSINESS ALERT! Eyes closed for too long</div>', unsafe_allow_html=True)
        elif detector.live_values["is_closed"]:
            st.markdown('<div class="status-box status-warning">Eyes detected as closed</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-success">Eyes detected as open - Alert and awake</div>', unsafe_allow_html=True)
        
        # Audio alert
        if detector.should_play_beep and st.session_state.enable_audio:
            st.markdown("**ALERT!** Audio alert triggered:")
            # Generate beep sound
            sr = 24000
            dur = 0.3
            t = np.linspace(0, dur, int(sr*dur), endpoint=False)
            sig = (0.3*np.sin(2*np.pi*440*t)).astype(np.float32)
            
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                data16 = (sig * 32767).astype(np.int16).tobytes()
                w.writeframes(data16)
            st.audio(buf.getvalue(), format="audio/wav")
else:
    st.info("Click 'Start Live Tracking' to begin monitoring.")

# Live metrics with tooltips
st.subheader("Live Metrics")
if detector.live_values["ear"] > 0:
    live = detector.live_values
    ear = live["ear"]
    perclos = live["perclos"]
    blink_rate = live["blink_rate"]
    fps = live["fps"]
    is_closed = live["is_closed"]
    long_closure = live["long_closure"]

    def badge(text, cls, tooltip=""):
        if tooltip:
            st.markdown(f'<span class="metric-badge {cls}" title="{tooltip}">{text}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="metric-badge {cls}">{text}</span>', unsafe_allow_html=True)

    badge(f"EAR: {ear:.3f}", "ok" if not is_closed else "warn", "Eye Aspect Ratio - measures eye openness (higher = more open)")
    badge(f"PERCLOS: {perclos:.1f}%", "ok" if perclos < 30 else ("warn" if perclos < 50 else "alert"), "Percentage of Eye Closure - drowsiness indicator")
    badge(f"Blink Rate: {blink_rate:.1f}/min", "ok", "Number of blinks per minute - normal range is 15-20")
    badge(f"FPS: {fps:.1f}", "ok", "Frames per second - processing speed")
    if long_closure:
        badge("ALERT: Long Closure", "alert", "Eyes closed for more than 1 second - drowsiness detected")
    
    # Download CSV
    if st.checkbox("Show/Download CSV log", value=False):
        df = pd.DataFrame(detector.event_log)
        st.dataframe(df.tail(300))
        if len(df) > 0:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="drowsy_log.csv", mime="text/csv")
else:
    st.info("Start live tracking to view metrics.")

# Bottom settings panel
st.markdown("---")
st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
st.subheader("Settings")

settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 1])

with settings_col1:
    st.session_state.enable_audio = st.checkbox("Audio Alerts", value=True)
    st.markdown('<span class="help-icon" title="Play beep sounds when drowsiness is detected">?</span>', unsafe_allow_html=True)
    
    alert_threshold = st.slider("PERCLOS Alert Threshold (%)", 10, 90, 40, 1)
    st.markdown('<span class="help-icon" title="Alert when PERCLOS exceeds this percentage">?</span>', unsafe_allow_html=True)
    st.session_state.perclos_alert_threshold = alert_threshold

with settings_col2:
    window_sec = st.slider("PERCLOS Window (sec)", 15, 180, DEFAULT_WINDOW_SEC, 5)
    st.markdown('<span class="help-icon" title="Time window for calculating PERCLOS percentage">?</span>', unsafe_allow_html=True)
    detector.metrics.window_sec = window_sec
    
    want_logging = st.checkbox("Enable CSV Logging", value=True)
    st.markdown('<span class="help-icon" title="Save all tracking data to CSV file">?</span>', unsafe_allow_html=True)

with settings_col3:
    st.markdown("**Privacy Notice**")
    st.caption("Runs entirely in-browser/server memory. No video is stored or uploaded beyond the app session.")
    
    st.markdown("**Tips for Best Results:**")
    st.caption("• Good lighting on your face\n• Center your face in camera\n• Stay 1-2 feet from camera\n• Avoid heavy reflections on glasses")

st.markdown('</div>', unsafe_allow_html=True)

# Tips
with st.expander("Detailed Help & Tips"):
    st.markdown("""
    ### What Each Metric Means:
    
    **EAR (Eye Aspect Ratio)**: Measures how open your eyes are
    - Higher values (0.25-0.35) = Eyes open
    - Lower values (0.10-0.20) = Eyes closed
    - Used to detect blinks and drowsiness
    
    **PERCLOS (Percentage of Eye Closure)**: Measures drowsiness over time
    - 0-20% = Alert and awake
    - 20-40% = Slightly drowsy
    - 40%+ = Very drowsy (alerts triggered)
    
    **Blink Rate**: Number of blinks per minute
    - Normal range: 15-20 blinks/minute
    - Too low (<10) = Possible drowsiness
    - Too high (>30) = Possible stress or fatigue
    
    **FPS (Frames Per Second)**: Processing speed
    - Higher FPS = More accurate detection
    - Should be 15+ for good performance
    
    ### Calibration Tips:
    - Use "auto" mode for best results
    - Record open eyes: Look straight ahead, eyes wide open
    - Record closed eyes: Gently close eyes completely
    - Clear calibration if results seem wrong
    
    ### Troubleshooting:
    - **No face detected**: Check lighting and camera position
    - **False alerts**: Recalibrate or adjust threshold
    - **Poor tracking**: Ensure good lighting and stable position
    """)