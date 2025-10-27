# Mobile Drowsiness Detector - Deployment Guide

## Overview
This guide covers deploying the drowsiness detector as a mobile app for car mount usage, with special focus on low-light conditions common during night driving.

## Mobile Deployment Options

### Option 1: Android APK (Recommended for Car Mount)
**Using Buildozer (Python to Android)**

1. **Install Buildozer:**
```bash
pip install buildozer
```

2. **Create buildozer.spec file:**
```ini
[app]
title = DrivingDrowsyDetector
package.name = drivingdrowsydetector
package.domain = com.ka2507.drowsydetector
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0
requirements = python3,kivy,opencv-python,numpy,pandas,mediapipe,pillow

[buildozer]
log_level = 2
warn_on_root = 1

[android]
permissions = CAMERA,RECORD_AUDIO,WRITE_EXTERNAL_STORAGE
```

3. **Build APK:**
```bash
buildozer android debug
```

### Option 2: iOS App (Using Kivy-iOS)
**For iPhone/iPad deployment:**

1. **Install Kivy-iOS:**
```bash
pip install kivy-ios
```

2. **Create iOS project:**
```bash
toolchain create DrivingDrowsyDetector mobile_drowsiness_detector.py
```

3. **Build and deploy:**
```bash
toolchain build
toolchain deploy
```

### Option 3: Web App (PWA)
**For cross-platform deployment:**

1. **Install Streamlit:**
```bash
pip install streamlit streamlit-webrtc
```

2. **Run web version:**
```bash
streamlit run app.py
```

3. **Access via mobile browser** - works on any device with camera

## Low-Light Optimization Features

### Automatic Low-Light Detection
- **Brightness Monitoring:** Continuously monitors average frame brightness
- **Adaptive Enhancement:** Automatically applies CLAHE and gamma correction
- **Visual Indicators:** Shows "LOW LIGHT MODE" when enhancement is active

### Camera Settings for Night Driving
- **Higher ISO:** Automatically adjusts camera sensitivity
- **Wider Aperture:** Uses maximum available aperture
- **Exposure Compensation:** Increases exposure for dark environments

### UI Optimizations for Car Mount
- **Large Text:** High contrast, readable fonts
- **Minimal Interface:** Clean, distraction-free design
- **Touch-Friendly:** Large buttons for easy operation while driving
- **Voice Commands:** Optional voice control integration

## Installation Instructions

### For Android Users:
1. Download the APK file
2. Enable "Install from Unknown Sources" in Android settings
3. Install the APK
4. Grant camera permissions when prompted
5. Mount phone in car and launch app

### For iOS Users:
1. Install via TestFlight or App Store (if published)
2. Grant camera permissions
3. Mount iPhone/iPad in car
4. Launch app and calibrate

### For Web Users:
1. Open mobile browser
2. Navigate to the web app URL
3. Grant camera permissions
4. Use in landscape mode for best car mount experience

## Usage Instructions

### Initial Setup:
1. **Mount Device:** Secure phone/tablet in car mount
2. **Position Camera:** Ensure camera can see driver's face clearly
3. **Calibrate:** Run calibration for open and closed eyes
4. **Test:** Verify detection works in your driving position

### Calibration Process:
1. **Open Eyes Calibration:**
   - Press "Calibrate Open Eyes"
   - Keep eyes wide open for 5-10 seconds
   - Press "Stop Calibration"

2. **Closed Eyes Calibration:**
   - Press "Calibrate Closed Eyes"
   - Gently close eyes (like when drowsy) for 5-10 seconds
   - Press "Stop Calibration"

### Driving Usage:
1. **Start Detection:** App automatically begins monitoring
2. **Monitor Status:** Watch for "EYES CLOSED" warnings
3. **Respond to Alerts:** Take breaks when drowsiness detected
4. **Save Data:** Export session data for analysis

## Low-Light Performance Tips

### Camera Positioning:
- **Avoid Backlighting:** Don't position camera facing bright lights
- **Use Interior Lights:** Gentle interior lighting improves detection
- **Clean Camera:** Keep camera lens clean for better low-light performance

### App Settings:
- **Enable Low-Light Mode:** Automatically activated when needed
- **Adjust Sensitivity:** Fine-tune detection thresholds
- **Battery Optimization:** Reduce processing for longer battery life

### Environmental Considerations:
- **Dashboard Lights:** Use dim dashboard lighting
- **Phone Brightness:** Set phone to auto-brightness
- **Reflections:** Avoid reflective surfaces near camera

## Troubleshooting

### Camera Issues:
- **No Camera Access:** Check app permissions
- **Poor Detection:** Recalibrate in current lighting conditions
- **Low FPS:** Close other apps, reduce video quality

### Low-Light Problems:
- **Poor Enhancement:** Ensure camera is clean
- **False Positives:** Recalibrate with current lighting
- **Battery Drain:** Reduce processing frequency

### Mobile-Specific Issues:
- **App Crashes:** Restart app, check available memory
- **Touch Issues:** Use landscape mode, larger buttons
- **Battery Life:** Use power-saving mode, external charger

## Safety Considerations

### While Driving:
- **Minimal Interaction:** Set up before driving
- **Voice Commands:** Use hands-free operation when possible
- **Pull Over:** Stop safely if alerts require attention

### Legal Compliance:
- **Local Laws:** Check regulations about phone usage while driving
- **Insurance:** Verify coverage with mobile device usage
- **Privacy:** Understand data collection and storage policies

## Performance Optimization

### Battery Life:
- **Power Saving:** Reduce processing frequency when not needed
- **External Power:** Use car charger for long trips
- **Background Apps:** Close unnecessary applications

### Processing Speed:
- **Lower Resolution:** Use 480p for better performance
- **Frame Skipping:** Process every other frame if needed
- **Hardware Acceleration:** Use GPU when available

## Data Export and Analysis

### CSV Export:
- **Session Data:** Timestamp, EAR, blink count, PERCLOS
- **Low-Light Events:** Records when low-light mode was active
- **Alert History:** Track drowsiness events over time

### Analysis Tools:
- **Excel/Google Sheets:** Import CSV for basic analysis
- **Python Scripts:** Custom analysis with pandas
- **Dashboard:** Real-time monitoring during long trips

## Future Enhancements

### Planned Features:
- **Voice Alerts:** Audio warnings for drowsiness
- **Cloud Sync:** Upload data for analysis
- **Machine Learning:** Personalized detection models
- **Integration:** Connect with car systems

### Hardware Integration:
- **OBD-II:** Connect with car diagnostics
- **Smart Mirrors:** Integration with car displays
- **Wearables:** Connect with smartwatches/fitness trackers
