# Advanced Drowsiness Detector - Testing Guide

## 🧪 Testing Without Being in a Car

### **Simulation Mode Testing**
The advanced detector includes a simulation mode perfect for testing without being in a car:

1. **Start the application:**
   ```bash
   python advanced_drowsiness_detector.py
   ```

2. **Enable Simulation Mode:**
   - Press `m` to toggle simulation mode
   - You'll see "SIMULATION MODE" indicator on screen

3. **Test Drowsiness Detection:**
   - Press `t` to simulate drowsiness
   - Press `n` to simulate normal state
   - Watch how the ML model learns your patterns

### **Manual Testing Scenarios**

#### **Scenario 1: Mirror Check Simulation**
1. **Look left** (as if checking left mirror) for 2-3 seconds
2. **Look right** (as if checking right mirror) for 2-3 seconds
3. **Observe:** System should NOT alert during mirror checks
4. **Expected:** "Looking: mirror_check" appears, no drowsiness alert

#### **Scenario 2: Brief Glances**
1. **Look down** briefly (as if checking speedometer)
2. **Look up** at rearview mirror
3. **Observe:** System should NOT alert for brief glances
4. **Expected:** "Looking: looking_down" appears, no alert

#### **Scenario 3: Actual Drowsiness**
1. **Close eyes** for 3+ seconds while looking forward
2. **Observe:** System should alert with beep
3. **Expected:** "ALERT: eye_closure!" appears

#### **Scenario 4: ML Learning**
1. **Use simulation mode** (`m` then `t` for drowsy, `n` for normal)
2. **Watch ML probability** change over time
3. **Expected:** Model learns your patterns and becomes more accurate

## 🎯 Key Features to Test

### **1. Head Pose Detection**
- **Yaw Angle:** Left/right head turns
- **Pitch Angle:** Up/down head movement
- **Mirror Check Detection:** Prevents false alerts
- **Looking Down Detection:** Phone/instrument checks

### **2. Smart Error Handling**
- **Mirror Checks:** No alerts when looking at mirrors
- **Brief Glances:** No alerts for quick looks
- **Context Awareness:** Different thresholds for different actions

### **3. Machine Learning Personalization**
- **Pattern Learning:** Model learns your normal behavior
- **Adaptive Thresholds:** Adjusts to your blinking patterns
- **Time-based Learning:** Considers time of day
- **Behavioral Analysis:** Learns your driving habits

### **4. Enhanced Metrics**
- **ML Drowsiness Probability:** 0.0 to 1.0 scale
- **Head Pose Angles:** Real-time yaw and pitch
- **Look Away Detection:** Reason for looking away
- **Alert Reasoning:** Why alert was triggered

## 📊 Testing Checklist

### **Basic Functionality**
- [ ] Camera opens and shows video feed
- [ ] Face detection works
- [ ] Eye tracking landmarks visible
- [ ] EAR calculation updates in real-time

### **Head Pose Detection**
- [ ] Yaw angle changes when turning head left/right
- [ ] Pitch angle changes when looking up/down
- [ ] Mirror check detection works
- [ ] Looking down detection works

### **Smart Alerts**
- [ ] No alert during mirror checks
- [ ] No alert during brief glances
- [ ] Alert triggers for prolonged eye closure
- [ ] Alert triggers for high ML drowsiness probability

### **Machine Learning**
- [ ] ML model loads existing data (if available)
- [ ] ML probability updates in real-time
- [ ] Model learns from new data
- [ ] Personalized thresholds improve over time

### **Simulation Mode**
- [ ] Simulation mode toggles on/off
- [ ] Drowsiness simulation works
- [ ] Normal state simulation works
- [ ] Simulation mode indicator visible

## 🔧 Troubleshooting

### **Camera Issues**
- **No video feed:** Check camera permissions
- **Poor detection:** Ensure good lighting
- **Lag:** Close other applications

### **Head Pose Issues**
- **Inaccurate angles:** Ensure face is clearly visible
- **False mirror detection:** Adjust `HEAD_TURN_THRESHOLD` if needed
- **False looking down:** Adjust `HEAD_DOWN_THRESHOLD` if needed

### **ML Model Issues**
- **No learning:** Ensure enough training data (50+ samples)
- **Poor predictions:** Check feature extraction
- **Model not saving:** Check file permissions

## 📈 Performance Metrics

### **Expected Performance**
- **Frame Rate:** 25-30 FPS
- **Detection Accuracy:** 90%+ for eye closure
- **False Positive Rate:** <5% with head pose filtering
- **ML Learning:** Improves accuracy by 10-20% over time

### **Resource Usage**
- **CPU:** Moderate (MediaPipe + ML)
- **Memory:** ~200MB
- **Storage:** ~50MB for model files

## 🚗 Real-World Testing

### **When You Get in a Car**
1. **Mount phone** securely in car mount
2. **Position camera** to see your face clearly
3. **Calibrate** open and closed eyes
4. **Test mirror checks** while parked
5. **Start driving** and observe real-world performance

### **Safety Notes**
- **Never test while driving** - use simulation mode
- **Pull over** if alerts require attention
- **Focus on driving** - let the system work automatically
- **Take breaks** when system suggests

## 📱 Mobile Testing

### **Android Testing**
1. **Build APK** using buildozer
2. **Install on phone**
3. **Test in car mount**
4. **Verify all features work**

### **iOS Testing**
1. **Build iOS app** using kivy-ios
2. **Install via TestFlight**
3. **Test in car mount**
4. **Verify performance**

## 🎯 Success Criteria

### **The system is working correctly if:**
- ✅ **No false alerts** during mirror checks
- ✅ **No false alerts** during brief glances
- ✅ **Accurate alerts** for actual drowsiness
- ✅ **ML model learns** and improves over time
- ✅ **Smooth performance** without lag
- ✅ **Clear visual feedback** for all states

### **The system needs adjustment if:**
- ❌ **Too many false alerts** during normal driving
- ❌ **Missed drowsiness** events
- ❌ **Poor head pose detection**
- ❌ **ML model not learning**
- ❌ **Performance issues** or lag
- ❌ **Unclear visual feedback**
