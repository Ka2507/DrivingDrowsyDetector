# Eye Blink & Drowsiness Detector

A real-time drowsiness detection system using MediaPipe FaceMesh and OpenCV. This application monitors eye movement, calculates Eye Aspect Ratio (EAR), and alerts when drowsiness is detected.

## Features

- Real-time eye tracking using MediaPipe FaceMesh
- Eye Aspect Ratio (EAR) calculation for blink detection
- PERCLOS (Percentage of Eye Closure) monitoring
- Continuous calibration system
- Audio alerts for drowsiness detection
- CSV data logging for analysis
- Professional OpenCV-based interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ka2507/drowsy-detector-mediapipe.git
cd drowsy-detector-mediapipe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### OpenCV Version (Recommended)

Run the main OpenCV application:
```bash
python drowsiness_detector.py
```

**Controls:**
- `q` - Quit application
- `o` - Start calibrating open eyes (keep eyes open)
- `c` - Start calibrating closed eyes (close eyes)
- `s` - Stop calibration
- `d` - Download CSV data

### Streamlit Version (Alternative)

Run the Streamlit web interface:
```bash
streamlit run app.py
```

## How It Works

### Eye Aspect Ratio (EAR)
The system calculates EAR using 6 landmark points around each eye:
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

### Drowsiness Detection
- **Blink Detection**: Monitors transitions from open to closed eyes
- **PERCLOS**: Calculates percentage of eye closure over time
- **Long Closure**: Detects sustained eye closure (>1 second)
- **Audio Alerts**: Plays beep when drowsiness is detected

### Calibration
The system supports automatic calibration:
1. Press `o` and keep eyes open for continuous calibration
2. Press `c` and close eyes for continuous calibration
3. Press `s` to stop calibration
4. The system automatically calculates optimal thresholds

## Technical Details

### Dependencies
- OpenCV 4.10.0+
- MediaPipe 0.10.14+
- NumPy 1.26.4+
- Pandas 2.2.3+
- Streamlit 1.39.0+ (optional)

### Performance
- Real-time processing at 15-30 FPS
- Low latency eye tracking
- Efficient memory usage
- Cross-platform compatibility

## Data Output

The system logs the following metrics:
- Timestamp
- Eye Aspect Ratio (EAR)
- Eye closure status
- PERCLOS percentage
- Blink count and rate
- Processing FPS
- Long closure alerts

Data is automatically saved to CSV files with timestamps.

## Configuration

### Thresholds
- **EAR Threshold**: Default 0.21 (auto-calibrated)
- **PERCLOS Alert**: Default 40%
- **Long Closure**: Default 1.0 second
- **Alert Cooldown**: Default 3.0 seconds

### Camera Settings
- Resolution: 1280x720 (configurable)
- Frame Rate: 30 FPS
- Auto-exposure and white balance

## Troubleshooting

### Common Issues

**No face detected:**
- Ensure good lighting on your face
- Position face centered in camera view
- Check camera permissions

**False alerts:**
- Recalibrate using `o` and `c` keys
- Adjust manual threshold if needed
- Ensure stable lighting conditions

**Poor tracking:**
- Maintain 1-2 feet distance from camera
- Avoid heavy reflections on glasses
- Keep head relatively still

### Performance Optimization

**For better FPS:**
- Close other applications
- Use dedicated graphics if available
- Reduce camera resolution if needed

**For better accuracy:**
- Use good lighting conditions
- Calibrate regularly
- Maintain consistent camera position

## Privacy & Security

- All processing happens locally
- No video data is stored or transmitted
- Only metrics are logged to CSV files
- No internet connection required

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For questions or issues, please open an issue on the GitHub repository.