#!/usr/bin/env python3
"""
Simple camera test script to check camera permissions
"""

import cv2
import sys

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for i in range(3):
        print(f"Trying camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} is working! Resolution: {frame.shape}")
                cap.release()
                return i
            else:
                print(f"Camera {i} opened but couldn't read frame")
        else:
            print(f"Camera {i} failed to open")
        cap.release()
    
    print("No working camera found!")
    return None

if __name__ == "__main__":
    camera_index = test_camera()
    if camera_index is not None:
        print(f"\nCamera {camera_index} is ready to use!")
        print("You can now run: python drowsiness_detector.py")
    else:
        print("\nPlease check camera permissions:")
        print("1. Go to System Preferences > Security & Privacy > Privacy > Camera")
        print("2. Make sure Terminal is checked")
        print("3. Restart Terminal and try again")
