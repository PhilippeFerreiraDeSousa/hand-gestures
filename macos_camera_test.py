#!/usr/bin/env python3
"""
macOS Camera Test Script
This script tries multiple approaches to access the camera on macOS
"""

import cv2
import time
import platform
import os
import sys

def test_camera():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Check if we're running in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Running in virtual environment: {in_venv}")
    
    # Print OpenCV build information for debugging
    print("\nOpenCV Build Information:")
    print(cv2.getBuildInformation())
    
    # Try different camera APIs
    backends = [
        # Regular VideoCapture with index
        (cv2.CAP_ANY, 0, "Default backend, camera 0"),
        
        # macOS specific backends 
        (cv2.CAP_AVFOUNDATION, 0, "AVFoundation, camera 0"),  # Built-in camera
        (cv2.CAP_AVFOUNDATION, 1, "AVFoundation, camera 1"),  # External camera
        
        # String-based URLs for AVFoundation
        (cv2.CAP_ANY, "avfoundation:0", "String URL for built-in camera"),
        (cv2.CAP_ANY, "avfoundation:1", "String URL for external camera"),
        
        # QuickTime legacy support
        (cv2.CAP_QT, 0, "QuickTime, camera 0")
    ]
    
    success = False
    
    for api, camera_id, description in backends:
        try:
            print(f"\nTrying: {description}")
            if isinstance(camera_id, int):
                cap = cv2.VideoCapture(camera_id, api)
            else:
                cap = cv2.VideoCapture(camera_id)
                
            if not cap.isOpened():
                print(f"❌ Failed to open camera")
                continue
                
            print(f"✅ Camera opened successfully")
            
            # Try to read 10 frames with increasing delays
            for i in range(1, 11):
                delay = i * 0.5  # Increasing delay for each attempt
                time.sleep(delay)
                print(f"   Attempt {i} (delay {delay}s): ", end="")
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"✅ Success! Frame received: {w}x{h}")
                    
                    # Save the frame as a file for verification
                    output_file = f"camera_test_{api}_{camera_id}.jpg"
                    cv2.imwrite(output_file, frame)
                    print(f"   Frame saved to {output_file}")
                    
                    success = True
                    break
                else:
                    print(f"❌ No frame received")
            
            # Release this camera before trying the next one
            cap.release()
            
            if success:
                break
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    if not success:
        print("\n🔴 RESULT: Could not access any camera frames after multiple attempts.")
        print("Possible solutions:")
        print("1. Check camera permissions in System Preferences -> Security & Privacy -> Camera")
        print("2. Restart your computer to reset all camera services")
        print("3. Try running: sudo killall VDCAssistant")
        print("4. Disconnect and reconnect external camera if applicable")
    else:
        print("\n🟢 RESULT: Successfully captured camera frames!")
        
if __name__ == "__main__":
    test_camera() 
