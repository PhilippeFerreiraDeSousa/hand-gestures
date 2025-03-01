#!/usr/bin/env python3
"""
Direct AVFoundation Camera Access for macOS
This approach directly uses AVFoundation via OpenCV for more reliable camera access on macOS
"""

import cv2
import mediapipe as mp
import time
import os
import sys
import subprocess
import re

# def list_usb_devices():
#     """List USB devices to help diagnose connection issues"""
#     try:
#         result = subprocess.run(["system_profiler", "SPUSBDataType"], capture_output=True, text=True)
#         print("\n----- USB Devices Connected -----")
        
#         # Parse and display USB devices in a more readable format
#         usb_sections = re.split(r'\s{4}[^:\s]+:', result.stdout)
#         for section in usb_sections:
#             if "Vendor ID" in section or "Camera" in section or "Video" in section or "Logitech" in section:
#                 # Clean up and display relevant parts
#                 lines = [line.strip() for line in section.split('\n') if line.strip()]
#                 print('\n'.join(lines[:10]))  # Show first few lines of relevant sections
        
#         print("---------------------------------\n")
#     except Exception as e:
#         print(f"Could not list USB devices: {e}")

def main():
    # List USB devices to help diagnose any connection issues
    # list_usb_devices()
    
    # print("Initializing camera using macOS AVFoundation...")
    # print("NOTE: Having multiple USB devices can cause bandwidth or power issues.")
    # print("If camera detection fails, try disconnecting other USB devices temporarily.\n")
    
    # # Use different camera options, prioritizing external camera configurations
    # # This ordering prioritizes configurations that work better with multiple USB devices
    # camera_options = [
    #     # First try direct index access with explicit backend
    #     (0, cv2.CAP_AVFOUNDATION, "AVFoundation backend, camera 0"),
    #     (1, cv2.CAP_AVFOUNDATION, "AVFoundation backend, camera 1"),
        
    #     # Then try string-based camera access
    #     ("avfoundation:0", None, "String URL for camera 0"),
    #     ("avfoundation:1", None, "String URL for camera 1"),
    #     ("avfoundation:0:0", None, "AVFoundation camera 0, audio 0"),
        
    #     # Try with lower resolution hint (helps with bandwidth issues)
    #     ("avfoundation:0::320x240", None, "Camera 0 with lower resolution"),
    #     ("avfoundation:1::320x240", None, "Camera 1 with lower resolution"),
        
    #     # Last resort options
    #     (0, cv2.CAP_ANY, "Default backend, camera 0"),
    #     (1, cv2.CAP_ANY, "Default backend, camera 1")
    # ]
    
    # # Try all camera options
    # cap = None
    # successful_option = None
    # successful_description = None
    
    # for option, backend, description in camera_options:
    #     print(f"Trying: {description}")
    #     try:
    #         if backend is not None:
    #             # Use index with specific backend
    #             cap = cv2.VideoCapture(option, backend)
    #         else:
    #             # Use string format
    #             cap = cv2.VideoCapture(option)
                
    #         if cap is not None and cap.isOpened():
    #             print(f"  ✅ Camera opened with {description}")
                
    #             # Try to read a test frame with increasing delay
    #             success = False
    #             for attempt in range(1, 5):
    #                 delay = attempt * 1.0  # Increasing delay for each attempt
    #                 print(f"  Waiting {delay}s for camera to initialize (attempt {attempt}/4)...")
    #                 time.sleep(delay)
                    
    #                 ret, frame = cap.read()
    #                 if ret and frame is not None:
    #                     print(f"  ✅ Success! Frame received: {frame.shape[1]}x{frame.shape[0]}")
    #                     successful_option = option
    #                     successful_description = description
    #                     success = True
    #                     break
    #                 else:
    #                     print(f"  ❌ No frame received after {delay}s wait")
                
    #             if success:
    #                 break
    #             else:
    #                 print(f"❌ Option '{description}' connected but no frames - trying next option")
    #                 cap.release()
    #                 cap = None
    #         else:
    #             print(f"❌ Failed to open camera with option: {description}")
    #             if cap is not None:
    #                 cap.release()
    #                 cap = None
    #     except Exception as e:
    #         print(f"❌ Error with {description}: {str(e)}")
    #         if cap is not None:
    #             cap.release()
    #             cap = None
    
    # if cap is None or not cap.isOpened():
    #     print("\n❌ ERROR: Could not access any camera after trying all options.")
    #     print("\nTROUBLESHOOTING FOR MULTIPLE USB DEVICES:")
    #     print("1. USB BANDWIDTH ISSUES: Try disconnecting other USB devices")
    #     print("2. TRY DIFFERENT USB PORTS: Some ports may have more bandwidth or power")
    #     print("3. USE A POWERED USB HUB: This can resolve power-related issues")
    #     print("4. DISCONNECT COMPUTER: If another computer is connected via USB, try disconnecting it")
    #     return
    
    # print(f"\n✅ Successfully connected to camera using {successful_description}")

    option = 0
    successful_option = 0
    backend = cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(option, backend)
    
    # # Try lower resolution first for better performance with multiple USB devices
    # camera_width = 320
    # camera_height = 240
    
    # # Try to set camera properties
    # original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f"Original camera resolution: {original_width}x{original_height}")
    
    # # Always try to set a lower resolution when multiple USB devices are connected
    # # This helps with bandwidth issues
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    # Check what resolution we actually got
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Working with resolution: {actual_width}x{actual_height}")
    
    # Initialize MediaPipe Pose with lightweight settings to improve performance
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    
    # Use lighter model complexity due to potential USB bandwidth limitations
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # 0=Lite for better performance
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # FPS variables
    prev_time = 0
    curr_time = 0
    frame_count = 0
    start_time = time.time()
    
    print("Starting video stream. Press 'q' to quit.")
    
    # Variables to track frame stability
    consecutive_failures = 0
    max_failures = 5
    
    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"Failed to get frame (attempt {consecutive_failures}/{max_failures})...")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Attempting to reset the camera...")
                    cap.release()
                    time.sleep(1)
                    
                    # Try to reopen the camera with the last successful option
                    if isinstance(successful_option, int) and backend is not None:
                        cap = cv2.VideoCapture(successful_option, backend)
                    else:
                        cap = cv2.VideoCapture(successful_option)
                    
                    if not cap.isOpened():
                        print("Could not reopen camera. Exiting.")
                        break
                    
                    # Reset counter
                    consecutive_failures = 0
                    continue
                
                time.sleep(0.1)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Update FPS
            frame_count += 1
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = pose.process(rgb_frame)
            
            # Draw pose on the frame
            output_frame = frame.copy()
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_drawing_spec,
                    connection_drawing_spec=pose_drawing_spec
                )
            
            # Display FPS and info
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Res: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('MediaPipe Pose (AVFoundation)', output_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(0.1)
    
    # Clean up
    print(f"Processed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main() 
