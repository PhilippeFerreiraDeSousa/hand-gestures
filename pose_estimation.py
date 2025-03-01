import cv2
import numpy as np
import time
import mediapipe as mp
import sys
import os
import argparse
import platform

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MediaPipe Pose Estimation with webcam')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use (default: auto-detect)')
    parser.add_argument('--retry', type=int, default=5, help='Number of times to retry camera initialization')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay in seconds after camera initialization')
    parser.add_argument('--width', type=int, default=640, help='Camera capture width')
    parser.add_argument('--height', type=int, default=480, help='Camera capture height')
    args = parser.parse_args()

    # Camera initialization
    cap = None
    is_macos = platform.system() == 'Darwin'

    # Print system info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")

    # Try specific camera index if provided
    if args.camera is not None:
        print(f"Trying specified camera index {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if cap is not None and cap.isOpened():
            camera_index = args.camera
            print(f"Successfully connected to camera at index {args.camera}")
        else:
            print(f"Failed to connect to specified camera at index {args.camera}")
            if cap is not None:
                cap.release()
            cap = None

    # Try automatic detection if no camera specified or specified camera failed
    if cap is None:
        print("Attempting to auto-detect webcam...")

        # On macOS, try some known device paths first
        if is_macos:
            mac_devices = [
                0,  # Default camera
                1,  # Common external camera index
                '/dev/video0',  # USB cameras on newer macOS
                'avfoundation:0',  # Built-in FaceTime camera
                'avfoundation:1'   # First external camera with AVFoundation
            ]

            for device in mac_devices:
                print(f"Trying macOS device: {device}")
                cap = cv2.VideoCapture(device)
                if cap is not None and cap.isOpened():
                    camera_index = device
                    print(f"Successfully connected to macOS camera: {device}")
                    break
                else:
                    print(f"Failed to connect to macOS camera: {device}")
                    if cap is not None:
                        cap.release()
                    cap = None

        # Try standard indices if still not connected
        if cap is None:
            max_attempts = 4  # Try indices 0-3
            for i in range(max_attempts):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    camera_index = i
                    print(f"Successfully connected to camera at index {i}")
                    break
                else:
                    print(f"Failed to connect to camera at index {i}")
                    if cap is not None:
                        cap.release()
                    cap = None

    # If we still couldn't open the camera
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam on any available index.")
        print("Please check your camera connection and permissions.")
        if is_macos:
            print("On macOS, try running the macOS_fix.sh script to reset the camera system:")
            print("chmod +x macOS_fix.sh && ./macOS_fix.sh")
            print("Also ensure terminal/Cursor has permission to access the camera in System Preferences > Security & Privacy > Camera")
        return

    # Give the camera time to initialize and stabilize
    print(f"Waiting {args.delay} seconds for camera to initialize...")
    time.sleep(args.delay)

    # Try to set camera properties
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Original camera resolution: {original_width}x{original_height}")

    # Try to set the requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Verify what resolution we actually got
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Actual camera resolution: {actual_width}x{actual_height}")

    # Retry logic for getting the first frame
    retry_count = 0
    test_frame = None

    while retry_count < args.retry:
        # Check if the camera is actually returning frames
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"Successfully captured test frame with shape: {test_frame.shape}")
            break

        print(f"Retry {retry_count+1}/{args.retry}: Failed to get test frame, waiting...")
        time.sleep(1)
        retry_count += 1

    if test_frame is None:
        print("Error: Camera connected but not returning frames after multiple attempts.")
        print("Try running with --delay 5 to give the camera more time to initialize.")
        print("Also check if another application is using the camera.")
        if is_macos:
            print("On macOS, try the macOS_fix.sh script to reset the camera system.")
        return

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    # Initialize Pose model with desired settings
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    frame_count = 0
    start_time = time.time()
    last_frame_time = time.time()

    # Hang detection variables
    max_frame_delay = 5.0  # seconds without a new frame before considering hung

    print("Starting video stream. Press 'q' to quit.")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Check for camera hang
        current_time = time.time()
        if current_time - last_frame_time > max_frame_delay:
            print(f"Warning: No new frames for {max_frame_delay} seconds, attempting to restart camera...")
            cap.release()
            time.sleep(1)

            # Try to reopen the same camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print("Error: Could not reconnect to webcam.")
                break

            # Reset the timer and try again
            last_frame_time = time.time()
            continue

        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            time.sleep(0.1)  # Small delay to avoid CPU spin
            continue

        # Update timing for hang detection
        last_frame_time = current_time

        # Update frame count for average FPS calculation
        frame_count += 1

        # Calculate instantaneous FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Calculate average FPS
        elapsed_time = curr_time - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Convert frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        output_frame = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_drawing_spec,
                connection_drawing_spec=pose_drawing_spec
            )

        # Display FPS and frame size on the frame
        cv2.putText(output_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Frame: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow('MediaPipe Pose Estimation', output_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds (Avg FPS: {avg_fps:.2f})")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
