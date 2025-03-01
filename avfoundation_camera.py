#!/usr/bin/env python3
"""
Direct AVFoundation Camera Access for macOS with MediaPipe Hand Detection
This approach directly uses AVFoundation via OpenCV for more reliable camera access on macOS
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import math

def main():

    option = 0
    backend = cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(option, backend)

    # Try lower resolution first for better performance with multiple USB devices
    # set_camera_resolution(cap, 320, 240)

    # Check what resolution we actually got
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Working with resolution: {actual_width}x{actual_height}")

    # Initialize MediaPipe Hands with lightweight settings to improve performance
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize the hand detection model
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # FPS variables
    prev_time = 0
    curr_time = 0
    frame_count = 0

    # Zoom control variables
    zoom_scale = 1.0  # Initial zoom level (no zoom)
    min_zoom = 1.0
    max_zoom = 3.0
    zoom_speed = 0.1

    # Pinch gesture variables
    prev_pinch_distance = None
    zoom_smoothing = 0.2  # Lower value = smoother zoom but less responsive

    # For on-screen display of zoom control
    pinch_active = False
    zoom_rect_size = 100

    print("Starting video stream. Press 'q' to quit.")
    print("\nZoom Gesture Instructions:")
    print("1. Show one hand with thumb and index finger extended")
    print("2. Pinch thumb and index finger together to zoom in")
    print("3. Move thumb and index finger apart to zoom out")
    print("4. Press 'r' to reset zoom level")

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
                    print("Exiting.")
                    break

                time.sleep(0.1)
                continue

            # Reset failure counter on success
            consecutive_failures = 0

            # Update FPS
            frame_count += 1
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # Get original frame dimensions
            frame_h, frame_w = frame.shape[:2]

            # Apply zoom (center crop and resize)
            if zoom_scale > 1.0:
                # Calculate the region to crop based on zoom level
                crop_w = int(frame_w / zoom_scale)
                crop_h = int(frame_h / zoom_scale)

                # Calculate crop coordinates to keep the center
                x1 = int((frame_w - crop_w) / 2)
                y1 = int((frame_h - crop_h) / 2)
                x2 = x1 + crop_w
                y2 = y1 + crop_h

                # Crop the frame to the calculated region
                zoomed_frame = frame[y1:y2, x1:x2]

                # Resize back to original size
                frame = cv2.resize(zoomed_frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

            # Flip the image horizontally for a more natural selfie-view display
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # To improve performance, mark the image as not writeable
            rgb_frame.flags.writeable = False

            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)

            # Mark the image as writeable again for drawing
            rgb_frame.flags.writeable = True

            # Create output frame
            output_frame = frame.copy()

            # Reset pinch active flag
            pinch_active = False

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw the hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract thumb and index finger landmarks for zoom gesture
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Get pixel coordinates
                    h, w, c = output_frame.shape
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Calculate distance between thumb and index finger (pinch gesture)
                    pinch_distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

                    # Check if this is a pinching gesture (by checking if other fingers are closed)
                    # Simple heuristic: check position of middle finger compared to index
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    # Middle knuckle as reference point
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                    # Check if other fingers are curled (lower y value means higher on screen in image coordinates)
                    other_fingers_curled = (
                        middle_tip.y > middle_mcp.y and
                        ring_tip.y > middle_mcp.y and
                        pinky_tip.y > middle_mcp.y
                    )

                    # If we detect a proper pinch gesture (thumb, index extended, others curled)
                    if other_fingers_curled:
                        pinch_active = True

                        # Draw line between thumb and index finger
                        cv2.line(output_frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)

                        # Draw circles at fingertips
                        cv2.circle(output_frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
                        cv2.circle(output_frame, (index_x, index_y), 8, (0, 0, 255), -1)

                        # Calculate and display distance
                        pinch_distance_text = f"Pinch: {pinch_distance:.0f}px"
                        cv2.putText(output_frame, pinch_distance_text, (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        # Update zoom based on pinch distance
                        if prev_pinch_distance is not None:
                            # Calculate delta movement
                            delta = pinch_distance - prev_pinch_distance

                            # Apply zoom change with smoothing
                            if abs(delta) > 2:  # Small threshold to avoid tiny changes
                                zoom_delta = -delta * zoom_speed / 100.0  # Negative delta: pinch in = zoom in
                                new_zoom = zoom_scale + zoom_delta
                                # Apply smoothing
                                zoom_scale = (1 - zoom_smoothing) * zoom_scale + zoom_smoothing * new_zoom
                                # Clamp to min/max range
                                zoom_scale = max(min_zoom, min(max_zoom, zoom_scale))

                        # Update previous distance
                        prev_pinch_distance = pinch_distance
                    else:
                        # If not in pinch gesture, gradually reset previous distance
                        if prev_pinch_distance is not None:
                            prev_pinch_distance = None

            # Display zoom level even when hands aren't detected
            zoom_text = f"Zoom: {zoom_scale:.2f}x"
            cv2.putText(output_frame, zoom_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw zoom indicator rectangle
            rect_width = int(zoom_rect_size / max_zoom * zoom_scale)
            cv2.rectangle(output_frame,
                         (frame_w - zoom_rect_size - 10, 10),
                         (frame_w - 10, 30),
                         (100, 100, 100), -1)
            cv2.rectangle(output_frame,
                         (frame_w - zoom_rect_size - 10, 10),
                         (frame_w - zoom_rect_size - 10 + rect_width, 30),
                         (0, 255, 255) if pinch_active else (0, 200, 200), -1)

            # Display hand count
            hand_count = 0 if results.multi_hand_landmarks is None else len(results.multi_hand_landmarks)
            cv2.putText(output_frame, f"Hands: {hand_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Display FPS and info
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Res: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('MediaPipe Hand Detection with Zoom', output_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset zoom
                zoom_scale = 1.0
                print("Zoom reset to 1.0x")

        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(0.1)

    # Clean up
    print(f"Processed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def set_camera_resolution(cap, camera_width, camera_height):
    camera_width = 320
    camera_height = 240

    # Try to set camera properties
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Original camera resolution: {original_width}x{original_height}")

    # Always try to set a lower resolution when multiple USB devices are connected
    # This helps with bandwidth issues
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

if __name__ == "__main__":
    main()
