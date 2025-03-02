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

    # Zoom and rotation control variables
    zoom_scale = 1.0  # Initial zoom level (no zoom)
    rotation_angle = 0.0  # Initial rotation angle in degrees
    min_zoom = 1.0
    max_zoom = 3.0
    min_rotation = -45.0  # Limit rotation to ±45 degrees for usability
    max_rotation = 45.0
    zoom_speed = 0.8 # Speed factor for zoom
    rotation_speed = 2.0  # Speed factor for rotation

    # Two-hand gesture variables
    prev_hands_distance = None
    prev_hands_angle = None
    zoom_smoothing = 0.2  # Lower value = smoother zoom but less responsive
    rotation_smoothing = 0.15  # Smoother rotation
    two_hand_gesture_active = False

    # For on-screen display of control
    zoom_rect_size = 100
    rotation_arc_radius = 50

    print("Starting video stream. Press 'q' to quit.")
    print("\nTwo-Hand Gesture Instructions:")
    print("1. Show both hands with thumb and index finger pinching")
    print("2. Move hands diagonally apart from each other to zoom in")
    print("3. Move hands closer together to zoom out")
    print("4. Rotate your hands to rotate the view")
    print("5. Press 'r' to reset zoom and rotation")

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

            # Step 1: Apply rotation if needed
            if abs(rotation_angle) > 0.1:  # Only apply rotation if significant
                # Calculate rotation matrix
                center = (frame_w // 2, frame_h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

                # Apply rotation with border handling (black borders)
                frame = cv2.warpAffine(frame, rotation_matrix, (frame_w, frame_h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

            # Process current frame for transformation (zoom and rotation)
            # Step 2: Apply zoom (center crop and resize)
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

            # Reset gesture active flag
            two_hand_gesture_active = False

            # Data for pinch points
            pinch_points = []
            valid_pinches = []

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                # First, process all hands to detect pinch gestures
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw the hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract thumb and index finger landmarks for pinch gesture detection
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Get pixel coordinates
                    h, w, c = output_frame.shape
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                    # Calculate distance between thumb and index finger (pinch gesture)
                    pinch_distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

                    # Calculate pinch point (midpoint between thumb and index)
                    pinch_x = (thumb_x + index_x) // 2
                    pinch_y = (thumb_y + index_y) // 2
                    pinch_point = (pinch_x, pinch_y)

                    # Check if this is a pinching gesture
                    # We'll consider it a pinch if thumb and index are close enough
                    is_pinching = pinch_distance < 50  # Adjust threshold as needed

                    # Store the pinch point and validity
                    pinch_points.append(pinch_point)
                    valid_pinches.append(is_pinching)

                    # Draw pinch visualization
                    pinch_color = (0, 255, 0) if is_pinching else (0, 0, 255)
                    # Draw line between thumb and index finger
                    cv2.line(output_frame, (thumb_x, thumb_y), (index_x, index_y), pinch_color, 2)
                    # Draw circles at fingertips
                    cv2.circle(output_frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
                    cv2.circle(output_frame, (index_x, index_y), 8, (0, 0, 255), -1)
                    # Draw pinch point
                    cv2.circle(output_frame, pinch_point, 12, pinch_color, -1 if is_pinching else 2)

                    # Add pinch label
                    pinch_status = "Pinched" if is_pinching else "Not Pinched"
                    cv2.putText(output_frame, f"Hand {hand_idx+1}: {pinch_status}",
                                (pinch_x - 70, pinch_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)

                # Now check if we have two valid pinch gestures for zoom control
                if len(pinch_points) == 2 and valid_pinches[0] and valid_pinches[1]:
                    two_hand_gesture_active = True

                    # Calculate distance between the two pinch points
                    hands_distance = math.sqrt(
                        (pinch_points[0][0] - pinch_points[1][0])**2 +
                        (pinch_points[0][1] - pinch_points[1][1])**2
                    )

                    # Calculate angle between the two pinch points
                    dx = pinch_points[1][0] - pinch_points[0][0]
                    dy = pinch_points[1][1] - pinch_points[0][1]
                    current_angle = math.degrees(math.atan2(dy, dx))

                    # Draw connection line between pinch points
                    cv2.line(output_frame, pinch_points[0], pinch_points[1], (255, 255, 0), 2)

                    # Display the distance and angle
                    midpoint_x = (pinch_points[0][0] + pinch_points[1][0]) // 2
                    midpoint_y = (pinch_points[0][1] + pinch_points[1][1]) // 2

                    distance_text = f"Distance: {hands_distance:.0f}px"
                    cv2.putText(output_frame, distance_text,
                                (midpoint_x - 70, midpoint_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    angle_text = f"Angle: {current_angle:.1f}°"
                    cv2.putText(output_frame, angle_text,
                                (midpoint_x - 70, midpoint_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                    # Update zoom and rotation based on changes
                    if prev_hands_distance is not None and prev_hands_angle is not None:
                        # Calculate distance delta for zoom
                        distance_delta = hands_distance - prev_hands_distance

                        # Calculate angle delta for rotation
                        angle_delta = current_angle - prev_hands_angle
                        # Normalize angle delta to handle wrap-around (e.g., 179° to -179°)
                        if angle_delta > 180:
                            angle_delta -= 360
                        elif angle_delta < -180:
                            angle_delta += 360

                        # Apply zoom change with smoothing
                        if abs(distance_delta) > 5:  # Small threshold to avoid tiny changes
                            zoom_delta = distance_delta * zoom_speed / 200.0  # Positive delta: hands apart = zoom in
                            new_zoom = zoom_scale + zoom_delta
                            # Apply smoothing
                            zoom_scale = (1 - zoom_smoothing) * zoom_scale + zoom_smoothing * new_zoom
                            # Clamp to min/max range
                            zoom_scale = max(min_zoom, min(max_zoom, zoom_scale))

                        # Apply rotation change with smoothing
                        if abs(angle_delta) > 0.5:  # Small threshold for rotation stability
                            rotation_delta = angle_delta * rotation_speed
                            new_rotation = rotation_angle + rotation_delta
                            # Apply smoothing
                            rotation_angle = (1 - rotation_smoothing) * rotation_angle + rotation_smoothing * new_rotation
                            # Clamp to min/max range
                            rotation_angle = max(min_rotation, min(max_rotation, rotation_angle))

                    # Update previous values
                    prev_hands_distance = hands_distance
                    prev_hands_angle = current_angle

                else:
                    # If not in two-hand pinch gesture, reset previous values
                    if prev_hands_distance is not None:
                        prev_hands_distance = None
                    if prev_hands_angle is not None:
                        prev_hands_angle = None

            # Display zoom and rotation levels
            zoom_text = f"Zoom: {zoom_scale:.2f}x"
            rotation_text = f"Rotation: {rotation_angle:.1f}°"

            cv2.putText(output_frame, zoom_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output_frame, rotation_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # Draw zoom indicator rectangle
            rect_width = int(zoom_rect_size / max_zoom * zoom_scale)
            cv2.rectangle(output_frame,
                         (frame_w - zoom_rect_size - 10, 10),
                         (frame_w - 10, 30),
                         (100, 100, 100), -1)
            cv2.rectangle(output_frame,
                         (frame_w - zoom_rect_size - 10, 10),
                         (frame_w - zoom_rect_size - 10 + rect_width, 30),
                         (0, 255, 255) if two_hand_gesture_active else (0, 200, 200), -1)

            # Draw rotation indicator arc
            center_x = frame_w - rotation_arc_radius - 10
            center_y = rotation_arc_radius + 50

            # Draw arc background
            cv2.ellipse(output_frame, (center_x, center_y), (rotation_arc_radius, rotation_arc_radius),
                      0, 180, 360, (100, 100, 100), -1)

            # Calculate arc angle based on rotation
            rotation_percentage = (rotation_angle - min_rotation) / (max_rotation - min_rotation)
            arc_angle = 180 + rotation_percentage * 180

            # Draw active arc portion
            cv2.ellipse(output_frame, (center_x, center_y), (rotation_arc_radius, rotation_arc_radius),
                      0, 270, arc_angle, (255, 200, 0) if two_hand_gesture_active else (200, 150, 0), -1)

            # Draw indicator line
            indicator_angle_rad = math.radians(arc_angle - 90)
            end_x = int(center_x + rotation_arc_radius * math.cos(indicator_angle_rad))
            end_y = int(center_y + rotation_arc_radius * math.sin(indicator_angle_rad))
            cv2.line(output_frame, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)

            # Draw zero indicator
            zero_x = int(center_x)
            zero_y = int(center_y - rotation_arc_radius)
            cv2.circle(output_frame, (zero_x, zero_y), 3, (255, 255, 255), -1)

            # Display gesture status
            gesture_status = "Active" if two_hand_gesture_active else "Inactive"
            cv2.putText(output_frame, f"Gesture: {gesture_status}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0) if two_hand_gesture_active else (200, 200, 200), 2)

            # Display hand count
            hand_count = 0 if results.multi_hand_landmarks is None else len(results.multi_hand_landmarks)
            cv2.putText(output_frame, f"Hands: {hand_count}/2", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Display FPS and info
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Res: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add instruction reminders
            cv2.putText(output_frame, "Pinch both hands: move apart = zoom, rotate = rotate",
                        (10, frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(output_frame, "Press 'r' to reset view",
                        (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Display the frame
            cv2.imshow('Hand Gesture Zoom & Rotation Control', output_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset zoom and rotation
                zoom_scale = 1.0
                rotation_angle = 0.0
                print("View reset: Zoom=1.0x, Rotation=0°")

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
