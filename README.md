# MediaPipe Pose Estimation with Webcam

This project captures video from a USB webcam and processes it through MediaPipe's pose estimation model, displaying pose estimation results in real-time.

## How to use

# End to end

Start an RTMP server on your computer (https://github.com/sallar/mac-local-rtmp-server) for smartglasses' video feed to stream to.
Then process it in python with :
```
python avfoundation_camera.py --rtmp rtmp://10.0.0.215/live/r1NfNH-jk
```
You can check the rtmp stream is on by playing it with ffmpeg :
```
ffplay rtmp://10.0.0.215/live/r1NfNH-jkg
```

To use the computer's webcam as input video feed do :
```
python avfoundation_camera.py --webcam
```

This will start a dev preview feed in opencv as well as serve it oon http://localhost:8080.
You have a minimal version at http://localhost:8080/minimal to opened as a webview on the smartglasses.
To enable sound to play the camera hutter sound, interact with the webpage with click for example.

Photos taken are viewable in a gallery at http://localhost:8080/view_photos, as well as on disk in static/

You can also run both processing from the webcam and smartglasses at the same time.
Just give a different port to one of them with `--port 8081`.

Support gestures :
- zooming in and out : pinch index and thumb in both hands and move hands apart or together
- rotating : pinch index and thumb in both hands and rotate your hands
- taking photos making a frame with hands, make an L-shape with both hands thumb/index and form the diagnonal corners of a rectangle, keep the gesture for a second. A flash and shutter sound will happen.

## Prerequisites

- Python 3.8 or higher
- A Logitech webcam (or any webcam connected via USB)
- Git (for cloning the repository)

## Setup and Running

1. Clone the repository or download the files
2. Make the setup script executable:

```bash
chmod +x setup_and_run.sh
```

3. Run the setup script to create a virtual environment, install dependencies, and run the application:

```bash
./setup_and_run.sh
```

The script will:
- Create a Python virtual environment (if it doesn't exist)
- Install the required dependencies from requirements.txt
- Run the pose estimation application

## Advanced Usage

The pose estimation script supports several command-line arguments for troubleshooting camera issues:

```bash
python pose_estimation.py --camera 1     # Use a specific camera index
python pose_estimation.py --delay 5      # Wait 5 seconds after camera init (helps with some webcams)
python pose_estimation.py --width 1280 --height 720   # Use HD resolution if supported
```

## Manual Setup (alternative)

If you prefer to set up manually:

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Run the script:
```bash
python pose_estimation.py
```

## Controls

- Press 'q' to quit the application

## Troubleshooting Webcam Issues

### Camera Not Found or Not Returning Frames

If you see "Camera connected but not returning frames":

1. Run the enhanced macOS camera reset script:
   ```bash
   chmod +x macOS_fix.sh
   ./macOS_fix.sh
   ```

2. Try increasing the initialization delay:
   ```bash
   python pose_estimation.py --delay 5
   ```

3. Specify a different camera index:
   ```bash
   python pose_estimation.py --camera 1
   ```

4. Close any other applications that might be using your camera (Zoom, Teams, FaceTime, Photo Booth, etc.)

### On macOS

macOS can be particularly troublesome with webcams:

1. Check Camera Permissions:
   - Go to System Preferences > Security & Privacy > Camera
   - Ensure Terminal, Python, and Cursor have permission to access the camera

2. For External USB Webcams:
   - Try different USB ports
   - Try a different USB cable
   - Disconnect and reconnect the webcam after running the macOS_fix.sh script

3. For Built-in FaceTime Camera:
   - The script will attempt to use it if no external camera is found

### On Windows

If you encounter issues with the webcam on Windows:

1. Check Device Manager to ensure your webcam is properly recognized
2. Update your webcam drivers
3. Try running as Administrator if permission issues occur

### Performance Tips

- If pose detection is slow, try:
  ```bash
  python pose_estimation.py --width 320 --height 240  # Lower resolution for speed
  ```

- The script automatically detects when the camera hangs and attempts to reconnect
- If you see frequent freezes, try restarting your computer or reconnecting the camera

## Troubleshooting

- If the webcam cannot be accessed, try changing the camera index in the code from 0 to 1 or another value
- If you encounter performance issues, try reducing the resolution in the code
- For GPU acceleration, ensure you have the appropriate CUDA libraries installed 
