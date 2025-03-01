#!/bin/bash

# Exit on error
set -e

echo "Setting up MediaPipe Pose Estimation with webcam in a virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Make scripts executable
chmod +x macOS_fix.sh

# Check if running on macOS and suggest camera test
if [ "$(uname)" == "Darwin" ]; then
    echo "Detected macOS system"
    
    # Ask if the user wants to reset the camera system
    read -p "Do you want to run the macOS camera reset script first? (y/n): " reset_camera
    if [[ $reset_camera == "y" || $reset_camera == "Y" ]]; then
        echo "Running macOS camera reset script..."
        sudo ./macOS_fix.sh
        echo "Waiting 5 seconds for camera services to restart..."
        sleep 5
    fi
    
    # Ask if user wants to run the camera test
    read -p "Do you want to run the camera test first to verify camera access? (y/n): " run_test
    if [[ $run_test == "y" || $run_test == "Y" ]]; then
        echo "Running macOS camera test..."
        python macos_camera_test.py
    fi
    
    # Run the macOS specific camera script
    echo "Running AVFoundation camera pose estimation script..."
    python avfoundation_camera.py
else
    # On other platforms, use the regular script
    echo "Running pose estimation script..."
    python pose_estimation.py
fi

# Deactivate virtual environment when done
deactivate

echo "Done!" 
