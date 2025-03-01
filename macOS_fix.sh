#!/bin/bash

echo "=== macOS Camera Reset Tool ==="
echo "This script will attempt to reset the macOS camera system."
echo "This may help if your webcam is not being detected properly."
echo "You will need to enter your admin password for some operations."

# Function to check if command executed successfully
check_result() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 (failed, but continuing)"
    fi
}

# Close applications that might be using the camera
echo -e "\n1. Checking for applications using the camera..."
CAM_APPS=(
    "Photo Booth"
    "FaceTime"
    "Zoom"
    "Microsoft Teams"
    "Skype"
    "Google Chrome"
    "Firefox"
    "Safari"
)

for app in "${CAM_APPS[@]}"; do
    if pgrep -x "$app" > /dev/null; then
        echo "Found $app running, attempting to close..."
        killall "$app" 2>/dev/null
        check_result "Closed $app"
    fi
done

# Kill camera service processes
echo -e "\n2. Resetting camera service processes..."
sudo killall VDCAssistant 2>/dev/null
check_result "Reset VDCAssistant"
sudo killall AppleCameraAssistant 2>/dev/null
check_result "Reset AppleCameraAssistant"

# Force unload and reload the camera kernel extension (more aggressive)
echo -e "\n3. Checking camera kernel extensions..."
if kextstat | grep -q "com.apple.driver.AppleUSBVideoSupport" || kextstat | grep -q "com.apple.driver.AppleCameraInterface"; then
    echo "Camera kernel extensions found, attempting to reload..."
    sudo kextunload -b com.apple.driver.AppleUSBVideoSupport 2>/dev/null
    sudo kextunload -b com.apple.driver.AppleCameraInterface 2>/dev/null
    sleep 2
    sudo kextload -b com.apple.driver.AppleUSBVideoSupport 2>/dev/null
    sudo kextload -b com.apple.driver.AppleCameraInterface 2>/dev/null
    check_result "Reloaded camera kernel extensions"
else
    echo "Camera kernel extensions not found or not unloadable (normal for newer macOS)"
fi

# Check for camera permissions
echo -e "\n4. Checking camera permissions..."
if [ -d "/Library/Application Support/com.apple.TCC" ]; then
    echo "You may need to check camera permissions in System Preferences > Security & Privacy > Camera"
    echo "Make sure Terminal, Python, and any relevant apps have camera access."
fi

# Restart coreaudiod (audio can sometimes affect camera on Macs)
echo -e "\n5. Restarting audio service (can help with camera on some Macs)..."
sudo killall coreaudiod 2>/dev/null
check_result "Restarted audio service"

echo -e "\n=== Camera reset complete ==="
echo "Please wait 10-15 seconds before trying your camera again."
echo "If issues persist, you can try running the pose_estimation.py script with extra delay:"
echo "python pose_estimation.py --delay 5"
echo "Or specify a specific camera index:"
echo "python pose_estimation.py --camera 1" 
