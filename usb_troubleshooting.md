# USB Camera Troubleshooting Guide for macOS

## Problem: Camera Detection Issues When Multiple USB Devices Are Connected

When you have multiple USB devices connected (like a webcam and another computer), several issues can occur:

### 1. USB Bandwidth Constraints

**Symptoms:**
- Camera is detected but no frames are received
- Camera works intermittently 
- Low frame rate or freezing

**Solutions:**
- **Disconnect other USB devices** temporarily to test the camera
- **Use USB ports on different controllers** (different sides of laptop often use different controllers)
- **Reduce resolution** in the application to decrease bandwidth needs
- **Directly connect the camera** without using USB hubs

### 2. Power Distribution Issues

**Symptoms:**
- Camera disconnects randomly
- Camera is detected but doesn't initialize properly
- System reports "USB device drawing too much power"

**Solutions:**
- **Use a powered USB hub** for the camera
- **Connect directly** to the computer's USB ports (not through another hub)
- **Try USB-C ports** if available (they often provide more power)
- **Disconnect high-power USB devices** when using the camera

### 3. macOS-Specific USB Enumeration Issues

**Symptoms:**
- Camera is shown in System Information but not accessible in apps
- "Camera in use by another application" errors
- Camera works in some apps but not others

**Solutions:**
- **Check Privacy Settings**: System Settings → Privacy & Security → Camera
- **Reset USB ports**: Run `sudo killall -STOP -c usbd; sudo killall -CONT -c usbd`
- **Reset the entire camera system** using the macOS_fix.sh script
- **Restart your Mac** after disconnecting all USB devices, then reconnect only the camera first

### 4. Testing Your Configuration

1. Run the diagnostic script to see which camera access method works:
   ```
   python macos_camera_test.py
   ```

2. Check which USB devices are connected and their specifications:
   ```
   system_profiler SPUSBDataType
   ```

3. Try the camera with just the essential USB devices connected:
   ```
   python avfoundation_camera.py
   ```

## USB Port Priority

Not all USB ports on a Mac are created equal. Try these approaches:

1. **Direct MacBook/iMac Ports**: These typically have the highest priority and most power
2. **Left vs Right Side**: On MacBooks, ports on different sides may use different USB controllers
3. **USB-C vs USB-A**: USB-C ports often have higher bandwidth and power availability
4. **Thunderbolt ports**: These typically have the highest bandwidth and best camera support

## Camera Priority

If you're having trouble with camera detection while another computer is connected:

1. **Connect and power on the camera first** before connecting the other computer
2. **Use a powered USB hub** for the other computer, direct connection for the camera
3. **Use different USB bus controllers** (different sides of the Mac) for the devices 
