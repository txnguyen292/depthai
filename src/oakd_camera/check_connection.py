import depthai as dai

# Create a pipeline
pipeline = dai.Pipeline()

# Connect to the device
with dai.Device(pipeline) as device:
    # Print device information
    print("Connected to device:")
    print(f"Device name: {device.getDeviceName()}")
    print(f"USB speed: {device.getUsbSpeed()}")
    print(f"Connected cameras: {device.getConnectedCameras()}")
    print(f"Available stereo pairs: {device.getAvailableStereoPairs()}")
