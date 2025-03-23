import depthai as dai
from loguru import logger


def main():
    logger.info("Starting connection check...")
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Connect to the device
    with dai.Device(pipeline) as device:
        # Print device information
        logger.info("Connected to device:")
        logger.info(f"Device name: {device.getDeviceName()}")
        logger.info(f"USB speed: {device.getUsbSpeed()}")
        logger.info(f"Connected cameras: {device.getConnectedCameras()}")
        logger.info(f"Available stereo pairs: {device.getAvailableStereoPairs()}")

if __name__ == "__main__":
    main()