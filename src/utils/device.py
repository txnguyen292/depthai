import depthai as dai
from loguru import logger

def check_connection_status() -> dict:
    """
    Checks connection to OAK-D device and returns device details.
    """
    logger.info("Starting connection check...")
    pipeline = dai.Pipeline()
    
    try:
        with dai.Device(pipeline) as device:
            info = {
                "device_name": device.getDeviceName(),
                "usb_speed": device.getUsbSpeed(),
                "connected_cameras": device.getConnectedCameras(),
                "stereo_pairs": device.getAvailableStereoPairs()
            }
            logger.info(f"Connected to device: {info['device_name']}")
            return info
    except Exception as e:
        logger.error(f"Failed to connect to device: {e}")
        raise
