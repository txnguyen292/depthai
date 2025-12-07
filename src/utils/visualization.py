import depthai as dai
import cv2
import numpy as np
from loguru import logger

def show_video_stream():
    """
    Streams and displays RGB and Depth video from the OAK-D camera.
    """
    logger.info("Starting video stream...")
    
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define a color camera node
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)

    # Create a depth camera node
    mono_left = pipeline.createMonoCamera()
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    mono_right = pipeline.createMonoCamera()
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Create XLinkOut nodes to stream video to the host
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    try:
        # Connect to the device and start the pipeline
        with dai.Device(pipeline) as device:
            # Get the video output queues
            rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            logger.info("Video stream started. Press 'q' to exit.")

            while True:
                # Get the latest RGB frame
                rgb_frame = rgb_queue.get().getCvFrame()

                # Get the latest depth frame
                depth_packet = depth_queue.get()
                depth_frame = depth_packet.getFrame()

                # Normalize depth frame for visualization
                depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_frame = cv2.convertScaleAbs(depth_frame)

                # Apply a colormap for better visualization
                depth_colored = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

                # Display the frames in separate windows
                cv2.imshow("RGB Video", rgb_frame)
                cv2.imshow("Depth Video", depth_colored)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up
            cv2.destroyAllWindows()
            logger.info("Video stream stopped.")
            
    except Exception as e:
        logger.error(f"Error during video streaming: {e}")
        raise
