import depthai as dai
import cv2
import numpy as np
import time
import yaml
import os
from loguru import logger

class OakDCamera:
    def __init__(self, config):
        self.config = config
        self.rgb_resolution = tuple(self.config["camera"]["rgb_resolution"])
        self.fps = self.config["camera"]["fps"]
        self.recording_time = self.config["camera"]["recording_time"]
        
        # Ensure output directory exists and is writable
        self.output_path = self.config["output"]["base_path"]
        try:
            os.makedirs(self.output_path, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(self.output_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            logger.error(f"No permission to create/write to directory: {self.output_path}")
            raise
        except OSError as e:
            logger.error(f"Failed to create/access output directory {self.output_path}: {e}")
            raise
        
        self.pipeline = None
        self.rgb_writer = None
        self.depth_writer = None
        self.frame_count = 0
        
        self.setup_pipeline()
        self.setup_video_writers()

    def setup_pipeline(self):
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        left = self.pipeline.create(dai.node.MonoCamera)
        right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutDepth.setStreamName("depth")

        # Properties
        camRgb.setPreviewSize(*self.rgb_resolution)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Linking
        camRgb.preview.link(xoutRgb.input)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.depth.link(xoutDepth.input)

    def setup_video_writers(self):
        # Video writers with H.264 codec for Mac compatibility
        rgb_path = os.path.join(self.output_path, self.config["output"]["rgb_filename"])
        depth_path = os.path.join(self.output_path, self.config["output"]["depth_filename"])
        
        try:
            self.rgb_writer = cv2.VideoWriter(
                rgb_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                self.rgb_resolution
            )
            if not self.rgb_writer.isOpened():
                raise IOError(f"Failed to initialize RGB video writer at {rgb_path}")

            self.depth_writer = cv2.VideoWriter(
                depth_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                self.rgb_resolution
            )
            if not self.depth_writer.isOpened():
                self.rgb_writer.release()  # Clean up RGB writer if depth writer fails
                raise IOError(f"Failed to initialize depth video writer at {depth_path}")

            logger.info(f"Video writers initialized:\n  RGB: {rgb_path}\n  Depth: {depth_path}")
        except Exception as e:
            logger.error(f"Error initializing video writers: {e}")
            # Clean up any writers that were successfully created
            if hasattr(self, 'rgb_writer') and self.rgb_writer is not None:
                self.rgb_writer.release()
            if hasattr(self, 'depth_writer') and self.depth_writer is not None:
                self.depth_writer.release()
            raise

    def process_depth_frame(self, depth_frame):
        if self.config["depth"]["normalize"]:
            depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        
        if self.config["depth"]["equalize_hist"]:
            depth_frame = cv2.equalizeHist(depth_frame)
        
        colormap = getattr(cv2, self.config["depth"]["colormap"])
        depth_frame = cv2.applyColorMap(depth_frame, colormap)
        return cv2.resize(depth_frame, self.rgb_resolution)

    def add_timestamp(self, frame):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def record(self):
        logger.info(f"Starting camera test - will record {self.recording_time} seconds of RGB and Depth streams...")

        with dai.Device(self.pipeline) as device:
            logger.info('Connected cameras:', device.getConnectedCameras())
            
            # Output queues
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            start_time = time.time()
            
            while time.time() - start_time < self.recording_time:
                inRgb = qRgb.get()
                inDepth = qDepth.get()

                # Process frames
                rgb_frame = inRgb.getCvFrame()
                depth_frame = self.process_depth_frame(inDepth.getFrame())
                
                # Add timestamps
                rgb_frame = self.add_timestamp(rgb_frame)
                depth_frame = self.add_timestamp(depth_frame)
                
                # Write frames
                self.rgb_writer.write(rgb_frame)
                self.depth_writer.write(depth_frame)
                
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    logger.info(f"Recorded {self.frame_count} frames...")

            self.cleanup()

    def cleanup(self):
        # Release writers
        self.rgb_writer.release()
        self.depth_writer.release()
        
        logger.success(f"\nTest complete! Recorded {self.frame_count} frames")
        logger.debug(f"Saved RGB stream to '{self.config['output']['rgb_filename']}'")
        logger.debug(f"Saved Depth stream to '{self.config['output']['depth_filename']}'")
        logger.debug("\nBoth files are in MP4 format and should be viewable on your MacBook")

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.exception(f"Error loading config from {config_path}: {e}")