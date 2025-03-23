import depthai as dai
import cv2
import numpy as np
import time
import os
from loguru import logger
from .oakd_base import OakDBase

class OakDCamera(OakDBase):
    def __init__(self, config):
        super().__init__(config)
        self.rgb_writer = None
        self.depth_writer = None
        
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

    def cleanup(self, display=False):
        # Release writers
        if hasattr(self, 'rgb_writer') and self.rgb_writer is not None:
            self.rgb_writer.release()
        if hasattr(self, 'depth_writer') and self.depth_writer is not None:
            self.depth_writer.release()
        
        super().cleanup(display)
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