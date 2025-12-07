# Import all necessary modules
from pathlib import Path
import os
import blobconverter
import cv2
import depthai as dai
import numpy as np
from loguru import logger

from .base import OakDBase
from src.utils.config import ConfigManager


class OakDObjectDetectionApp(OakDBase):
    def __init__(self, confidence_threshold=0.5, preview_size=(304, 304), save_video=False, display_info=True, output_path=None, config=None):
        # Use provided config or default
        if config is None:
            config = ConfigManager.DEFAULT_CONFIG.copy()
            # Override base path if output_path is provided, otherwise use default or current dir
            if output_path:
                 config["output"]["base_path"] = os.path.dirname(output_path)
            else:
                 config["output"]["base_path"] = os.path.dirname(os.path.abspath(__file__))

        # Initialize the base class
        super().__init__(config)
        
        # Object detection specific attributes
        self.confidence_threshold = confidence_threshold
        self.preview_size = preview_size
        self.save_video = save_video
        self.display_info = display_info
        self.video_output_path = output_path if output_path else os.path.join(self.output_path, "object_detection.mp4")
        self.frame = None
        self.detections = []
        self.video_writer = None
        
        # Initialize the labels for MobileNet-SSD
        self.labels = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
            "sheep", "sofa", "train", "tvmonitor"
        ]
        
        # Create and configure the pipeline
        self.pipeline = self.create_pipeline()
        
    def setup_pipeline(self):
        """Override the base class method to set up the object detection pipeline"""
        return self.create_pipeline()
        
    def create_pipeline(self):
        pipeline = dai.Pipeline()
        
        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        
        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutDepth.setStreamName("depth")
        
        # Properties
        camRgb.setPreviewSize(self.preview_size[0], self.preview_size[1])
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(30)
        
        # Set mono camera properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        
        # Set stereo depth properties
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # Make sure dimensions are multiples of 16 for stereo depth
        stereo_width = (self.preview_size[0] // 16) * 16
        stereo_height = (self.preview_size[1] // 16) * 16
        stereo.setOutputSize(stereo_width, stereo_height)
        
        # Set up the MobileNet spatial detection network
        spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        spatialDetectionNetwork.setConfidenceThreshold(self.confidence_threshold)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        
        # Linking
        camRgb.preview.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        spatialDetectionNetwork.out.link(xoutNN.input)
        
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        stereo.depth.link(xoutDepth.input)
        
        return pipeline
    
    def frameNorm(self, frame, bbox):
        """
        Convert normalized bounding box coordinates to pixel coordinates
        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def visualize_detections(self, frame, detections):
        """
        Draw bounding boxes, labels, and distance information for detections on the frame
        """
        # Add a counter for detected objects and their information
        detected_objects = []
        
        for detection in detections:
            # Get the label for this detection
            label = self.labels[detection.label]
            confidence = detection.confidence
            
            # Get bounding box coordinates - SpatialImgDetection uses different attribute names
            x1 = int(detection.xmin * frame.shape[1])
            y1 = int(detection.ymin * frame.shape[0])
            x2 = int(detection.xmax * frame.shape[1])
            y2 = int(detection.ymax * frame.shape[0])
            
            # Get spatial coordinates (3D position)
            spatial_coords = detection.spatialCoordinates
            x, y, z = spatial_coords.x, spatial_coords.y, spatial_coords.z
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label inside the bounding box
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Store object information for display in corner
            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'x': x/1000,  # Convert to meters
                'y': y/1000,  # Convert to meters
                'z': z/1000   # Convert to meters
            })
        
        # Display information in the corner of the frame if enabled
        if detected_objects and self.display_info:
            # Background for text
            padding = 10
            line_height = 25
            max_width = 300
            total_height = padding * 2 + line_height * (len(detected_objects) + 1)
            
            # Create semi-transparent overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, (frame.shape[1] - max_width - padding, padding), 
                         (frame.shape[1] - padding, total_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add title
            cv2.putText(frame, "Detected Objects:", 
                       (frame.shape[1] - max_width, padding + line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add object information
            for i, obj in enumerate(detected_objects):
                y_pos = padding + line_height * (i + 2)
                info_text = f"{obj['label']} ({obj['confidence']:.2f}) - {obj['z']:.2f}m"
                cv2.putText(frame, info_text, 
                           (frame.shape[1] - max_width, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        Run the object detection application with spatial detection
        """
        # Find an available device and run the pipeline
        try:
            with dai.Device(self.pipeline) as device:
                # Log connected cameras and device info
                logger.info(f'Connected cameras: {device.getConnectedCameras()}')
                logger.info(f'Device name: {device.getDeviceName()}')
                
                # Get output queues
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
                logger.info("Starting object detection with depth-based distance measurement. Press 'q' to quit.")
                
                # Initialize video writer if saving is enabled
                if self.save_video:
                    # Wait for first frame to get dimensions
                    while True:
                        inRgb = qRgb.tryGet()
                        if inRgb is not None:
                            frame = inRgb.getCvFrame()
                            h, w = frame.shape[:2]
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(os.path.abspath(self.video_output_path)), exist_ok=True)
                            self.video_writer = cv2.VideoWriter(
                                self.video_output_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                self.fps, # Use fps from config
                                (w, h)
                            )
                            logger.info(f"Recording video to {self.video_output_path}")
                            break
                
                while True:
                    # Try to get data from the queues
                    inRgb = qRgb.tryGet()
                    inDet = qDet.tryGet()
                    inDepth = qDepth.tryGet()
                    
                    if inRgb is not None:
                        # Get the frame in OpenCV format
                        self.frame = inRgb.getCvFrame()
                    
                    if inDet is not None:
                        # Get the detections with spatial data
                        self.detections = inDet.detections
                    
                    if self.frame is not None:
                        # Process the frame with detections and spatial information
                        frame_with_detections = self.visualize_detections(self.frame.copy(), self.detections)
                        
                        # Save frame to video if enabled
                        if self.save_video and self.video_writer is not None:
                            self.video_writer.write(frame_with_detections)
                        
                        # Display the frame
                        cv2.imshow("OAK-D Spatial Object Detection", frame_with_detections)
                    
                    # Check for key press to exit
                    if cv2.waitKey(1) == ord('q'):
                        break
        except Exception as e:
            logger.exception(f"Error: {e}")

        finally:
            # Clean up resources
            self.cleanup(True)
            
    def cleanup(self, display=True):
        """Override the base class cleanup method to handle video writer"""
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video saved to {self.video_output_path}")
        
        # Call the parent class cleanup method
        super().cleanup(display)
