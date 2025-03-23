import depthai as dai
import cv2
import numpy as np
import time
import os
from loguru import logger

class OakDBase:
    def __init__(self, config):
        self.config = config
        self.rgb_resolution = tuple(self.config["camera"]["rgb_resolution"])
        self.fps = self.config["camera"]["fps"]
        self.recording_time = self.config["camera"]["recording_time"]
        
        # Ensure output directory exists and is writable
        self.output_path = os.path.join(self.config["output"]["base_path"], "data")
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
        self.frame_count = 0
        
    def setup_pipeline(self):
        """
        Set up the DepthAI pipeline. This method should be overridden by child classes.
        """
        raise NotImplementedError("Subclasses must implement setup_pipeline()")
    
    def add_timestamp(self, frame):
        """
        Add a timestamp to a frame
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def process_depth_frame(self, depth_frame):
        """
        Process a depth frame for visualization
        """
        if self.config["depth"]["normalize"]:
            depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        
        if self.config["depth"]["equalize_hist"]:
            depth_frame = cv2.equalizeHist(depth_frame)
        
        colormap = getattr(cv2, self.config["depth"]["colormap"])
        depth_frame = cv2.applyColorMap(depth_frame, colormap)
        return cv2.resize(depth_frame, self.rgb_resolution)
    
    def cleanup(self, display=False):
        """
        Clean up resources. This method should be extended by child classes.
        """
        # Close any open windows
        if display:
            cv2.destroyAllWindows()
        
        logger.success(f"\nOperation complete! Processed {self.frame_count} frames")
