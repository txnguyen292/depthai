#!/usr/bin/env python3

import os
import sys
import signal
import typer
from loguru import logger
from typing_extensions import Annotated

from src import OakDCamera, load_config, OakDObjectDetectionApp

# Global variables for cleanup
camera = None
app = None

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.warning("Received interrupt signal, cleaning up...")
        cleanup()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def cleanup():
    """Clean up resources before exiting"""
    global camera
    if camera is not None:
        logger.info("Cleaning up camera resources...")
        camera.cleanup()

def run_object_detection(config):
    """Run the application in object detection mode"""
    global app
    logger.info("Starting object detection mode")
    
    # Get object detection configuration
    detection_config = config.get('detection', {})
    confidence_threshold = detection_config.get('confidence_threshold', 0.5)
    preview_size = tuple(detection_config.get('preview_size', (304, 304)))
    save_video = detection_config.get('save_video', False)
    display_info = detection_config.get('display_info', True)
    
    # Set output path for video recording
    output_path = None
    if save_video:
        output_dir = config.get('output', {}).get('base_path', '')
        output_filename = detection_config.get('output_filename', 'object_detection.mp4')
        output_path = os.path.join(output_dir, output_filename)
    
    # Create and run the object detection app
    app = OakDObjectDetectionApp(
        confidence_threshold=confidence_threshold,
        preview_size=preview_size,
        save_video=save_video,
        display_info=display_info,
        output_path=output_path,
        config=config
    )
    app.run()

def run_video_recording(config):
    """Run the application in video recording mode"""
    global camera
    logger.info("Starting video recording mode")
    camera = OakDCamera(config)
    camera.record()
app = typer.Typer()

@app.command("main", help="Main entry point for the application")
def main(
    config_path: Annotated[str, typer.Option(
        "-cp", "--config-path", help="Path to the configuration file")]
    ):
    """Main entry point for the application"""
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Set up logging
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        log_file = config.get('logging', {}).get('log_file', 'app.log')
        log_level = config.get('logging', {}).get('log_level', 'INFO')
        logger.add(log_file, level=log_level, rotation='5MB')
        
        # Determine which mode to run in
        mode = config.get('mode', 'record')  # Default to record mode if not specified
        
        # Run the appropriate mode
        if mode == 'object_detection':
            run_object_detection(config)
        else:
            run_video_recording(config)
    except Exception as e:
        logger.exception(f"Error during camera operation: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    app()
