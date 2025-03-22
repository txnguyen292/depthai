from src import OakDCamera, load_config
from loguru import logger
        

if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        logger.warning("Received interrupt signal, cleaning up...")
        if 'camera' in locals():
            camera.cleanup()
        sys.exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        config = load_config("config.yml")
        camera = OakDCamera(config)
        camera.record()
    except Exception as e:
        logger.exception(f"Error during camera operation: {e}")
        if 'camera' in locals():
            camera.cleanup()
        sys.exit(1)
