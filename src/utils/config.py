import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import os

class ConfigManager:
    DEFAULT_CONFIG = {
        "camera": {
            "rgb_resolution": [1280, 800],
            "fps": 30,
            "recording_time": 10
        },
        "output": {
            "base_path": "./data",
            "rgb_filename": "rgb_video.mp4",
            "depth_filename": "depth_video.mp4"
        },
        "depth": {
            "colormap": "COLORMAP_JET",
            "normalize": True,
            "equalize_hist": True
        }
    }

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a file or return defaults.
        """
        config = ConfigManager.DEFAULT_CONFIG.copy()
        
        if config_path:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        user_config = yaml.safe_load(f)
                        if user_config:
                            ConfigManager._update_recursive(config, user_config)
                    logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logger.error(f"Error loading config from {config_path}: {e}")
                    raise
            else:
                logger.warning(f"Config file {config_path} not found. Using defaults.")
        
        return config

    @staticmethod
    def _update_recursive(d: Dict, u: Dict) -> Dict:
        """
        Recursively update dictionary d with values from u.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = ConfigManager._update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @staticmethod
    def create_config_from_args(output_dir: Path, duration: int, fps: int, rgb_res: list = [1280, 800]) -> Dict[str, Any]:
        """
        Creates a config dictionary from CLI arguments, overriding defaults.
        """
        config = ConfigManager.DEFAULT_CONFIG.copy()
        
        # We need to make sure we are working with a deep copy for nested dicts if we modify them
        # But since we are assigning new values to keys, it should be fine for this level of depth
        # To be safe, let's do a proper deep copy if we were using a library, but here:
        import copy
        config = copy.deepcopy(ConfigManager.DEFAULT_CONFIG)

        config["output"]["base_path"] = str(output_dir)
        config["camera"]["recording_time"] = duration
        config["camera"]["fps"] = fps
        config["camera"]["rgb_resolution"] = rgb_res
        
        return config
