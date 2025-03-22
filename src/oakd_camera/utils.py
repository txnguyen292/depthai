import yaml
from loguru import logger

def load_config(config_path: str) -> dict[str, any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.exception(f"Error loading config from {config_path}: {e}")
