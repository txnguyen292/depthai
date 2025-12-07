import pytest
from pathlib import Path
from src.utils.config import ConfigManager

def test_load_defaults():
    config = ConfigManager.load_config()
    assert 'camera' in config
    assert config['camera']['fps'] == 30

def test_create_from_args(tmp_path):
    output_dir = tmp_path / 'output'
    config = ConfigManager.create_config_from_args(output_dir, 20, 60)
    assert config['camera']['recording_time'] == 20
    assert config['camera']['fps'] == 60
    assert config['output']['base_path'] == str(output_dir)
