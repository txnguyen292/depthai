import pytest
from unittest.mock import MagicMock, patch
from src.core.recorder import OakDCamera

@pytest.fixture
def mock_config(tmp_path):
    return {
        'camera': {'rgb_resolution': [1280, 800], 'fps': 30, 'recording_time': 10},
        'output': {'base_path': str(tmp_path), 'rgb_filename': 'rgb.mp4', 'depth_filename': 'depth.mp4'},
        'depth': {'colormap': 'COLORMAP_JET', 'normalize': True, 'equalize_hist': True}
    }

def test_init(mock_config):
    with patch('src.core.recorder.dai.Pipeline') as mock_pipeline:
        recorder = OakDCamera(mock_config)
        assert recorder.pipeline is not None
        mock_pipeline.assert_called()
