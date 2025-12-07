import pytest
from unittest.mock import MagicMock, patch
from src.core.base import OakDBase
import numpy as np

@pytest.fixture
def mock_config(tmp_path):
    return {
        'camera': {'rgb_resolution': [1280, 800], 'fps': 30, 'recording_time': 10},
        'output': {'base_path': str(tmp_path), 'rgb_filename': 'rgb.mp4', 'depth_filename': 'depth.mp4'},
        'depth': {'colormap': 'COLORMAP_JET', 'normalize': True, 'equalize_hist': True}
    }

def test_init(mock_config):
    base = OakDBase(mock_config)
    assert base.fps == 30
    assert base.recording_time == 10

def test_setup_output_directory(mock_config, tmp_path):
    base = OakDBase(mock_config)
    assert (tmp_path / 'data').exists()

def test_add_timestamp(mock_config):
    base = OakDBase(mock_config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # We mock cv2.putText to verify it gets called
    with patch('src.core.base.cv2.putText') as mock_put_text:
        base.add_timestamp(frame)
        mock_put_text.assert_called_once()

def test_process_depth_frame(mock_config):
    base = OakDBase(mock_config)
    depth_frame = np.zeros((100, 100), dtype=np.uint8)
    
    # Mock the cv2 functions used in process_depth_frame
    with patch('src.core.base.cv2') as mock_cv2:
        # Setup return values for chained calls
        mock_cv2.normalize.return_value = depth_frame
        mock_cv2.equalizeHist.return_value = depth_frame
        mock_cv2.applyColorMap.return_value = depth_frame
        mock_cv2.resize.return_value = depth_frame
        mock_cv2.COLORMAP_JET = 2 # Mock the constant
        
        base.process_depth_frame(depth_frame)
        
        # Verify calls based on config
        mock_cv2.normalize.assert_called_once()
        mock_cv2.equalizeHist.assert_called_once()
        mock_cv2.applyColorMap.assert_called_once()
        mock_cv2.resize.assert_called_once()
