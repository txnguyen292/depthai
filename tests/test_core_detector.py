import pytest
from unittest.mock import MagicMock, patch
from src.core.detector import OakDObjectDetectionApp

def test_init():
    with patch('src.core.detector.dai.Pipeline') as mock_pipeline:
        app = OakDObjectDetectionApp()
        assert app.pipeline is not None
        mock_pipeline.assert_called()
