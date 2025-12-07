import pytest
from unittest.mock import MagicMock, patch
from src.utils.device import check_connection_status

def test_check_connection_status():
    with patch('src.utils.device.dai.Device') as mock_device_cls:
        mock_device = MagicMock()
        mock_device_cls.return_value.__enter__.return_value = mock_device
        mock_device.getDeviceName.return_value = 'OAK-D'
        
        info = check_connection_status()
        assert info['device_name'] == 'OAK-D'
