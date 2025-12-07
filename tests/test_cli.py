from typer.testing import CliRunner
from src.cli import app
from unittest.mock import patch

runner = CliRunner()

def test_check_connection_command():
    with patch('src.cli.check_connection_status') as mock_check:
        mock_check.return_value = {
            'device_name': 'OAK-D',
            'usb_speed': 'SUPER',
            'connected_cameras': [],
            'stereo_pairs': []
        }
        result = runner.invoke(app, ['check-connection'])
        assert result.exit_code == 0
        assert 'Connected to device!' in result.stdout
