import pytest
from unittest.mock import MagicMock
import sys

@pytest.fixture(autouse=True)
def mock_depthai(monkeypatch):
    mock_dai = MagicMock()
    monkeypatch.setitem(sys.modules, 'depthai', mock_dai)
    return mock_dai

@pytest.fixture(autouse=True)
def mock_cv2(monkeypatch):
    mock_cv = MagicMock()
    monkeypatch.setitem(sys.modules, 'cv2', mock_cv)
    return mock_cv
