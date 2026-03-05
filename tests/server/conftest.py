"""Shared fixtures for server tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from llm_punctuator.server import app
from llm_punctuator.server.app import ModelStatus


@pytest.fixture
def mock_punctuator() -> MagicMock:
    """A mock punctuator that avoids loading a real model."""
    mock = MagicMock()
    mock.add_punctuation.return_value = "你好，世界。"
    return mock


@pytest.fixture
def client(mock_punctuator: MagicMock) -> TestClient:
    """Test client with a loaded mock punctuator."""
    app.state.punctuator = mock_punctuator
    app.state.model_status = ModelStatus.ready
    return TestClient(app)


@pytest.fixture
def client_no_model() -> TestClient:
    """Test client with no model loaded."""
    app.state.punctuator = None
    app.state.model_status = ModelStatus.failed
    return TestClient(app, raise_server_exceptions=False)
