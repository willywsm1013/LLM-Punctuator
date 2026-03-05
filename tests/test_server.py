"""Tests for FastAPI server endpoints."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _make_mock_punctuator() -> MagicMock:
    """Create a mock punctuator that avoids loading a real model."""
    mock = MagicMock()
    mock.add_punctuation.return_value = "你好，世界。"
    return mock


def _make_client(mock_punctuator: MagicMock) -> TestClient:
    """Create a test client with mocked punctuator."""
    from llm_punctuator.server import app

    app.state.punctuator = mock_punctuator
    app.state.model_loaded = True
    return TestClient(app)


def test_health_healthy() -> None:
    mock = _make_mock_punctuator()
    client = _make_client(mock)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_unhealthy() -> None:
    from llm_punctuator.server import app

    app.state.punctuator = None
    app.state.model_loaded = False
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_info() -> None:
    mock = _make_mock_punctuator()
    client = _make_client(mock)
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "supported_languages" in data
    assert "version" in data
    assert data["supported_languages"] == ["zh", "en"]


def test_punctuate_zh() -> None:
    mock = _make_mock_punctuator()
    mock.add_punctuation.return_value = "你好，世界。"
    client = _make_client(mock)
    response = client.post("/api/v1/punctuate", json={"text": "你好世界"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "你好，世界。"
    assert data["language"] == "zh"
    mock.add_punctuation.assert_called_once_with(
        "你好世界", language="zh", chunk_size=50
    )


def test_punctuate_en() -> None:
    mock = _make_mock_punctuator()
    mock.add_punctuation.return_value = "hello, world."
    client = _make_client(mock)
    response = client.post(
        "/api/v1/punctuate",
        json={"text": "hello world", "language": "en", "chunk_size": 100},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "hello, world."
    assert data["language"] == "en"
    mock.add_punctuation.assert_called_once_with(
        "hello world", language="en", chunk_size=100
    )


def test_punctuate_empty_text() -> None:
    mock = _make_mock_punctuator()
    client = _make_client(mock)
    response = client.post("/api/v1/punctuate", json={"text": ""})
    assert response.status_code == 422


def test_punctuate_invalid_language() -> None:
    mock = _make_mock_punctuator()
    client = _make_client(mock)
    response = client.post(
        "/api/v1/punctuate", json={"text": "hello", "language": "fr"}
    )
    assert response.status_code == 422


def test_punctuate_model_not_loaded() -> None:
    from llm_punctuator.server import app

    app.state.punctuator = None
    app.state.model_loaded = False
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/v1/punctuate", json={"text": "hello"})
    assert response.status_code == 503
