"""Tests for FastAPI server endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


class TestHealth:
    def test_returns_200_when_model_loaded(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_returns_503_when_model_not_loaded(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestInfo:
    def test_returns_model_and_supported_languages(self, client: TestClient) -> None:
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert data["supported_languages"] == ["zh", "en"]
        assert "version" in data


class TestPunctuate:
    @pytest.mark.parametrize(
        ("text", "language", "chunk_size", "expected_text"),
        [
            ("你好世界", "zh", 50, "你好，世界。"),
            ("hello world", "en", 100, "hello, world."),
        ],
        ids=["zh_default_params", "en_custom_params"],
    )
    def test_returns_punctuated_text(
        self,
        client: TestClient,
        mock_punctuator: MagicMock,
        text: str,
        language: str,
        chunk_size: int,
        expected_text: str,
    ) -> None:
        mock_punctuator.add_punctuation.return_value = expected_text

        response = client.post(
            "/api/v1/punctuate",
            json={"text": text, "language": language, "chunk_size": chunk_size},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == expected_text
        assert data["language"] == language
        mock_punctuator.add_punctuation.assert_called_once_with(
            text, language=language, chunk_size=chunk_size
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"text": ""},
            {"text": "hello", "language": "fr"},
        ],
        ids=["empty_text", "unsupported_language"],
    )
    def test_rejects_invalid_input_with_422(
        self, client: TestClient, payload: dict
    ) -> None:
        response = client.post("/api/v1/punctuate", json=payload)

        assert response.status_code == 422

    def test_returns_503_when_model_not_loaded(self, client_no_model: TestClient) -> None:
        response = client_no_model.post("/api/v1/punctuate", json={"text": "hello"})

        assert response.status_code == 503
