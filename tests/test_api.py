"""Тесты HTTP-слоя.

Приложение поднимается целиком, с реальным lifespan, но на offline-провайдере:
сеть в тестах не нужна.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_health_reports_mode(client: TestClient) -> None:
    payload = client.get("/api/v1/health").json()

    assert payload["status"] == "ok"
    assert payload["provider"] == "offline"
    assert "ocr_available" in payload


def test_health_never_leaks_keys(client: TestClient) -> None:
    body = client.get("/api/v1/health").text
    assert "sk-" not in body and "api_key" not in body


def test_samples_are_listed(client: TestClient) -> None:
    samples = client.get("/api/v1/samples").json()

    assert len(samples) == 3
    assert all(item["name"].endswith(".pdf") for item in samples)


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Тендерный суммаризатор" in response.text


def test_openapi_is_generated(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/summarize" in paths


def test_summarize_sample_returns_structured_result(client: TestClient) -> None:
    response = client.post(
        "/api/v1/summarize/sample", params={"name": "tender-44fz-remont-krovli.pdf"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["price"]["amount"] == "12480350.00"
    assert payload["document"]["pages"] == 4
    assert payload["meta"]["provider"] == "offline"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_summarize_upload(client: TestClient, roof_pdf: bytes) -> None:
    response = client.post(
        "/api/v1/summarize",
        files={"file": ("документация.pdf", roof_pdf, "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["document"]["filename"] == "документация.pdf"


def test_request_id_header_is_set(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.headers["X-Request-Id"]
    assert response.headers["X-Process-Time-Ms"].isdigit()


def test_non_pdf_extension_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/summarize", files={"file": ("отчёт.docx", b"PK\x03\x04", "application/msword")}
    )

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"


def test_fake_pdf_is_rejected_by_signature(client: TestClient) -> None:
    """Расширению верить нельзя: проверяется сигнатура файла."""
    response = client.post(
        "/api/v1/summarize",
        files={"file": ("документ.pdf", b"not a pdf at all", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "pdf_parse_error"


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/summarize", files={"file": ("пусто.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 415


def test_oversized_upload_is_rejected(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()

    with TestClient(create_app()) as small_client:
        response = small_client.post(
            "/api/v1/summarize",
            files={"file": ("большой.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_missing_file_is_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/summarize")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_sample_path_traversal_is_blocked(client: TestClient) -> None:
    """Имя приходит из запроса: без обрезки до базовой части утечёт .env."""
    response = client.post("/api/v1/summarize/sample", params={"name": "../../.env"})

    assert response.status_code == 415
    assert "не найден" in response.json()["message"]


def test_unknown_route_returns_json_error(client: TestClient) -> None:
    response = client.get("/api/v1/не-существует")
    assert response.status_code == 404
    assert response.json()["code"] == "http_404"


def test_error_payload_carries_request_id(client: TestClient) -> None:
    response = client.post("/api/v1/summarize/sample", params={"name": "нет.pdf"})
    assert response.json()["request_id"]
