from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.analyzer_core import CNTAnalyzer
from src.core.workflow import DetectionSettings, PreprocessSettings, run_analysis
from src.webapp.app import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPTCNT_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(web_dir=tmp_path))


def _sample_jpeg() -> bytes:
    paths = sorted((Path(__file__).parents[1] / "DATA").rglob("*.jpg"))
    if paths:
        return paths[0].read_bytes()
    image = np.full((540, 800, 3), 235, dtype=np.uint8)
    for offset in (0, 36, 72):
        cv2.line(image, (125, 110 + offset), (400, 185 + offset), (30, 30, 30), 3)
    cv2.rectangle(image, (600, 450), (700, 455), (20, 20, 20), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _create_session(client: TestClient, filename: str = "sample.jpg") -> tuple[str, bytes]:
    payload = _sample_jpeg()
    response = client.post(
        "/api/v1/sessions",
        files={"file": (filename, payload, "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    return response.json()["session_id"], payload


def test_original_jpeg_is_served_without_png_reencoding(client: TestClient):
    session_id, payload = _create_session(client)

    response = client.get(f"/api/v1/sessions/{session_id}/images/original?v=0")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "immutable" in response.headers["cache-control"]
    assert response.content == payload


def test_health_describes_full_shared_engine(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["analysis_engine"] == "CNTAnalyzer"
    assert response.json()["queue_backend"] == "sqlite-lease"


def test_worker_api_is_closed_when_token_is_not_configured(client: TestClient):
    response = client.post(
        "/api/v1/workers/heartbeat",
        json={
            "worker_id": "wsl-test",
            "kind": "wsl2",
            "engine_version": "optcnt-comparison-2026.08.13",
            "capacity": 1,
        },
    )

    assert response.status_code == 503


def test_comparison_rejects_batch_over_total_upload_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPTCNT_MAX_BATCH_TOTAL_MB", "0.001")
    limited_client = TestClient(create_app(web_dir=tmp_path))
    image = np.random.default_rng(42).integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    payload = encoded.tobytes()

    response = limited_client.post(
        "/api/v1/comparisons",
        files=[
            ("base_files", ("base.png", payload, "image/png")),
            ("experiment_files", ("experiment.png", payload, "image/png")),
        ],
        data={"settings": "{}"},
    )

    assert response.status_code == 413
    assert "整批图像" in response.json()["detail"]


def test_session_supports_scale_roi_preprocess_and_exports(client: TestClient):
    session_id, _ = _create_session(client, "中文图像.jpg")

    roi_response = client.post(
        f"/api/v1/sessions/{session_id}/rois",
        json={"name": "复核区", "x": 120, "y": 90, "width": 320, "height": 240},
    )
    assert roi_response.status_code == 201, roi_response.text

    scale_response = client.post(
        f"/api/v1/sessions/{session_id}/scale",
        json={
            "pixels": 100,
            "micrometers": 10,
            "start": {"x": 600, "y": 450},
            "end": {"x": 700, "y": 450},
        },
    )
    assert scale_response.status_code == 200, scale_response.text
    assert scale_response.json()["scale_status"]["source"] == "manual"

    preview_response = client.post(
        f"/api/v1/sessions/{session_id}/preprocess",
        json={"roi_name": "复核区", "generate_skeleton": True},
    )
    assert preview_response.status_code == 200, preview_response.text
    assert client.get(f"/api/v1/sessions/{session_id}/images/binary").status_code == 200
    assert client.get(f"/api/v1/sessions/{session_id}/images/skeleton-preview").status_code == 200

    analysis_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze",
        json={
            "roi_name": "复核区",
            "preprocess": {"blur_kernel": 9, "adaptive_block": 11, "adaptive_c": 3},
            "detection": {
                "min_length_um": 5,
                "max_length_um": 1000,
                "min_slenderness": 4,
                "detection_profile": "balanced",
                "split_mode": "off",
                "merge_distance_px": 0,
            },
        },
    )
    assert analysis_response.status_code == 200, analysis_response.text
    result = analysis_response.json()
    assert result["analysis_roi"]["name"] == "复核区"
    assert "core_metrics" in result
    assert "evaluation_framework" in result
    assert "measurement_count" in result
    assert "measurements_url" in result
    assert "particle_statistics" in result
    assert "particle_measurements" in result
    assert result["particle_statistics"]["count"] == len(result["particle_measurements"])
    for particle in result["particle_measurements"]:
        assert {"id", "centroid", "bbox", "area_um2", "equivalent_diameter_um", "confidence"} <= particle.keys()

    measurements_response = client.get(result["measurements_url"])
    assert measurements_response.status_code == 200, measurements_response.text
    assert len(measurements_response.json()["measurements"]) == result["measurement_count"]

    particle_view = client.get(f"/api/v1/sessions/{session_id}/images/particles")
    assert particle_view.status_code == 200, particle_view.text
    assert particle_view.headers["content-type"] == "image/jpeg"
    assert "immutable" in particle_view.headers["cache-control"]
    decoded_view = cv2.imdecode(np.frombuffer(particle_view.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded_view is not None

    session = client.app.state.sessions.get(session_id)
    cached_particle_view = session.view_cache["particles"]
    repeated_view = client.get(f"/api/v1/sessions/{session_id}/images/particles")
    assert repeated_view.content == cached_particle_view
    assert session.view_cache["particles"] is cached_particle_view

    for kind in ("json", "csv", "report", "overlay", "skeleton", "particles"):
        export = client.get(f"/api/v1/sessions/{session_id}/exports/{kind}")
        assert export.status_code == 200, (kind, export.text)
        assert export.content
        assert "中文图像" in unquote(export.headers["content-disposition"])
        if kind in {"overlay", "skeleton", "particles"}:
            assert export.headers["content-type"] == "image/png"


def test_auto_roi_name_after_deleting_a_region(client: TestClient):
    session_id, _ = _create_session(client)
    url = f"/api/v1/sessions/{session_id}/rois"
    bounds = {"x": 10, "y": 10, "width": 80, "height": 80}
    first, second = (client.post(url, json=bounds).json()["roi"]["name"] for _ in range(2))
    assert client.delete(f"{url}/{first}").status_code == 200
    response = client.post(url, json=bounds)
    assert response.status_code == 201
    assert response.json()["roi"]["name"] != second


def test_view_urls_change_when_derived_state_changes(client: TestClient):
    session_id, _ = _create_session(client)
    initial = client.get(f"/api/v1/sessions/{session_id}").json()["image_urls"]

    preview = client.post(
        f"/api/v1/sessions/{session_id}/preprocess",
        json={"blur_kernel": 9, "adaptive_block": 11, "adaptive_c": 3, "generate_skeleton": False},
    )
    assert preview.status_code == 200, preview.text
    preview_urls = preview.json()["image_urls"]

    assert preview_urls["original"] == initial["original"]
    assert preview_urls["binary"] != initial["binary"]
    assert "?v=1" in preview_urls["binary"]


def test_http_analysis_matches_direct_core_exactly(client: TestClient):
    session_id, payload = _create_session(client)
    request = {
        "preprocess": {
            "blur_kernel": 9,
            "adaptive_block": 11,
            "adaptive_c": 3,
            "bridge_strength": 0,
            "threshold_invert": True,
        },
        "detection": {
            "min_length_um": 5,
            "max_length_um": 1000,
            "min_slenderness": 4,
            "detection_profile": "balanced",
            "split_mode": "off",
            "merge_distance_px": 0,
        },
    }
    web_response = client.post(f"/api/v1/sessions/{session_id}/analyze", json=request)
    assert web_response.status_code == 200, web_response.text
    web = web_response.json()
    web_measurements = client.get(web["measurements_url"]).json()["measurements"]

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    direct_analyzer = CNTAnalyzer()
    direct_analyzer.set_image(image)
    direct_analyzer.apply_detected_scale(default_micrometers=10.0, recognize_text=True)
    direct = run_analysis(
        direct_analyzer,
        name="sample.jpg",
        preprocess=PreprocessSettings(**request["preprocess"]),
        detection=DetectionSettings(**request["detection"]),
    )

    assert web["stats"]["count"] == direct["stats"]["count"]
    assert web["stats"]["length_mean"] == pytest.approx(direct["stats"]["length_mean"])
    assert web["core_metrics"] == pytest.approx(direct["core_metrics"])
    assert len(web_measurements) == len(direct["measurements"])
    assert web["particle_statistics"] == pytest.approx(direct["particle_statistics"])
    assert len(web["particle_measurements"]) == len(direct["particle_measurements"])
    for web_item, direct_item in zip(web_measurements, direct["measurements"]):
        assert web_item["length_um"] == pytest.approx(direct_item.length_um)
        assert web_item["width_mean_um"] == pytest.approx(direct_item.width_mean_um)
        assert web_item["width_median_um"] == pytest.approx(direct_item.width_median_um)
    for web_item, direct_item in zip(web["particle_measurements"], direct["particle_measurements"]):
        assert web_item["area_um2"] == pytest.approx(direct_item.area_um2)
        assert web_item["equivalent_diameter_um"] == pytest.approx(direct_item.equivalent_diameter_um)
        assert web_item["confidence"] == pytest.approx(direct_item.confidence)
