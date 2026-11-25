from __future__ import annotations

from pathlib import Path

import pytest

from src.core.workflow import BatchSettings
from src.webapp.jobs import ENGINE_VERSION, ComparisonJobStore
from src.webapp.app import create_app
from fastapi.testclient import TestClient


class MutableClock:
    def __init__(self, value: float = 1_000.0):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _create_job(store: ComparisonJobStore) -> str:
    work_dir = store.new_work_dir()
    base_dir = work_dir / "base"
    experiment_dir = work_dir / "experiment"
    base_dir.mkdir()
    experiment_dir.mkdir()
    base_path = base_dir / "000.png"
    experiment_path = experiment_dir / "000.png"
    base_path.write_bytes(b"base")
    experiment_path.write_bytes(b"experiment")
    job = store.create(
        work_dir,
        [str(base_path)],
        [str(experiment_path)],
        BatchSettings(),
    )
    return job.id


def _store(tmp_path: Path, clock: MutableClock) -> ComparisonJobStore:
    return ComparisonJobStore(
        data_root=tmp_path,
        workers=0,
        lease_seconds=30,
        heartbeat_timeout_seconds=10,
        clock=clock,
    )


def test_healthy_wsl_worker_has_priority_over_server_fallback(tmp_path: Path):
    clock = MutableClock()
    store = _store(tmp_path, clock)
    job_id = _create_job(store)

    store.heartbeat(
        "wsl-main",
        kind="wsl2",
        engine_version=ENGINE_VERSION,
        capacity=2,
    )

    assert store.lease("server-1", kind="server", engine_version=ENGINE_VERSION) is None
    lease = store.lease("wsl-main", kind="wsl2", engine_version=ENGINE_VERSION)

    assert lease is not None
    assert lease["job_id"] == job_id
    assert store.snapshot(job_id)["worker_kind"] == "wsl2"


def test_server_takes_over_only_after_lease_and_heartbeat_expire(tmp_path: Path):
    clock = MutableClock()
    store = _store(tmp_path, clock)
    job_id = _create_job(store)
    store.heartbeat(
        "wsl-main",
        kind="wsl2",
        engine_version=ENGINE_VERSION,
        capacity=1,
    )
    old_lease = store.lease("wsl-main", kind="wsl2", engine_version=ENGINE_VERSION)
    assert old_lease is not None

    clock.value += 31
    replacement = store.lease("server-1", kind="server", engine_version=ENGINE_VERSION)

    assert replacement is not None
    assert replacement["job_id"] == job_id
    assert store.snapshot(job_id)["attempts"] == 2
    with pytest.raises(PermissionError, match="不属于"):
        store.renew(
            job_id,
            "wsl-main",
            old_lease["lease_token"],
            current=1,
            message="迟到的 WSL2 进度",
        )


def test_server_waits_while_wsl_lease_is_valid_even_if_heartbeat_is_stale(tmp_path: Path):
    clock = MutableClock()
    store = _store(tmp_path, clock)
    _create_job(store)
    store.heartbeat(
        "wsl-main",
        kind="wsl2",
        engine_version=ENGINE_VERSION,
        capacity=1,
    )
    lease = store.lease("wsl-main", kind="wsl2", engine_version=ENGINE_VERSION)
    assert lease is not None

    clock.value += 11

    assert store.lease("server-1", kind="server", engine_version=ENGINE_VERSION) is None


def test_worker_version_mismatch_is_rejected(tmp_path: Path):
    clock = MutableClock()
    store = _store(tmp_path, clock)

    with pytest.raises(ValueError, match="版本不一致"):
        store.heartbeat(
            "wsl-main",
            kind="wsl2",
            engine_version="old-engine",
            capacity=1,
        )


def test_worker_http_protocol_requires_token_and_lease_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPTCNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPTCNT_WORKER_API_TOKEN", "test-worker-secret")
    monkeypatch.setenv("OPTCNT_JOB_WORKERS", "0")
    client = TestClient(create_app(web_dir=tmp_path / "missing-web"))
    store = client.app.state.jobs
    job_id = _create_job(store)
    heartbeat_payload = {
        "worker_id": "wsl-test",
        "kind": "wsl2",
        "engine_version": ENGINE_VERSION,
        "capacity": 1,
    }

    assert client.post("/api/v1/workers/heartbeat", json=heartbeat_payload).status_code == 401
    headers = {"X-OptCNT-Worker-Token": "test-worker-secret"}
    heartbeat = client.post("/api/v1/workers/heartbeat", json=heartbeat_payload, headers=headers)
    assert heartbeat.status_code == 200, heartbeat.text
    lease = client.post(
        "/api/v1/workers/lease",
        json={
            "worker_id": "wsl-test",
            "kind": "wsl2",
            "engine_version": ENGINE_VERSION,
        },
        headers=headers,
    )
    assert lease.status_code == 200, lease.text
    lease_payload = lease.json()
    first_file = lease_payload["files"][0]

    missing_credentials = client.get(
        f"/api/v1/workers/jobs/{job_id}/files/{first_file['file_id']}",
        headers=headers,
    )
    assert missing_credentials.status_code == 401
    file_response = client.get(
        f"/api/v1/workers/jobs/{job_id}/files/{first_file['file_id']}",
        headers={
            **headers,
            "X-OptCNT-Worker-Id": "wsl-test",
            "X-OptCNT-Lease-Token": lease_payload["lease_token"],
        },
    )
    assert file_response.status_code == 200, file_response.text
    assert len(file_response.content) == first_file["size_bytes"]

    complete = client.post(
        f"/api/v1/workers/jobs/{job_id}/complete",
        headers=headers,
        json={
            "worker_id": "wsl-test",
            "lease_token": lease_payload["lease_token"],
            "result": {"base": {}, "experiment": {}, "metrics": []},
            "representatives": {},
        },
    )
    assert complete.status_code == 200, complete.text
    assert complete.json()["status"] == "complete"
    assert complete.json()["worker"] == "wsl-test"
