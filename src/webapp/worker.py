"""Pull-based OptCNT comparison worker for a WSL2 compute node."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .jobs import ENGINE_VERSION, batch_settings_from_dict, run_comparison_work


class WorkerApi:
    def __init__(self, base_url: str, api_token: str, *, timeout: float = 60.0):
        self.base_url = str(base_url).rstrip("/")
        self.api_token = str(api_token)
        self.timeout = max(5.0, float(timeout))
        self.ssl_context = ssl.create_default_context()

    def json(self, method: str, path: str, payload: Optional[dict] = None) -> tuple[int, Optional[dict]]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-OptCNT-Worker-Token": self.api_token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                content = response.read()
                return response.status, json.loads(content) if content else None
        except urllib.error.HTTPError as exc:
            content = exc.read()
            detail = content.decode("utf-8", errors="replace")
            raise RuntimeError(f"Worker API {exc.code}: {detail}") from exc

    def download(self, path: str, destination: Path, *, worker_id: str, lease_token: str) -> None:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method="GET",
            headers={
                "X-OptCNT-Worker-Token": self.api_token,
                "X-OptCNT-Worker-Id": worker_id,
                "X-OptCNT-Lease-Token": lease_token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                destination.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Worker download {exc.code}: {detail}") from exc


class LeaseKeeper:
    def __init__(
        self,
        api: WorkerApi,
        worker_id: str,
        job_id: str,
        lease_token: str,
        *,
        interval: float,
        heartbeat,
    ):
        self.api = api
        self.worker_id = worker_id
        self.job_id = job_id
        self.lease_token = lease_token
        self.interval = max(5.0, float(interval))
        self.heartbeat = heartbeat
        self.current = 0
        self.message = "WSL2 正在计算"
        self.error: Optional[Exception] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="optcnt-lease-keeper", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, current: int, message: str) -> None:
        with self._lock:
            self.current = max(self.current, int(current))
            self.message = str(message)[:240]
        self.renew()

    def renew(self) -> None:
        with self._lock:
            current = self.current
            message = self.message
        self.api.json(
            "POST",
            f"/api/v1/workers/jobs/{self.job_id}/renew",
            {
                "worker_id": self.worker_id,
                "lease_token": self.lease_token,
                "current": current,
                "message": message,
            },
        )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.heartbeat()
                self.renew()
            except Exception as exc:
                self.error = exc
                return


class WslComparisonWorker:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        worker_id: str,
        capacity: int = 1,
        poll_seconds: float = 3.0,
        work_root: Optional[Path] = None,
    ):
        self.api = WorkerApi(base_url, api_token)
        self.worker_id = str(worker_id)
        self.capacity = max(1, min(64, int(capacity)))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.work_root = Path(work_root or tempfile.gettempdir()) / "optcnt-worker"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.lease_seconds = 120
        self.heartbeat_timeout_seconds = 20

    def run(self, *, once: bool = False) -> int:
        while not self.stop_event.is_set():
            try:
                heartbeat = self._heartbeat()
                self.lease_seconds = int(heartbeat.get("lease_seconds", 120))
                self.heartbeat_timeout_seconds = int(
                    heartbeat.get("heartbeat_timeout_seconds", 20)
                )
                status, lease = self.api.json(
                    "POST",
                    "/api/v1/workers/lease",
                    {
                        "worker_id": self.worker_id,
                        "kind": "wsl2",
                        "engine_version": ENGINE_VERSION,
                    },
                )
                if status == 204 or lease is None:
                    if once:
                        return 0
                    self.stop_event.wait(self.poll_seconds)
                    continue
                self._execute(lease)
                if once:
                    return 0
            except Exception as exc:
                print(f"OptCNT Worker: {exc}", flush=True)
                if once:
                    return 1
                self.stop_event.wait(min(30.0, self.poll_seconds * 3.0))
        return 0

    def _heartbeat(self) -> dict:
        _, response = self.api.json(
            "POST",
            "/api/v1/workers/heartbeat",
            {
                "worker_id": self.worker_id,
                "kind": "wsl2",
                "engine_version": ENGINE_VERSION,
                "capacity": self.capacity,
                "metadata": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
            },
        )
        return response or {}

    def _execute(self, lease: dict) -> None:
        job_id = str(lease["job_id"])
        lease_token = str(lease["lease_token"])
        job_dir = self.work_root / job_id
        shutil.rmtree(job_dir, ignore_errors=True)
        (job_dir / "base").mkdir(parents=True)
        (job_dir / "experiment").mkdir(parents=True)
        keeper = LeaseKeeper(
            self.api,
            self.worker_id,
            job_id,
            lease_token,
            interval=max(
                5.0,
                min(self.lease_seconds / 3.0, self.heartbeat_timeout_seconds / 3.0),
            ),
            heartbeat=self._heartbeat,
        )
        keeper.start()
        try:
            for item in lease.get("files") or []:
                group = str(item["group"])
                if group not in {"base", "experiment"}:
                    raise ValueError("任务包含未知文件组")
                raw_filename = str(item.get("filename") or "")
                filename = Path(raw_filename).name
                if not filename or filename != raw_filename:
                    raise ValueError("任务文件名无效")
                destination = job_dir / group / filename
                self.api.download(
                    f"/api/v1/workers/jobs/{job_id}/files/{urllib.parse.quote(str(item['file_id']))}",
                    destination,
                    worker_id=self.worker_id,
                    lease_token=lease_token,
                )
                if destination.stat().st_size != int(item["size_bytes"]):
                    raise ValueError("任务文件大小校验失败")
                if _sha256(destination) != str(item["sha256"]):
                    raise ValueError("任务文件 SHA-256 校验失败")
            result, representatives = run_comparison_work(
                job_dir,
                batch_settings_from_dict(lease["settings"]),
                progress=keeper.update,
                job_id=job_id,
            )
            keeper.close()
            self.api.json(
                "POST",
                f"/api/v1/workers/jobs/{job_id}/complete",
                {
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "result": result,
                    "representatives": {
                        group: base64.b64encode(content).decode("ascii")
                        for group, content in representatives.items()
                    },
                },
            )
        except Exception as exc:
            try:
                keeper.close()
            except Exception:
                pass
            try:
                self.api.json(
                    "POST",
                    f"/api/v1/workers/jobs/{job_id}/fail",
                    {
                        "worker_id": self.worker_id,
                        "lease_token": lease_token,
                        "error": str(exc),
                        "retryable": True,
                    },
                )
            except Exception:
                pass
            raise
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="OptCNT WSL2 comparison worker")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()
    base_url = os.environ.get("OPTCNT_SERVER_URL", "").strip()
    api_token = os.environ.get("OPTCNT_WORKER_API_TOKEN", "").strip()
    if not base_url or not api_token:
        parser.error("OPTCNT_SERVER_URL and OPTCNT_WORKER_API_TOKEN are required")
    worker = WslComparisonWorker(
        base_url=base_url,
        api_token=api_token,
        worker_id=os.environ.get("OPTCNT_WORKER_ID", f"wsl2-{socket.gethostname()}"),
        capacity=int(os.environ.get("OPTCNT_WORKER_CAPACITY", "1")),
        poll_seconds=float(os.environ.get("OPTCNT_WORKER_POLL_SECONDS", "3")),
        work_root=Path(os.environ.get("OPTCNT_WORKER_DATA_DIR", "/var/tmp")),
    )
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: worker.stop_event.set())
    return worker.run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
