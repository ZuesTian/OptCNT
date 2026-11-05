"""Persistent, lease-based comparison jobs shared by server and WSL2 workers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from ..core.comparison import compute_group_comparison_tests, summarize_group_results
from ..core.workflow import (
    BatchSettings,
    DetectionSettings,
    PreprocessSettings,
    analyze_image_path,
)
from .serialization import json_safe


ENGINE_VERSION = "optcnt-comparison-2026.11.05"
ACTIVE_STATUSES = ("leased", "running")
TERMINAL_STATUSES = ("complete", "failed")


def batch_settings_dict(settings: BatchSettings) -> dict:
    return dataclasses.asdict(settings)


def batch_settings_from_dict(payload: dict) -> BatchSettings:
    payload = payload or {}
    return BatchSettings(
        preprocess=PreprocessSettings(**(payload.get("preprocess") or {})),
        detection=DetectionSettings(**(payload.get("detection") or {})),
        scale_um=float(payload.get("scale_um", 10.0)),
        manual_scale_pixels=float(payload.get("manual_scale_pixels", 0.0)),
        center_fraction=float(payload.get("center_fraction", 0.75)),
        recognize_scale_text=bool(payload.get("recognize_scale_text", False)),
    )


class ComparisonJob:
    """Read-only job facade retained for the existing API layer."""

    def __init__(self, store: "ComparisonJobStore", job_id: str):
        self._store = store
        self.id = str(job_id)

    def snapshot(self) -> dict:
        return self._store.snapshot(self.id)

    @property
    def status(self) -> str:
        return str(self.snapshot()["status"])

    @property
    def total(self) -> int:
        return int(self.snapshot()["total"])

    @property
    def result(self) -> Optional[dict]:
        return self.snapshot().get("result")


class ComparisonJobStore:
    """SQLite-backed queue with expiring leases and a low-concurrency fallback worker."""

    def __init__(
        self,
        *,
        data_root: Path,
        ttl_seconds: int = 3600,
        max_jobs: int = 8,
        workers: int = 1,
        lease_seconds: int = 120,
        heartbeat_timeout_seconds: int = 20,
        poll_seconds: float = 2.0,
        engine_version: str = ENGINE_VERSION,
        clock: Callable[[], float] = time.time,
    ):
        self.data_root = Path(data_root).resolve()
        self.jobs_root = self.data_root / "comparison-jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.jobs_root / "queue.sqlite3"
        self.ttl_seconds = max(300, int(ttl_seconds))
        self.max_jobs = max(1, int(max_jobs))
        self.server_workers = max(0, int(workers))
        self.lease_seconds = max(30, int(lease_seconds))
        self.heartbeat_timeout_seconds = max(10, int(heartbeat_timeout_seconds))
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.engine_version = str(engine_version)
        self._clock = clock
        self._db_lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    touched_at REAL NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    work_dir TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_kind TEXT,
                    lease_token_hash TEXT,
                    lease_expires_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    completed_by TEXT,
                    engine_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_files (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    file_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY(job_id, file_id)
                );
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    last_seen REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_workers_kind_seen
                    ON workers(kind, last_seen);
                """
            )
        self.recover_expired_leases()

    def create(
        self,
        work_dir: Path,
        base_paths: List[str],
        experiment_paths: List[str],
        settings: BatchSettings,
    ) -> ComparisonJob:
        work_dir = Path(work_dir).resolve()
        if work_dir.parent != self.jobs_root:
            raise ValueError("任务目录必须位于持久任务根目录")
        now = self._clock()
        job_id = work_dir.name
        rows = []
        for group_name, paths in (("base", base_paths), ("experiment", experiment_paths)):
            for index, raw_path in enumerate(paths):
                path = Path(raw_path).resolve()
                if work_dir not in path.parents or not path.is_file():
                    raise ValueError("任务文件不在任务目录内")
                relative_path = path.relative_to(work_dir).as_posix()
                rows.append((
                    job_id,
                    f"{group_name}-{index:03d}",
                    group_name,
                    relative_path,
                    int(path.stat().st_size),
                    _file_sha256(path),
                ))
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'leased', 'running')"
            ).fetchone()[0]
            if int(active) >= self.max_jobs:
                connection.rollback()
                raise RuntimeError("服务器对比任务已满，请稍后重试")
            connection.execute(
                """
                INSERT INTO jobs(
                    id, status, current, total, message, created_at, touched_at,
                    work_dir, settings_json, engine_version
                ) VALUES (?, 'queued', 0, ?, '等待计算节点', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    len(rows),
                    now,
                    now,
                    str(work_dir),
                    json.dumps(batch_settings_dict(settings), ensure_ascii=False),
                    self.engine_version,
                ),
            )
            connection.executemany(
                """
                INSERT INTO job_files(
                    job_id, file_id, group_name, relative_path, size_bytes, sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
        return ComparisonJob(self, job_id)

    def new_work_dir(self) -> Path:
        for _ in range(10):
            path = self.jobs_root / uuid.uuid4().hex
            try:
                path.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return path
        raise RuntimeError("无法创建唯一任务目录")

    def get(self, job_id: str) -> ComparisonJob:
        self._row(job_id)
        return ComparisonJob(self, job_id)

    def snapshot(self, job_id: str, *, touch: bool = True) -> dict:
        now = self._clock()
        with self._db_lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if touch:
                connection.execute("UPDATE jobs SET touched_at = ? WHERE id = ?", (now, str(job_id)))
        result = json.loads(row["result_json"]) if row["result_json"] else None
        total = int(row["total"])
        return {
            "job_id": str(row["id"]),
            "status": str(row["status"]),
            "current": int(row["current"]),
            "total": total,
            "progress": int(row["current"]) / total if total else 0.0,
            "message": str(row["message"]),
            "error": row["error"],
            "result": result,
            "worker": row["completed_by"] or row["lease_owner"],
            "worker_kind": row["lease_kind"],
            "attempts": int(row["attempts"]),
        }

    def status(self) -> dict:
        self.recover_expired_leases()
        with self._connect() as connection:
            jobs = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
            active = int(connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'leased', 'running')"
            ).fetchone()[0])
            wsl = self._healthy_wsl_count(connection, self._clock())
        return {
            "jobs": jobs,
            "active_jobs": active,
            "max_jobs": self.max_jobs,
            "healthy_wsl_workers": wsl,
            "fallback_workers": self.server_workers,
            "queue_backend": "sqlite-lease",
        }

    def heartbeat(
        self,
        worker_id: str,
        *,
        kind: str,
        engine_version: str,
        capacity: int,
        metadata: Optional[dict] = None,
    ) -> dict:
        worker_id = _worker_id(worker_id)
        kind = str(kind).strip().lower()
        if kind != "wsl2":
            raise ValueError("外部计算节点类型必须为 wsl2")
        if str(engine_version) != self.engine_version:
            raise ValueError("计算节点分析引擎版本不一致")
        now = self._clock()
        capacity = max(1, min(64, int(capacity)))
        with self._db_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers(id, kind, engine_version, capacity, last_seen, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    engine_version=excluded.engine_version,
                    capacity=excluded.capacity,
                    last_seen=excluded.last_seen,
                    metadata_json=excluded.metadata_json
                """,
                (
                    worker_id,
                    kind,
                    self.engine_version,
                    capacity,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return {
            "worker_id": worker_id,
            "kind": kind,
            "engine_version": self.engine_version,
            "capacity": capacity,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "lease_seconds": self.lease_seconds,
        }

    def lease(self, worker_id: str, *, kind: str, engine_version: str) -> Optional[dict]:
        worker_id = _worker_id(worker_id)
        kind = str(kind).strip().lower()
        if str(engine_version) != self.engine_version:
            raise ValueError("计算节点分析引擎版本不一致")
        now = self._clock()
        raw_token = secrets.token_urlsafe(32)
        token_hash = _token_hash(raw_token)
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_leases(connection, now)
            if kind == "wsl2":
                worker = connection.execute(
                    "SELECT * FROM workers WHERE id = ? AND kind = 'wsl2'",
                    (worker_id,),
                ).fetchone()
                if (
                    worker is None or
                    worker["engine_version"] != self.engine_version or
                    float(worker["last_seen"]) < now - self.heartbeat_timeout_seconds
                ):
                    connection.rollback()
                    raise ValueError("计算节点心跳无效或已过期")
                active = int(connection.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                    WHERE lease_owner = ? AND status IN ('leased', 'running')
                      AND lease_expires_at > ?
                    """,
                    (worker_id, now),
                ).fetchone()[0])
                if active >= int(worker["capacity"]):
                    connection.commit()
                    return None
            elif kind == "server":
                if self._healthy_wsl_count(connection, now) > 0:
                    connection.commit()
                    return None
            else:
                connection.rollback()
                raise ValueError("未知计算节点类型")

            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' AND engine_version = ? "
                "ORDER BY created_at, id LIMIT 1",
                (self.engine_version,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            expires_at = now + self.lease_seconds
            updated = connection.execute(
                """
                UPDATE jobs SET
                    status='leased', message=?, touched_at=?, lease_owner=?, lease_kind=?,
                    lease_token_hash=?, lease_expires_at=?, attempts=attempts+1, error=NULL
                WHERE id=? AND status='queued'
                """,
                (
                    "已分配至 WSL2" if kind == "wsl2" else "由服务器兜底计算",
                    now,
                    worker_id,
                    kind,
                    token_hash,
                    expires_at,
                    row["id"],
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                return None
            files = connection.execute(
                "SELECT * FROM job_files WHERE job_id = ? ORDER BY group_name, file_id",
                (row["id"],),
            ).fetchall()
            connection.commit()
        return {
            "job_id": str(row["id"]),
            "lease_token": raw_token,
            "lease_expires_at": expires_at,
            "engine_version": self.engine_version,
            "settings": json.loads(row["settings_json"]),
            "files": [
                {
                    "file_id": str(item["file_id"]),
                    "group": str(item["group_name"]),
                    "filename": Path(str(item["relative_path"])).name,
                    "size_bytes": int(item["size_bytes"]),
                    "sha256": str(item["sha256"]),
                }
                for item in files
            ],
        }

    def renew(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        *,
        current: Optional[int] = None,
        message: Optional[str] = None,
    ) -> dict:
        now = self._clock()
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_lease(connection, job_id, worker_id, lease_token, now)
            resolved_current = int(row["current"] if current is None else current)
            resolved_current = max(int(row["current"]), min(int(row["total"]), resolved_current))
            resolved_message = str(message or row["message"])[:240]
            connection.execute(
                """
                UPDATE jobs SET status='running', current=?, message=?, touched_at=?, lease_expires_at=?
                WHERE id=?
                """,
                (resolved_current, resolved_message, now, now + self.lease_seconds, str(job_id)),
            )
            connection.commit()
        return self.snapshot(job_id, touch=False)

    def file_path(self, job_id: str, file_id: str, worker_id: str, lease_token: str) -> Path:
        now = self._clock()
        with self._connect() as connection:
            row = self._require_lease(connection, job_id, worker_id, lease_token, now)
            file_row = connection.execute(
                "SELECT * FROM job_files WHERE job_id = ? AND file_id = ?",
                (str(job_id), str(file_id)),
            ).fetchone()
            if file_row is None:
                raise KeyError(file_id)
        work_dir = Path(row["work_dir"]).resolve()
        path = (work_dir / str(file_row["relative_path"])).resolve()
        if work_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(file_id)
        return path

    def complete(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        *,
        result: dict,
        representative_images: Dict[str, bytes],
    ) -> dict:
        safe_result = json_safe(result)
        if not isinstance(safe_result, dict) or not {"base", "experiment", "metrics"} <= set(safe_result):
            raise ValueError("计算结果结构无效")
        serialized_result = json.dumps(
            safe_result,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized_result.encode("utf-8")) > 32 * 1024 * 1024:
            raise ValueError("计算结果超过大小限制")
        if any(len(content) > 12 * 1024 * 1024 for content in representative_images.values()):
            raise ValueError("代表图超过大小限制")
        now = self._clock()
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
            if existing is None:
                connection.rollback()
                raise KeyError(job_id)
            if existing["status"] == "complete" and existing["completed_by"] == _worker_id(worker_id):
                connection.commit()
                return self.snapshot(job_id, touch=False)
            row = self._require_lease(connection, job_id, worker_id, lease_token, now)
            work_dir = Path(row["work_dir"]).resolve()
            artifact_dir = work_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for group in ("base", "experiment"):
                content = representative_images.get(group) or b""
                if content:
                    (artifact_dir / f"representative-{group}.jpg").write_bytes(content)
            connection.execute(
                """
                UPDATE jobs SET
                    status='complete', current=total, message='对比分析完成', touched_at=?,
                    result_json=?, error=NULL, completed_by=?, lease_token_hash=NULL,
                    lease_expires_at=NULL
                WHERE id=?
                """,
                (
                    now,
                    serialized_result,
                    _worker_id(worker_id),
                    str(job_id),
                ),
            )
            connection.commit()
        _remove_raw_inputs(work_dir)
        return self.snapshot(job_id, touch=False)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        error: str,
        *,
        retryable: bool = True,
    ) -> dict:
        now = self._clock()
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_lease(connection, job_id, worker_id, lease_token, now)
            should_retry = bool(retryable and int(row["attempts"]) < 3)
            connection.execute(
                """
                UPDATE jobs SET status=?, message=?, error=?, touched_at=?, lease_owner=NULL,
                    lease_kind=NULL, lease_token_hash=NULL, lease_expires_at=NULL
                WHERE id=?
                """,
                (
                    "queued" if should_retry else "failed",
                    "等待其他计算节点重试" if should_retry else "对比分析失败",
                    str(error)[:2000],
                    now,
                    str(job_id),
                ),
            )
            connection.commit()
        return self.snapshot(job_id, touch=False)

    def representative_image(self, job_id: str, group: str) -> bytes:
        row = self._row(job_id)
        path = Path(row["work_dir"]).resolve() / "artifacts" / f"representative-{group}.jpg"
        if not path.is_file():
            raise KeyError(group)
        return path.read_bytes()

    def recover_expired_leases(self) -> int:
        now = self._clock()
        with self._db_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recovered = self._recover_expired_leases(connection, now)
            connection.commit()
        return recovered

    def cleanup(self) -> int:
        cutoff = self._clock() - self.ttl_seconds
        with self._db_lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, work_dir FROM jobs WHERE status IN ('complete', 'failed') AND touched_at < ?",
                (cutoff,),
            ).fetchall()
            connection.execute("BEGIN IMMEDIATE")
            for row in rows:
                connection.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            connection.commit()
        for row in rows:
            shutil.rmtree(Path(row["work_dir"]), ignore_errors=True)
        return len(rows)

    def start(self) -> None:
        if self._threads or self.server_workers <= 0:
            return
        self._stop.clear()
        for index in range(self.server_workers):
            thread = threading.Thread(
                target=self._server_worker_loop,
                args=(f"server-{index + 1}",),
                name=f"optcnt-fallback-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=max(2.0, self.poll_seconds * 2.0))
        self._threads.clear()

    def _server_worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                lease = self.lease(worker_id, kind="server", engine_version=self.engine_version)
                if lease is None:
                    self.cleanup()
                    self._stop.wait(self.poll_seconds)
                    continue
                self._execute_local_lease(worker_id, lease)
            except Exception:
                self._stop.wait(self.poll_seconds)

    def _execute_local_lease(self, worker_id: str, lease: dict) -> None:
        job_id = lease["job_id"]
        token = lease["lease_token"]
        row = self._row(job_id)
        work_dir = Path(row["work_dir"])

        def progress(current: int, message: str) -> None:
            self.renew(job_id, worker_id, token, current=current, message=message)

        try:
            result, images = run_comparison_work(
                work_dir,
                batch_settings_from_dict(lease["settings"]),
                progress=progress,
                job_id=job_id,
            )
            self.complete(
                job_id,
                worker_id,
                token,
                result=result,
                representative_images=images,
            )
        except Exception as exc:
            try:
                self.fail(job_id, worker_id, token, str(exc), retryable=False)
            except Exception:
                pass

    def _row(self, job_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _require_lease(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        worker_id: str,
        lease_token: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if row["status"] not in ACTIVE_STATUSES:
            raise PermissionError("任务当前没有有效租约")
        if row["lease_owner"] != _worker_id(worker_id):
            raise PermissionError("任务租约不属于此计算节点")
        if not secrets.compare_digest(str(row["lease_token_hash"] or ""), _token_hash(lease_token)):
            raise PermissionError("任务租约令牌无效")
        if float(row["lease_expires_at"] or 0.0) <= now:
            raise PermissionError("任务租约已过期")
        return row

    def _recover_expired_leases(self, connection: sqlite3.Connection, now: float) -> int:
        return connection.execute(
            """
            UPDATE jobs SET status='queued', message='计算节点失联，等待重新分配',
                lease_owner=NULL, lease_kind=NULL, lease_token_hash=NULL,
                lease_expires_at=NULL, touched_at=?
            WHERE status IN ('leased', 'running') AND lease_expires_at <= ?
            """,
            (now, now),
        ).rowcount

    def _healthy_wsl_count(self, connection: sqlite3.Connection, now: float) -> int:
        return int(connection.execute(
            """
            SELECT COUNT(*) FROM workers
            WHERE kind='wsl2' AND engine_version=? AND last_seen >= ?
            """,
            (self.engine_version, now - self.heartbeat_timeout_seconds),
        ).fetchone()[0])


def run_comparison_work(
    work_dir: Path,
    settings: BatchSettings,
    *,
    progress: Optional[Callable[[int, str], None]] = None,
    job_id: str = "job",
) -> tuple[dict, Dict[str, bytes]]:
    work_dir = Path(work_dir)
    base_paths = [str(path) for path in sorted((work_dir / "base").iterdir()) if path.is_file()]
    experiment_paths = [
        str(path) for path in sorted((work_dir / "experiment").iterdir()) if path.is_file()
    ]
    current = 0

    def analyze(paths: List[str], label: str) -> tuple[List[dict], List[str]]:
        nonlocal current
        results: List[dict] = []
        failures: List[str] = []
        for path in paths:
            try:
                results.append(analyze_image_path(path, settings, include_visualization=False))
            except Exception as exc:
                failures.append(f"{Path(path).name}: {exc}")
            finally:
                current += 1
                if progress is not None:
                    progress(current, f"{label}: {Path(path).name}")
        return results, failures

    base_results, base_failures = analyze(base_paths, "base组")
    experiment_results, experiment_failures = analyze(experiment_paths, "实验组")
    if not base_results or not experiment_results:
        raise ValueError("base组或实验组没有成功分析的图像")

    base = summarize_group_results("base组", base_results, group_role="base")
    experiment = summarize_group_results("实验组", experiment_results, group_role="experiment")
    tests = compute_group_comparison_tests(base, experiment)
    result = {
        "base": _strip_internal_results(base),
        "experiment": _strip_internal_results(experiment),
        "tests": tests,
        "metrics": _comparison_metrics(base, experiment, tests),
        "failures": {"base": base_failures, "experiment": experiment_failures},
        "representative_urls": {
            "base": f"/api/v1/comparisons/{job_id}/representatives/base",
            "experiment": f"/api/v1/comparisons/{job_id}/representatives/experiment",
        },
    }
    images = {
        "base": _representative_image(base, base_results, settings),
        "experiment": _representative_image(experiment, experiment_results, settings),
    }
    return result, images


def _representative_image(group: dict, results: List[dict], settings: BatchSettings) -> bytes:
    details = group.get("file_details") or []
    if not details:
        return b""
    uniformity = group["spatial_stats"]["uniformity_score"]
    shadow = group["spatial_stats"]["shadow_aggregation_score"]
    representative = min(
        details,
        key=lambda item: (
            abs(item["uniformity_score"] - uniformity["mean"]) / max(uniformity["std"], 1.0)
            + abs(item["shadow_aggregation_score"] - shadow["mean"]) / max(shadow["std"], 1.0)
        ),
    )
    raw = next(item for item in results if item["name"] == representative["name"])
    visual = analyze_image_path(raw["path"], settings, include_visualization=True)["visualization"]
    ok, encoded = cv2.imencode(".jpg", visual, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes() if ok else b""


def _strip_internal_results(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "results"}


def _comparison_metrics(base: dict, experiment: dict, tests: dict) -> List[dict]:
    specs = [
        ("count", "总 CNT 数量", "count_stats", "仅呈现数量差异"),
        ("dispersed_ratio", "分散比例", "dispersed_ratio_stats", "越高越好"),
        ("grid_density_cv", "网格 CV", "grid_density_cv_stats", "越低越均匀"),
        ("agglomerated_area_ratio", "团聚面积占比", "agglomerated_area_ratio_stats", "越低越好"),
        ("width_p90_um", "P90 宽度", "width_p90_um_stats", "越低越细"),
        ("skeleton_length_mean_um", "平均骨架长度", "skeleton_length_mean_stats", "补充指标"),
        ("uniformity_score", "综合均匀性", "uniformity_score_stats", "补充指标"),
        ("hybrid_score", "混合评分", "hybrid_score_stats", "补充指标"),
    ]
    return [
        {
            "key": key,
            "label": label,
            "direction": direction,
            "base": base.get(stats_key) or {},
            "experiment": experiment.get(stats_key) or {},
            "t_pvalue": (tests.get(key) or {}).get("t_pvalue"),
            "mw_pvalue": (tests.get(key) or {}).get("mw_pvalue"),
        }
        for key, label, stats_key, direction in specs
    ]


def _worker_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or len(normalized) > 80 or not all(char.isalnum() or char in "-_." for char in normalized):
        raise ValueError("计算节点 ID 无效")
    return normalized


def _token_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_raw_inputs(work_dir: Path) -> None:
    for name in ("base", "experiment"):
        shutil.rmtree(Path(work_dir) / name, ignore_errors=True)
