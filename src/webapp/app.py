"""FastAPI application exposing the complete OptCNT analysis core."""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote
from typing import Annotated, Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..core.analyzer_core import CNTAnalyzer
from ..core.workflow import (
    BatchSettings,
    DetectionSettings,
    PreprocessSettings,
    analyze_image_path,
    make_roi,
    run_analysis,
)
from .jobs import ENGINE_VERSION, ComparisonJobStore, QueueFullError
from .serialization import (
    analysis_csv_bytes,
    analysis_json_bytes,
    analysis_report_bytes,
    comparison_csv_bytes,
    json_safe,
    public_analysis_result,
    public_measurements,
)
from .sessions import AnalysisSession, SessionStore

APP_VERSION = "2026.08.12"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"
PREVIEW_JPEG_QUALITY = 82
# Browser previews are downscaled to this longest edge before JPEG encoding;
# exports (PNG) always use the original resolution, so lossless quality is kept.
PREVIEW_MAX_EDGE = 1600
PREVIEW_CACHE_CONTROL = "private, max-age=1800, immutable"
ORIGINAL_PAYLOAD_CACHE_MAX_BYTES = 8 * 1024 * 1024


class PointRequest(BaseModel):
    x: float
    y: float


class ScaleRequest(BaseModel):
    pixels: float = Field(gt=0, le=100000)
    micrometers: float = Field(gt=0, le=10000)
    start: Optional[PointRequest] = None
    end: Optional[PointRequest] = None


class RoiRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    x: int
    y: int
    width: int
    height: int


class SuggestRequest(BaseModel):
    roi_name: Optional[str] = None
    detection_profile: Literal["precision", "balanced", "recall"] = "balanced"


class PreprocessRequest(BaseModel):
    roi_name: Optional[str] = None
    blur_kernel: int = Field(default=9, ge=1, le=31)
    adaptive_block: int = Field(default=11, ge=3, le=99)
    adaptive_c: int = Field(default=3, ge=-30, le=30)
    bridge_strength: int = Field(default=0, ge=0, le=10)
    threshold_invert: bool = True
    generate_skeleton: bool = True

    def settings(self) -> PreprocessSettings:
        return PreprocessSettings(
            blur_kernel=self.blur_kernel,
            adaptive_block=self.adaptive_block,
            adaptive_c=self.adaptive_c,
            bridge_strength=self.bridge_strength,
            threshold_invert=self.threshold_invert,
        )


class DetectionRequest(BaseModel):
    min_length_um: float = Field(default=5.0, ge=0, le=100000)
    max_length_um: float = Field(default=1000.0, gt=0, le=100000)
    min_slenderness: float = Field(default=4.0, ge=0, le=1000)
    detection_profile: Literal["precision", "balanced", "recall"] = "balanced"
    split_mode: Literal["off", "conservative", "aggressive"] = "off"
    merge_distance_px: float = Field(default=0.0, ge=0, le=20)

    def settings(self) -> DetectionSettings:
        return DetectionSettings(**self.model_dump())


class AnalyzeRequest(BaseModel):
    roi_name: Optional[str] = None
    preprocess: PreprocessRequest = Field(default_factory=PreprocessRequest)
    detection: DetectionRequest = Field(default_factory=DetectionRequest)


class BatchRequest(BaseModel):
    preprocess: PreprocessRequest = Field(default_factory=PreprocessRequest)
    detection: DetectionRequest = Field(default_factory=DetectionRequest)
    scale_um: float = Field(default=10.0, gt=0, le=10000)
    manual_scale_pixels: float = Field(default=0.0, ge=0, le=100000)
    center_fraction: float = Field(default=0.75, ge=0.1, le=1.0)
    recognize_scale_text: bool = False

    def settings(self) -> BatchSettings:
        return BatchSettings(
            preprocess=self.preprocess.settings(),
            detection=self.detection.settings(),
            scale_um=self.scale_um,
            manual_scale_pixels=self.manual_scale_pixels,
            center_fraction=self.center_fraction,
            recognize_scale_text=self.recognize_scale_text,
        )


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=80)
    kind: Literal["wsl2"] = "wsl2"
    engine_version: str
    capacity: int = Field(default=1, ge=1, le=64)
    metadata: dict = Field(default_factory=dict)


class WorkerLeaseRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=80)
    kind: Literal["wsl2"] = "wsl2"
    engine_version: str


class WorkerRenewRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=80)
    lease_token: str = Field(min_length=20, max_length=200)
    current: Optional[int] = Field(default=None, ge=0)
    message: Optional[str] = Field(default=None, max_length=240)


class WorkerCompleteRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=80)
    lease_token: str = Field(min_length=20, max_length=200)
    result: dict
    representatives: Dict[str, str] = Field(default_factory=dict)


class WorkerFailRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=80)
    lease_token: str = Field(min_length=20, max_length=200)
    error: str = Field(min_length=1, max_length=2000)
    retryable: bool = True


def create_app(*, web_dir: Optional[Path] = None) -> FastAPI:
    max_upload_bytes = int(float(os.getenv("OPTCNT_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
    max_batch_files = int(os.getenv("OPTCNT_MAX_BATCH_FILES", "20"))
    max_batch_total_bytes = int(float(os.getenv("OPTCNT_MAX_BATCH_TOTAL_MB", "200")) * 1024 * 1024)
    session_store = SessionStore(
        ttl_seconds=int(os.getenv("OPTCNT_SESSION_TTL_SECONDS", "1800")),
        max_sessions=int(os.getenv("OPTCNT_MAX_SESSIONS", "24")),
    )
    data_root = Path(os.getenv("OPTCNT_DATA_DIR", tempfile.gettempdir()))
    data_root.mkdir(parents=True, exist_ok=True)
    job_store = ComparisonJobStore(
        data_root=data_root,
        ttl_seconds=int(os.getenv("OPTCNT_JOB_TTL_SECONDS", "3600")),
        max_jobs=int(os.getenv("OPTCNT_MAX_JOBS", "8")),
        workers=int(os.getenv("OPTCNT_JOB_WORKERS", "1")),
        lease_seconds=int(os.getenv("OPTCNT_WORKER_LEASE_SECONDS", "120")),
        heartbeat_timeout_seconds=int(os.getenv("OPTCNT_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "20")),
        poll_seconds=float(os.getenv("OPTCNT_JOB_POLL_SECONDS", "2")),
    )
    worker_api_token = os.getenv("OPTCNT_WORKER_API_TOKEN", "").strip()

    app = FastAPI(
        title="OptCNT Full Analysis API",
        version=APP_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    app.state.sessions = session_store
    app.state.jobs = job_store

    @app.on_event("startup")
    def configure_runtime() -> None:
        cv2.setNumThreads(max(1, min(2, os.cpu_count() or 1)))
        try:
            cv2.ocl.setUseOpenCL(False)
        except AttributeError:
            pass
        job_store.start()

    @app.on_event("shutdown")
    def close_runtime() -> None:
        job_store.close()

    @app.get("/health")
    @app.get("/api/v1/health")
    def health(response: Response) -> dict:
        response.headers["Access-Control-Allow-Origin"] = "https://zuestian.github.io"
        return {
            "status": "ok",
            "service": "optcnt",
            "version": APP_VERSION,
            "analysis_engine": "CNTAnalyzer",
            "opencv": cv2.__version__,
            "skeletonization": "deterministic_morphology",
            **session_store.status(),
            **job_store.status(),
        }

    @app.get("/api/v1/config")
    def config() -> dict:
        return {
            "max_upload_mb": max_upload_bytes // (1024 * 1024),
            "max_batch_files_per_group": max_batch_files,
            "max_batch_total_mb": max_batch_total_bytes // (1024 * 1024),
            "session_ttl_seconds": session_store.ttl_seconds,
            "supported_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
            "capabilities": [
                "自动与手动比例尺",
                "多 ROI",
                "自动参数建议",
                "二值与骨架预览",
                "完整 CNT 检测与逐根测量",
                "深色紧凑颗粒识别与尺寸统计",
                "空间分布与团聚分析",
                "JSON、CSV、TXT、PNG 导出",
                "双图与组图统计对比",
            ],
        }

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(file: Annotated[UploadFile, File(...)]) -> dict:
        payload = await _read_upload(file, max_upload_bytes)
        image = await run_in_threadpool(_decode_image, payload)
        analyzer = CNTAnalyzer()
        await run_in_threadpool(analyzer.set_image, image)
        scale_result = await run_in_threadpool(
            analyzer.apply_detected_scale,
            10.0,
            True,
        )
        suggestion = await run_in_threadpool(analyzer.suggest_preprocess_params, None, "balanced")
        original_media_type = _browser_native_media_type(payload)
        keep_original_payload = bool(
            original_media_type and len(payload) <= ORIGINAL_PAYLOAD_CACHE_MAX_BYTES
        )
        session = session_store.create(
            _safe_filename(file.filename),
            analyzer,
            original_payload=payload if keep_original_payload else None,
            original_media_type=original_media_type if keep_original_payload else None,
        )
        height, width = analyzer.original_image.shape[:2]
        return {
            "session_id": session.id,
            "filename": session.filename,
            "width": int(width),
            "height": int(height),
            "scale_result": json_safe(scale_result),
            "scale_status": json_safe(analyzer.get_scale_status()),
            "suggestion": json_safe(suggestion),
            "rois": [],
            "image_urls": _image_urls(session),
            "expires_in_seconds": session_store.ttl_seconds,
        }

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        session_store.delete(session_id)
        return Response(status_code=204)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            image = session.analyzer.original_image
            return {
                "session_id": session.id,
                "filename": session.filename,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "scale_status": json_safe(session.analyzer.get_scale_status()),
                "rois": [_roi_dict(item) for item in session.analyzer.rois],
                "last_roi_name": session.last_roi_name,
                "has_result": session.last_result is not None,
                "image_urls": _image_urls(session),
            }

    @app.post("/api/v1/sessions/{session_id}/scale")
    def apply_scale(session_id: str, request: ScaleRequest) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            selection = None
            if request.start is not None and request.end is not None:
                selection = ((request.start.x, request.start.y), (request.end.x, request.end.y))
            session.analyzer.record_manual_scale(
                request.pixels,
                request.micrometers,
                source="manual",
                confidence="high",
                selection_line=selection,
            )
            _clear_derived_state(session)
            suggestion = session.analyzer.suggest_preprocess_params(None, "balanced")
            return {
                "scale_status": json_safe(session.analyzer.get_scale_status()),
                "suggestion": json_safe(suggestion),
                "image_urls": _image_urls(session),
            }

    @app.post("/api/v1/sessions/{session_id}/rois", status_code=201)
    def create_roi(session_id: str, request: RoiRequest) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            names = {roi.name for roi in session.analyzer.rois}
            name = request.name or next(f"ROI_{i}" for i in range(1, len(names) + 2) if f"ROI_{i}" not in names)
            if any(item.name == name for item in session.analyzer.rois):
                raise HTTPException(409, "ROI 名称已存在")
            roi = make_roi(
                session.analyzer,
                name=name,
                x=request.x,
                y=request.y,
                width=request.width,
                height=request.height,
            )
            session.analyzer.add_roi(roi)
            session.last_roi_name = roi.name
            _clear_derived_state(session, clear_roi_measurements=False)
            return {
                "roi": _roi_dict(roi),
                "rois": [_roi_dict(item) for item in session.analyzer.rois],
                "image_urls": _image_urls(session),
            }

    @app.delete("/api/v1/sessions/{session_id}/rois/{roi_name}")
    def delete_roi(session_id: str, roi_name: str) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            try:
                session.find_roi(roi_name)
            except KeyError:
                raise HTTPException(404, "ROI 不存在") from None
            session.analyzer.remove_roi(roi_name)
            if session.last_roi_name == roi_name:
                session.last_roi_name = None
            _clear_derived_state(session, clear_roi_measurements=False)
            return {
                "rois": [_roi_dict(item) for item in session.analyzer.rois],
                "image_urls": _image_urls(session),
            }

    @app.post("/api/v1/sessions/{session_id}/suggest")
    def suggest(session_id: str, request: SuggestRequest) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            roi = _find_roi(session, request.roi_name)
            return json_safe(session.analyzer.suggest_preprocess_params(roi, request.detection_profile))

    @app.post("/api/v1/sessions/{session_id}/preprocess")
    def preprocess(session_id: str, request: PreprocessRequest) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            roi = _find_roi(session, request.roi_name)
            session.analyzer.preprocess(
                roi=roi,
                **request.settings().as_kwargs(generate_skeleton=request.generate_skeleton),
            )
            session.invalidate_views()
            session.last_roi_name = request.roi_name
            session.last_result = None
            return {
                "status": "ready",
                "roi": None if roi is None else _roi_dict(roi),
                "image_urls": _image_urls(session),
            }

    @app.post("/api/v1/sessions/{session_id}/analyze")
    def analyze(session_id: str, request: AnalyzeRequest) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            roi = _find_roi(session, request.roi_name)
            result = run_analysis(
                session.analyzer,
                name=session.filename,
                preprocess=request.preprocess.settings(),
                detection=request.detection.settings(),
                roi=roi,
            )
            session.invalidate_views()
            session.last_result = result
            session.last_roi_name = request.roi_name
            response = public_analysis_result(result, include_measurements=False)
            response["image_urls"] = _image_urls(session)
            response["export_urls"] = {
                item: f"/api/v1/sessions/{session.id}/exports/{item}"
                for item in ("json", "csv", "report", "overlay", "skeleton", "particles")
            }
            response["measurements_url"] = f"/api/v1/sessions/{session.id}/measurements"
            return response

    @app.get("/api/v1/sessions/{session_id}/measurements")
    def get_measurements(session_id: str) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            result = _require_result(session)
            return public_measurements(result)

    @app.get("/api/v1/sessions/{session_id}/result")
    def get_result(session_id: str) -> dict:
        session = _get_session(session_store, session_id)
        with session.lock:
            result = _require_result(session)
            response = public_analysis_result(result, include_measurements=False)
            response["image_urls"] = _image_urls(session)
            response["measurements_url"] = f"/api/v1/sessions/{session.id}/measurements"
            return response

    @app.get("/api/v1/sessions/{session_id}/images/{view}")
    def session_image(
        session_id: str,
        view: Literal["original", "enhanced", "binary", "skeleton-preview", "overlay", "skeleton", "particles"],
        v: Optional[int] = Query(default=None),
    ) -> Response:
        del v
        session = _get_session(session_store, session_id)
        with session.lock:
            return _preview_image_response(session, view)

    @app.get("/api/v1/sessions/{session_id}/exports/{kind}")
    def export_result(
        session_id: str,
        kind: Literal["json", "csv", "report", "overlay", "skeleton", "particles"],
    ) -> Response:
        session = _get_session(session_store, session_id)
        with session.lock:
            result = _require_result(session)
            stem = Path(session.filename).stem or "optcnt"
            if kind == "json":
                return _download(analysis_json_bytes(result), "application/json; charset=utf-8", f"{stem}-OptCNT.json")
            if kind == "csv":
                return _download(analysis_csv_bytes(result), "text/csv; charset=utf-8", f"{stem}-OptCNT.csv")
            if kind == "report":
                return _download(analysis_report_bytes(result), "text/plain; charset=utf-8", f"{stem}-OptCNT-report.txt")
            view = kind if kind in {"skeleton", "particles"} else "overlay"
            return _image_response(_resolve_view(session, view), f"{stem}-OptCNT-{view}.png", download=True)

    @app.post("/api/v1/comparisons", status_code=202)
    async def create_comparison(
        base_files: Annotated[List[UploadFile], File(...)],
        experiment_files: Annotated[List[UploadFile], File(...)],
        settings: Annotated[str, Form()] = "{}",
    ) -> dict:
        if not base_files or not experiment_files:
            raise HTTPException(422, "base组和实验组都必须包含图像")
        if len(base_files) > max_batch_files or len(experiment_files) > max_batch_files:
            raise HTTPException(413, f"每组最多 {max_batch_files} 张图像")
        declared_total = sum(int(upload.size or 0) for upload in [*base_files, *experiment_files])
        if declared_total > max_batch_total_bytes:
            raise HTTPException(413, f"整批图像不能超过 {max_batch_total_bytes // (1024 * 1024)} MB")
        try:
            batch_request = BatchRequest.model_validate(json.loads(settings or "{}"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(422, f"对比参数无效: {exc}") from exc

        work_dir = job_store.new_work_dir()
        try:
            base_paths, consumed_bytes = await _save_uploads(
                base_files,
                work_dir / "base",
                max_upload_bytes,
                max_total_bytes=max_batch_total_bytes,
            )
            experiment_paths, _ = await _save_uploads(
                experiment_files,
                work_dir / "experiment",
                max_upload_bytes,
                max_total_bytes=max_batch_total_bytes,
                initial_total_bytes=consumed_bytes,
            )
            job = job_store.create(work_dir, base_paths, experiment_paths, batch_request.settings())
        except QueueFullError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(429, str(exc), headers={"Retry-After": "30"}) from exc
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        return {"job_id": job.id, "status": job.status, "total": job.total, "status_url": f"/api/v1/comparisons/{job.id}"}

    @app.get("/api/v1/comparisons/{job_id}")
    def comparison_status(job_id: str) -> dict:
        try:
            return json_safe(job_store.get(job_id).snapshot())
        except KeyError:
            raise HTTPException(404, "对比任务不存在或已过期") from None

    @app.get("/api/v1/comparisons/{job_id}/representatives/{group}")
    def representative(job_id: str, group: Literal["base", "experiment"]) -> Response:
        try:
            content = job_store.representative_image(job_id, group)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "对比任务不存在或已过期") from None
        return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

    @app.get("/api/v1/comparisons/{job_id}/export.csv")
    def export_comparison(job_id: str) -> Response:
        try:
            snapshot = job_store.get(job_id).snapshot()
        except KeyError:
            raise HTTPException(404, "对比任务不存在或已过期") from None
        if snapshot["status"] != "complete" or not snapshot["result"]:
            raise HTTPException(409, "对比任务尚未完成")
        return _download(comparison_csv_bytes(snapshot["result"]), "text/csv; charset=utf-8", "OptCNT-group-comparison.csv")

    def require_worker_token(value: Optional[str]) -> None:
        if not worker_api_token:
            raise HTTPException(503, "外部计算节点尚未启用")
        if value is None or not secrets.compare_digest(value, worker_api_token):
            raise HTTPException(401, "计算节点令牌无效")

    @app.post("/api/v1/workers/heartbeat")
    def worker_heartbeat(
        request: WorkerHeartbeatRequest,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
    ) -> dict:
        require_worker_token(x_optcnt_worker_token)
        try:
            return job_store.heartbeat(
                request.worker_id,
                kind=request.kind,
                engine_version=request.engine_version,
                capacity=request.capacity,
                metadata=request.metadata,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/workers/lease")
    def worker_lease(
        request: WorkerLeaseRequest,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
    ):
        require_worker_token(x_optcnt_worker_token)
        try:
            lease = job_store.lease(
                request.worker_id,
                kind=request.kind,
                engine_version=request.engine_version,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204) if lease is None else JSONResponse(content=json_safe(lease))

    @app.post("/api/v1/workers/jobs/{job_id}/renew")
    def worker_renew(
        job_id: str,
        request: WorkerRenewRequest,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
    ) -> dict:
        require_worker_token(x_optcnt_worker_token)
        try:
            return job_store.renew(
                job_id,
                request.worker_id,
                request.lease_token,
                current=request.current,
                message=request.message,
            )
        except KeyError:
            raise HTTPException(404, "任务不存在") from None
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/workers/jobs/{job_id}/files/{file_id}")
    def worker_file(
        job_id: str,
        file_id: str,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
        x_optcnt_worker_id: Annotated[Optional[str], Header()] = None,
        x_optcnt_lease_token: Annotated[Optional[str], Header()] = None,
    ) -> Response:
        require_worker_token(x_optcnt_worker_token)
        if not x_optcnt_worker_id or not x_optcnt_lease_token:
            raise HTTPException(401, "缺少计算节点租约凭据")
        try:
            path = job_store.file_path(
                job_id,
                file_id,
                x_optcnt_worker_id,
                x_optcnt_lease_token,
            )
        except KeyError:
            raise HTTPException(404, "任务文件不存在") from None
        except FileNotFoundError:
            raise HTTPException(410, "任务文件已清理") from None
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(
            content=path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/v1/workers/jobs/{job_id}/complete")
    def worker_complete(
        job_id: str,
        request: WorkerCompleteRequest,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
    ) -> dict:
        require_worker_token(x_optcnt_worker_token)
        representatives = {}
        for group in ("base", "experiment"):
            encoded = request.representatives.get(group, "")
            if len(encoded) > 16 * 1024 * 1024:
                raise HTTPException(413, "代表图超过传输限制")
            try:
                representatives[group] = base64.b64decode(encoded, validate=True) if encoded else b""
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(422, f"{group} 代表图编码无效") from exc
        try:
            return job_store.complete(
                job_id,
                request.worker_id,
                request.lease_token,
                result=request.result,
                representative_images=representatives,
            )
        except KeyError:
            raise HTTPException(404, "任务不存在") from None
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/workers/jobs/{job_id}/fail")
    def worker_fail(
        job_id: str,
        request: WorkerFailRequest,
        x_optcnt_worker_token: Annotated[Optional[str], Header()] = None,
    ) -> dict:
        require_worker_token(x_optcnt_worker_token)
        try:
            return job_store.fail(
                job_id,
                request.worker_id,
                request.lease_token,
                request.error,
                retryable=request.retryable,
            )
        except KeyError:
            raise HTTPException(404, "任务不存在") from None
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc

    resolved_web_dir = Path(web_dir or os.getenv("OPTCNT_WEB_DIR", DEFAULT_WEB_DIR)).resolve()
    if resolved_web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(resolved_web_dir), html=True), name="web")
    return app


def _get_session(store: SessionStore, session_id: str) -> AnalysisSession:
    try:
        return store.get(session_id)
    except KeyError:
        raise HTTPException(404, "分析会话不存在或已过期") from None


def _find_roi(session: AnalysisSession, roi_name: Optional[str]):
    try:
        return session.find_roi(roi_name)
    except KeyError:
        raise HTTPException(404, "ROI 不存在") from None


def _require_result(session: AnalysisSession) -> dict:
    if session.last_result is None:
        raise HTTPException(409, "请先完成 CNT 检测")
    return session.last_result


def _clear_derived_state(session: AnalysisSession, *, clear_roi_measurements: bool = True) -> None:
    session.invalidate_views()
    analyzer = session.analyzer
    analyzer.clear_measurements()
    if clear_roi_measurements:
        for roi in analyzer.rois:
            roi.measurements = []
            roi.particle_measurements = []
    analyzer.binary_image = None
    analyzer.processed_image = None
    analyzer.skeleton_image = None
    analyzer.skeleton_overlay = None
    session.last_result = None


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    content = await upload.read(max_bytes + 1)
    await upload.close()
    if not content:
        raise HTTPException(400, "上传文件为空")
    if len(content) > max_bytes:
        raise HTTPException(413, f"单张图像不能超过 {max_bytes // (1024 * 1024)} MB")
    return content


def _decode_image(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise HTTPException(415, "无法解码图像，请使用 PNG、JPEG、TIFF、BMP 或 WebP")
    if image.shape[0] * image.shape[1] > 80_000_000:
        raise HTTPException(413, "图像像素总数过大")
    return image


def _browser_native_media_type(payload: bytes) -> Optional[str]:
    """Return a browser-native image type only when the payload signature is trusted."""
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _save_uploads(
    files: List[UploadFile],
    directory: Path,
    max_bytes: int,
    *,
    max_total_bytes: int,
    initial_total_bytes: int = 0,
) -> tuple[List[str], int]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    total_bytes = int(initial_total_bytes)
    for index, upload in enumerate(files):
        payload = await _read_upload(upload, max_bytes)
        total_bytes += len(payload)
        if total_bytes > max_total_bytes:
            raise HTTPException(413, f"整批图像不能超过 {max_total_bytes // (1024 * 1024)} MB")
        # Decode now so corrupt files fail before the background job is accepted.
        await run_in_threadpool(_decode_image, payload)
        suffix = Path(upload.filename or "image.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            suffix = ".png"
        path = directory / f"{index:03d}{suffix}"
        path.write_bytes(payload)
        paths.append(str(path))
    return paths, total_bytes


def _resolve_view(session: AnalysisSession, view: str) -> np.ndarray:
    analyzer = session.analyzer
    roi = _find_roi(session, session.last_roi_name)
    if view == "original":
        return analyzer.original_image.copy()
    if view == "enhanced":
        return analyzer.image.copy()
    if view == "binary":
        if analyzer.binary_image is None:
            raise HTTPException(409, "请先生成预处理预览")
        return analyzer.binary_image.copy()
    if view == "skeleton-preview":
        if analyzer.skeleton_image is None:
            raise HTTPException(409, "请先生成骨架预览")
        return analyzer.get_skeleton_preview(roi)
    if view == "overlay":
        _require_result(session)
        return analyzer.get_visualization(roi)
    if view == "skeleton":
        _require_result(session)
        return analyzer.get_visualization_with_skeleton(roi)
    if view == "particles":
        _require_result(session)
        return analyzer.get_particle_visualization(roi)
    raise HTTPException(404, "未知图像视图")


def _image_response(image: np.ndarray, filename: str, *, download: bool = False) -> Response:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "图像编码失败")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = _attachment_header(filename)
    return Response(content=encoded.tobytes(), media_type="image/png", headers=headers)


def _preview_image_response(session: AnalysisSession, view: str) -> Response:
    """Serve a small versioned browser preview while keeping exports lossless."""
    headers = {"Cache-Control": PREVIEW_CACHE_CONTROL}
    if view == "original" and session.original_payload and session.original_media_type:
        return Response(
            content=session.original_payload,
            media_type=session.original_media_type,
            headers=headers,
        )

    encoded = session.view_cache.get(view)
    if encoded is None:
        image = _resolve_view(session, view)
        if view == "binary":
            extension = ".png"
            media_type = "image/png"
            parameters = []
        else:
            extension = ".jpg"
            media_type = "image/jpeg"
            parameters = [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
            image = _downscale_preview(image)
        ok, buffer = cv2.imencode(extension, image, parameters)
        if not ok:
            raise HTTPException(500, "图像预览编码失败")
        encoded = buffer.tobytes()
        session.cache_view(view, encoded)
    else:
        media_type = "image/png" if view == "binary" else "image/jpeg"

    return Response(content=encoded, media_type=media_type, headers=headers)


def _downscale_preview(image: np.ndarray) -> np.ndarray:
    """Downscale a browser preview so the low-end server transfers less data."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= PREVIEW_MAX_EDGE:
        return image
    scale = PREVIEW_MAX_EDGE / float(longest)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _attachment_header(filename: str) -> str:
    encoded = quote(filename, safe="")
    if encoded == filename:
        return f'attachment; filename="{filename}"'
    return f"attachment; filename*=UTF-8''{encoded}"


def _download(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": _attachment_header(filename), "Cache-Control": "no-store"},
    )


def _safe_filename(filename: Optional[str]) -> str:
    name = Path(filename or "image").name.strip()
    return name[:180] or "image"


def _roi_dict(roi) -> dict:
    return {"name": roi.name, "x": int(roi.x), "y": int(roi.y), "width": int(roi.width), "height": int(roi.height)}


def _image_urls(session: AnalysisSession) -> dict:
    base = f"/api/v1/sessions/{session.id}/images"
    revision = int(session.view_revision)
    return {
        "original": f"{base}/original?v=0",
        "enhanced": f"{base}/enhanced?v={revision}",
        "binary": f"{base}/binary?v={revision}",
        "skeleton_preview": f"{base}/skeleton-preview?v={revision}",
        "overlay": f"{base}/overlay?v={revision}",
        "skeleton": f"{base}/skeleton?v={revision}",
        "particles": f"{base}/particles?v={revision}",
    }


app = create_app()
