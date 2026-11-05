"""In-process bounded session storage for the OptCNT web application."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..core.analyzer_core import CNTAnalyzer
from ..core.models import ROIRegion

VIEW_CACHE_MAX_BYTES = 8 * 1024 * 1024


@dataclass
class AnalysisSession:
    id: str
    filename: str
    analyzer: CNTAnalyzer
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    last_result: Optional[dict] = None
    last_roi_name: Optional[str] = None
    original_payload: Optional[bytes] = field(default=None, repr=False)
    original_media_type: Optional[str] = None
    view_revision: int = 0
    view_cache: Dict[str, bytes] = field(default_factory=dict, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.touched_at = time.time()

    def invalidate_views(self) -> None:
        """Advance the immutable browser-view version and release encoded previews."""
        self.view_revision += 1
        self.view_cache.clear()

    def cache_view(self, view: str, content: bytes) -> None:
        """Keep a small insertion-ordered preview cache within the session memory budget."""
        if len(content) > VIEW_CACHE_MAX_BYTES:
            return
        self.view_cache.pop(view, None)
        cached_bytes = sum(len(item) for item in self.view_cache.values())
        while self.view_cache and cached_bytes + len(content) > VIEW_CACHE_MAX_BYTES:
            oldest = next(iter(self.view_cache))
            cached_bytes -= len(self.view_cache.pop(oldest))
        self.view_cache[view] = content

    def find_roi(self, name: Optional[str]) -> Optional[ROIRegion]:
        if not name:
            return None
        for roi in self.analyzer.rois:
            if roi.name == name:
                return roi
        raise KeyError(name)


class SessionStore:
    def __init__(self, *, ttl_seconds: int = 1800, max_sessions: int = 24):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: Dict[str, AnalysisSession] = {}
        self._lock = threading.RLock()

    def create(
        self,
        filename: str,
        analyzer: CNTAnalyzer,
        *,
        original_payload: Optional[bytes] = None,
        original_media_type: Optional[str] = None,
    ) -> AnalysisSession:
        with self._lock:
            self.cleanup()
            if len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions.values(), key=lambda item: item.touched_at)
                self._sessions.pop(oldest.id, None)
            session = AnalysisSession(
                uuid.uuid4().hex,
                filename,
                analyzer,
                original_payload=original_payload,
                original_media_type=original_media_type,
            )
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> AnalysisSession:
        with self._lock:
            self.cleanup()
            session = self._sessions.get(str(session_id))
            if session is None:
                raise KeyError(session_id)
            session.touch()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session_id), None) is not None

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, value in self._sessions.items() if value.touched_at < cutoff]
        for key in expired:
            self._sessions.pop(key, None)

    def status(self) -> dict:
        with self._lock:
            self.cleanup()
            return {"active_sessions": len(self._sessions), "max_sessions": self.max_sessions, "ttl_seconds": self.ttl_seconds}
