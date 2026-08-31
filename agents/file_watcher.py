# ============================================================
# JobHunter - Phase 6 plumbing: TSV -> WebSocket live streaming
#
# watchdog runs on a native OS thread, so every push into FastAPI's
# asyncio loop MUST go through asyncio.run_coroutine_threadsafe(...).
# ============================================================
from __future__ import annotations

import asyncio
import csv
import logging
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Awaitable, Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("jobhunter.file_watcher")

# broadcast_fn(session_id, payload_dict) -> awaitable  (e.g. ConnectionManager.send)
BroadcastFn = Callable[[str, dict], Awaitable[None]]


# Frontend payload keys -> tolerant list of possible TSV column names.
_COLUMN_MAP: dict[str, list[str]] = {
    "title": ["title", "job_title", "role"],
    "company": ["company", "employer", "company_name"],
    "url": ["url", "job_url", "link", "apply_url"],
    "score": ["score", "alignment_score", "match_score"],
    "reasoning": ["reasoning", "reason", "justification"],
    "cv_path": ["cv_path", "resume_path", "pdf_path", "cv"],
    "contact_email": ["contact_email", "recruiter_email", "email"],
}


def _to_int(value: object) -> Optional[int]:
    try:
        s = str(value).strip()
        return int(float(s)) if s else None
    except (ValueError, TypeError):
        return None


def _normalise_row(raw: dict) -> dict:
    """Map an arbitrary TSV row into the frontend result payload shape."""
    lowered = {(k or "").strip().lower(): (v if v is not None else "") for k, v in raw.items()}
    out: dict = {}
    for target, candidates in _COLUMN_MAP.items():
        value = ""
        for c in candidates:
            if c in lowered and str(lowered[c]).strip():
                value = str(lowered[c]).strip()
                break
        out[target] = value
    out["score"] = _to_int(out.get("score"))
    # keep session_id around for filtering (not sent to UI directly)
    out["_session_id"] = str(lowered.get("session_id", "")).strip()
    return out


def _row_key(job: dict) -> str:
    url = (job.get("url") or "").strip().lower().rstrip("/")
    if url:
        return url
    return f"{(job.get('company') or '').lower()}::{(job.get('title') or '').lower()}"


def _log_future_exc(fut: Future) -> None:
    """done-callback: surface exceptions raised inside the coroutine."""
    try:
        exc = fut.exception()
        if exc is not None:
            logger.warning("Broadcast coroutine raised: %s", exc)
    except Exception:  # noqa: BLE001 - future cancelled etc.
        pass


class ScanHistoryHandler(FileSystemEventHandler):
    """Reacts to scan-history.tsv writes and streams NEW rows to the UI."""

    def __init__(
        self,
        tsv_path: Path,
        session_id: str,
        broadcast_fn: BroadcastFn,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.tsv_path = Path(tsv_path).resolve()
        self.session_id = session_id
        self.broadcast_fn = broadcast_fn
        self.loop = loop
        self._seen_keys: set[str] = set()
        self._lock = threading.Lock()

    # --- watchdog hooks ---
    def on_modified(self, event: FileSystemEvent) -> None:
        self._maybe_process(event)

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe_process(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # editors/atomic writes often rename a temp file over the target
        self._maybe_process(event)

    def _maybe_process(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Match either the src or (for moves) the dest path.
        paths = [getattr(event, "src_path", None), getattr(event, "dest_path", None)]
        for p in paths:
            if not p:
                continue
            try:
                if Path(p).resolve() == self.tsv_path:
                    self.process()
                    return
            except OSError:
                continue

    # --- core ---
    def process(self) -> None:
        """Parse the TSV and broadcast any rows we haven't seen yet."""
        rows = self._parse_tsv()
        new_rows: list[dict] = []
        with self._lock:
            for job in rows:
                # Ignore rows belonging to another session (if column present).
                if job.get("_session_id") and job["_session_id"] != self.session_id:
                    continue
                key = _row_key(job)
                if not key or key in self._seen_keys:
                    continue
                self._seen_keys.add(key)
                new_rows.append(job)

        for job in new_rows:
            job.pop("_session_id", None)  # internal only
            self._schedule_broadcast({"type": "result", "job": job})

        if new_rows:
            logger.info(
                "Streamed %d new result(s) to session %s",
                len(new_rows),
                self.session_id,
            )

    def _parse_tsv(self) -> list[dict]:
        if not self.tsv_path.exists():
            return []
        rows: list[dict] = []
        try:
            with self.tsv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for raw in reader:
                    rows.append(_normalise_row(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse TSV %s: %s", self.tsv_path, exc)
        return rows

    def _schedule_broadcast(self, payload: dict) -> None:
        """
        CRUCIAL threading bridge: hop from the watchdog OS thread onto
        FastAPI's asyncio event loop safely.
        """
        if self.loop is None or not self.loop.is_running():
            logger.warning("Event loop not running; dropping payload for %s", self.session_id)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_fn(self.session_id, payload),
                self.loop,
            )
            future.add_done_callback(_log_future_exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to schedule broadcast: %s", exc)


class FileWatcher:
    """Lifecycle wrapper around a watchdog Observer for one session."""

    def __init__(
        self,
        tsv_path: Path | str,
        session_id: str,
        broadcast_fn: BroadcastFn,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.tsv_path = Path(tsv_path)
        self.session_id = session_id
        self.broadcast_fn = broadcast_fn
        # Capture the running loop if not supplied (must be called from async ctx).
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
        self.loop = loop

        self._observer: Optional[Observer] = None
        self._handler: Optional[ScanHistoryHandler] = None

    def start(self) -> None:
        # Ensure the directory exists so the observer can attach to it.
        self.tsv_path.parent.mkdir(parents=True, exist_ok=True)

        self._handler = ScanHistoryHandler(
            self.tsv_path, self.session_id, self.broadcast_fn, self.loop
        )
        self._observer = Observer()
        # Watch the *directory* — atomic writes replace the file inode.
        self._observer.schedule(
            self._handler, str(self.tsv_path.parent), recursive=False
        )
        self._observer.start()
        logger.info(
            "FileWatcher started: %s (session %s)", self.tsv_path, self.session_id
        )

        # Flush any rows already present at start-up.
        self._handler.process()

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("FileWatcher stopped for session %s", self.session_id)


def start_file_watcher(
    tsv_path: Path | str,
    session_id: str,
    broadcast_fn: BroadcastFn,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> FileWatcher:
    """Convenience factory: build, start, and return a FileWatcher."""
    watcher = FileWatcher(tsv_path, session_id, broadcast_fn, loop)
    watcher.start()
    return watcher
