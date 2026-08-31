"""
JobHunter — FastAPI Orchestrator (Phase 2 + Phase 3 core)
---------------------------------------------------------
Responsibilities in this file:
  • Serve the single-page frontend (templates/index.html)
  • Accept the onboarding form (PDF upload + text fields)
  • Persist the resume to /data/uploads
  • Generate portals.yml for the Career-Ops engine
  • Expose a WebSocket endpoint that streams pipeline logs to the browser
"""

from __future__ import annotations

from agents.orchestrator import run_orchestrator

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("jobhunter.api")


# ------------------------------------------------------------
# Settings (pydantic-settings)
# ------------------------------------------------------------
class Settings(BaseSettings):
    """Loads configuration from the container environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Server ---
    app_host: str = "0.0.0.0"
    app_port: int = 3000
    log_level: str = "info"

    # --- LLM ---
    active_llm_provider: str = "openai"
    min_alignment_score: int = 7

    # --- Internal container paths ---
    data_upload_dir: Path = Path("/app/data/uploads")
    data_output_dir: Path = Path("/app/data/output")
    data_logs_dir: Path = Path("/app/data/logs")
    config_dir: Path = Path("/app/config")
    career_ops_dir: Path = Path("/app/career-ops")
    scan_history_path: Path = Path("/app/data/logs/scan-history.tsv")

    # Map .env variable names -> fields where they differ
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


def _load_settings() -> Settings:
    """
    Resolve settings, honoring the .env var names from Phase 1
    (UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, CONFIG_DIR, CAREER_OPS_DIR,
    SCAN_HISTORY_PATH) which don't match the field names 1:1.
    """
    import os

    s = Settings()
    # Explicit overrides for the Phase 1 .env naming convention
    s.data_upload_dir = Path(os.getenv("UPLOAD_DIR", str(s.data_upload_dir)))
    s.data_output_dir = Path(os.getenv("OUTPUT_DIR", str(s.data_output_dir)))
    s.data_logs_dir = Path(os.getenv("LOG_DIR", str(s.data_logs_dir)))
    s.config_dir = Path(os.getenv("CONFIG_DIR", str(s.config_dir)))
    s.career_ops_dir = Path(os.getenv("CAREER_OPS_DIR", str(s.career_ops_dir)))
    s.scan_history_path = Path(
        os.getenv("SCAN_HISTORY_PATH", str(s.scan_history_path))
    )
    return s


settings = _load_settings()

# Ensure critical directories exist at boot
for _d in (
    settings.data_upload_dir,
    settings.data_output_dir,
    settings.data_logs_dir,
    settings.config_dir,
):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


# ------------------------------------------------------------
# ConnectionManager: session-isolated WebSocket broadcasting
# ------------------------------------------------------------
class ConnectionManager:
    """Tracks one live WebSocket per session_id and broadcasts
    structured JSON payloads (logs + results) to the browser."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[session_id] = websocket
        logger.info("WebSocket connected for session %s", session_id)

    async def disconnect(self, session_id: str) -> None:
        async with self._lock:
            self._connections.pop(session_id, None)
        logger.info("WebSocket disconnected for session %s", session_id)

    async def send(self, session_id: str, payload: dict[str, Any]) -> None:
        """Send a structured JSON payload to a specific session."""
        ws = self._connections.get(session_id)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send to %s: %s", session_id, exc)
            await self.disconnect(session_id)

    async def log(
        self,
        session_id: str,
        message: str,
        level: str = "INFO",
    ) -> None:
        """Convenience helper to emit a console log line."""
        await self.send(
            session_id,
            {
                "type": "log",
                "level": level.upper(),
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def result(self, session_id: str, job: dict[str, Any]) -> None:
        """Emit a single job result card to the dashboard."""
        await self.send(session_id, {"type": "result", "job": job})

    async def status(self, session_id: str, state: str) -> None:
        """Emit a pipeline status change (running/done/error)."""
        await self.send(session_id, {"type": "status", "state": state})


manager = ConnectionManager()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_portals_yml(
    session_id: str,
    target_role: str,
    location: str,
) -> Path:
    """Write config/<session_id>/portals.yml in the format the
    Career-Ops engine expects. Returns the written path."""
    session_config_dir = settings.config_dir / session_id
    session_config_dir.mkdir(parents=True, exist_ok=True)

    portals_data = {
        "search": {
            "role": target_role,
            "location": location,
            "results_wanted": 50,
        },
        "providers": {
            "greenhouse": {"enabled": True},
            "lever": {"enabled": True},
            "ashby": {"enabled": True},
        },
        "aggregators": {
            "linkedin": {"enabled": True},
            "indeed": {"enabled": True},
            "glassdoor": {"enabled": True},
        },
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    portals_path = session_config_dir / "portals.yml"
    with portals_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(portals_data, fh, sort_keys=False, default_flow_style=False)

    logger.info("Wrote portals.yml for session %s -> %s", session_id, portals_path)
    return portals_path


def _clear_stale_scan_logs() -> None:
    """Clear the flat scan-history.tsv so each run starts fresh."""
    try:
        path = settings.scan_history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        # Recreate an empty file with a header row
        path.write_text(
            "timestamp\tsession_id\tcompany\ttitle\turl\tscore\tstatus\n",
            encoding="utf-8",
        )
        logger.info("Cleared stale scan history at %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear scan history: %s", exc)


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="JobHunter", version="0.2.0")

# Static assets
_static_dir = Path("frontend/static")
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

_templates_dir = Path("frontend/templates")


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page dashboard."""
    index_path = _templates_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict[str, str]:
    """Docker health-check endpoint."""
    return {"status": "ok"}


@app.post("/api/scan/start", status_code=202)
async def start_scan(
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    location: str = Form(...),
    story_bank: str = Form(""),
    is_internship: bool = Form(False),
) -> JSONResponse:
    """Accept intake form, persist artifacts, and schedule the pipeline."""
    # --- Validate extension ---
    original_name = resume.filename or "resume"
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # --- New session ---
    session_id = uuid.uuid4().hex

    # --- Save uploaded CV ---
    session_upload_dir = settings.data_upload_dir / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = session_upload_dir / f"resume{ext}"

    contents = await resume.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    upload_path.write_bytes(contents)
    logger.info("Saved resume for session %s -> %s", session_id, upload_path)

    # --- Clear stale scan logs (fresh run) ---
    _clear_stale_scan_logs()

    # --- Build portals.yml ---
    build_portals_yml(session_id, target_role.strip(), location.strip())

    # --- Save story bank ---
    session_config_dir = settings.config_dir / session_id
    session_config_dir.mkdir(parents=True, exist_ok=True)
    (session_config_dir / "story_bank.txt").write_text(
        story_bank or "", encoding="utf-8"
    )

    # --- Schedule orchestrator ---
    background_tasks.add_task(
        run_orchestrator,
        session_id,
        upload_path,
        target_role.strip(),
        location.strip(),
        is_internship,
    )

    return JSONResponse(
        status_code=202,
        content={
            "session_id": session_id,
            "status": "accepted",
            "message": "Scan scheduled. Connect to the WebSocket for live logs.",
        },
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Real-time console streaming with a 30s heartbeat."""
    await manager.connect(session_id, websocket)
    await manager.log(session_id, "Console connected. Awaiting pipeline...", "INFO")

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception:  # noqa: BLE001
            return

    hb_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            # We keep the socket open; inbound messages (e.g. "pong")
            # are read to detect disconnects promptly.
            msg = await websocket.receive_text()
            # Optionally handle client pongs / commands here.
            _ = msg
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("WebSocket error for %s: %s", session_id, exc)
    finally:
        hb_task.cancel()
        await manager.disconnect(session_id)


@app.get("/api/download")
async def download(path: str) -> FileResponse:
    """
    Serve a generated PDF with airtight path-traversal protection.

    The `path` query param may be relative (to data_output_dir) or an
    absolute path. In every case the resolved candidate MUST live inside
    data_output_dir, or we return 404.
    """
    base_dir = settings.data_output_dir.resolve()

    raw = Path(path)
    # Resolve against base_dir for relative paths; absolute paths resolve as-is.
    candidate = (raw if raw.is_absolute() else base_dir / raw).resolve()

    # Verify the candidate stays within base_dir.
    try:
        candidate.relative_to(base_dir)
    except ValueError:
        # Escaped the sandbox -> treat as not found (don't leak details).
        raise HTTPException(status_code=404, detail="File not found.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=str(candidate),
        media_type="application/pdf",
        filename=candidate.name,
    )
