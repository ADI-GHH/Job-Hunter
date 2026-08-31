# ============================================================
# JobHunter - Phase 3: CV Profiler (extraction + seniority)
# ============================================================
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents.sourcer import SeniorityLevel  # canonical enum

logger = logging.getLogger("jobhunter.cv_profiler")

# Cap text sent to the LLM to keep classification cheap.
_MAX_LLM_CHARS = 8000


# ------------------------------------------------------------
# Profile result
# ------------------------------------------------------------
@dataclass
class CVProfile:
    seniority: SeniorityLevel
    raw_text: str
    char_count: int
    has_student_markers: bool
    classification_source: str  # "llm" | "regex_crosscheck" | "regex_fallback"


# ------------------------------------------------------------
# Text extraction (pdfplumber -> PyPDF2 fallback)
# ------------------------------------------------------------
def _extract_pdf_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber extraction failed: %s", exc)
        return ""


def _extract_pdf_pypdf2(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        parts = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyPDF2 extraction failed: %s", exc)
        return ""


def _extract_docx(path: Path) -> str:
    """Minimal, dependency-free .docx text extraction via zipfile."""
    try:
        import zipfile

        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = xml.replace("</w:p>", "\n")           # paragraph breaks
        text = re.sub(r"<[^>]+>", " ", xml)         # strip tags
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("docx extraction failed: %s", exc)
        return ""


def extract_text(cv_path: Path | str) -> str:
    """Extract raw text from an uploaded CV (.pdf preferred, .docx supported)."""
    path = Path(cv_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf_pdfplumber(path)
        if text.strip():
            return text
        logger.warning("pdfplumber returned empty text; falling back to PyPDF2")
        return _extract_pdf_pypdf2(path)

    if suffix == ".docx":
        return _extract_docx(path)

    logger.warning("Unsupported CV extension '%s'; attempting raw read", suffix)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------
# Regex safety net
# ------------------------------------------------------------
_STUDENT_MARKER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b1st[-\s]?year\b",
        r"\bfirst[-\s]?year\b",
        r"\b2nd[-\s]?year\b",
        r"\bsecond[-\s]?year\b",
        r"\bundergraduate\b",
        r"\bexpected\s+graduation\b",
        r"\bexpected\s+to\s+graduate\b",
        r"\bgraduation\s*[:\-]?\s*20\d{2}\b",
        r"\bcurrently\s+(?:studying|enrolled|pursuing)\b",
        r"\bpursuing\s+(?:a\s+)?(?:bachelor|b\.?sc|b\.?a|b\.?eng)\b",
        r"\bseeking\s+(?:an?\s+)?internship\b",
        r"\bsummer\s+internship\b",
        r"\bstudent\b",
    )
]


def _detect_student_markers(text: str) -> bool:
    """True if the CV shows strong current-student signals."""
    return any(p.search(text) for p in _STUDENT_MARKER_PATTERNS)


# Reuse the sourcer's YoE detector shape for a self-contained fallback.
_YOE_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.IGNORECASE)
_SENIOR_TITLE_RE = re.compile(
    r"\b(senior|lead|principal|staff|head\s+of|director|vp|vice\s+president|manager)\b",
    re.IGNORECASE,
)


def _regex_fallback_classify(text: str) -> SeniorityLevel:
    """Deterministic classification used when the LLM is unavailable."""
    if _detect_student_markers(text):
        return SeniorityLevel.INTERN

    years = [int(m.group(1)) for m in _YOE_RE.finditer(text)]
    years = [y for y in years if 0 < y <= 40]
    if years:
        max_y = max(years)
        if max_y <= 2:
            return SeniorityLevel.JUNIOR
        if max_y <= 5:
            return SeniorityLevel.MID
        return SeniorityLevel.SENIOR

    if _SENIOR_TITLE_RE.search(text):
        return SeniorityLevel.SENIOR

    return SeniorityLevel.UNKNOWN


# ------------------------------------------------------------
# LLM classification (with graceful fallback)
# ------------------------------------------------------------
# ------------------------------------------------------------
# LLM classification (with graceful fallback)
# ------------------------------------------------------------
_CLASSIFY_SYSTEM_PROMPT = (
    "You are a precise CV/resume classifier. Read the candidate's CV text and "
    "determine their CURRENT career seniority. Respond ONLY with a JSON object of "
    'the form {"seniority": "<label>", "confidence": <0..1>} where <label> is '
    "exactly one of: intern, junior, mid, senior.\n"
    "- 'intern': current students, 1st/2nd-year undergraduates, or those seeking "
    "internships/placements.\n"
    "- 'junior': 0-2 years professional experience, or recent graduates.\n"
    "- 'mid': 3-5 years experience.\n"
    "- 'senior': 5+ years experience, leadership, or staff/principal roles."
)

def _classify_with_source(text: str, factory: Optional[object] = None) -> tuple[SeniorityLevel, str]:
    """Attempt LLM classification, falling back to regex on failure."""
    if not text.strip():
        return SeniorityLevel.UNKNOWN, "empty_text"

    try:
        # Lazy import to avoid circular dependencies before Phase 5
        from agents.llm_factory import LLMFactory
        
        if factory is None:
            factory = LLMFactory.from_env()

        # Truncate text to save tokens/cost
        truncated_text = text[:_MAX_LLM_CHARS]

        result_dict = factory.complete_json(
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_prompt=f"Classify this CV:\n\n{truncated_text}"
        )

        seniority_str = result_dict.get("seniority", "unknown")
        return SeniorityLevel.from_string(seniority_str), "llm"

    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM classification failed (%s); regex fallback", exc)
        return _regex_fallback_classify(text), "regex_fallback"


def classify_seniority(text: str, factory: Optional[object] = None) -> SeniorityLevel:
    """Public API: classify CV text into a SeniorityLevel."""
    return _classify_with_source(text, factory)[0]


# ------------------------------------------------------------
# Top-level entry point
# ------------------------------------------------------------
def profile_cv(cv_path: Path | str, factory: Optional[object] = None) -> CVProfile:
    """Extract text and classify seniority for an uploaded CV."""
    path = Path(cv_path)
    text = extract_text(path)
    has_markers = _detect_student_markers(text)
    seniority, source = _classify_with_source(text, factory=factory)

    logger.info(
        "CV profiled: %s chars, seniority=%s (via %s), student_markers=%s",
        len(text),
        seniority.value,
        source,
        has_markers,
    )
    return CVProfile(
        seniority=seniority,
        raw_text=text,
        char_count=len(text),
        has_student_markers=has_markers,
        classification_source=source,
    )