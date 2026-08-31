# ============================================================
# JobHunter - Phase 6: Action & Tailoring Layer
#   - LaTeX resume generation via Santiago's generate-latex.mjs
#   - OSINT recruiter waterfall (Apollo -> Hunter.io) + outreach draft
# ============================================================
from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from agents.sourcer import JobRecord

logger = logging.getLogger("jobhunter.action_layer")

_RECRUITER_TITLES = [
    "Technical Recruiter", "Recruiter", "Talent Acquisition",
    "HR Manager", "Hiring Manager", "People Operations",
]

# Domains that are aggregators/ATS boards — not the employer's own domain.
_NON_EMPLOYER_DOMAINS = (
    "linkedin", "indeed", "glassdoor", "greenhouse", "lever", "ashbyhq",
    "myworkdayjobs", "smartrecruiters", "google", "bamboohr", "workable",
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _slugify(*parts: str) -> str:
    raw = "_".join(p for p in parts if p)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return (slug or "resume")[:80]


def _extract_domain(job: JobRecord) -> Optional[str]:
    """Best-effort employer domain from the job URL, else naive company guess."""
    if job.url:
        netloc = urlparse(job.url).netloc.lower().replace("www.", "")
        if netloc and not any(s in netloc for s in _NON_EMPLOYER_DOMAINS):
            return netloc
    if job.company:
        guess = re.sub(r"[^a-z0-9]", "", job.company.lower())
        if guess:
            return f"{guess}.com"  # low-confidence guess; APIs validate
    return None


# ------------------------------------------------------------
# 1. LaTeX resume generation
# ------------------------------------------------------------
def generate_resume(
    job: JobRecord,
    *,
    career_ops_dir: str | Path,
    cv_path: str | Path,
    story_bank: str,
    session_output_dir: str | Path,
    timeout: int = 180,
) -> Optional[Path]:
    """Invoke `node generate-latex.mjs` to produce a tailored PDF. Returns its path."""
    career_ops_dir = Path(career_ops_dir)
    script = career_ops_dir / "generate-latex.mjs"
    if not script.exists():
        logger.error("generate-latex.mjs not found at %s", script)
        return None

    session_output_dir = Path(session_output_dir)
    session_output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(job.company, job.title)
    output_pdf = session_output_dir / f"{slug}.pdf"

    # Stage the job description + story bank as files for the Node worker.
    tmp = session_output_dir / ".tmp"
    tmp.mkdir(exist_ok=True)
    jd_file = tmp / f"{slug}_jd.txt"
    story_file = tmp / f"{slug}_stories.txt"
    jd_file.write_text(job.description or "", encoding="utf-8")
    story_file.write_text(story_bank or "", encoding="utf-8")

    cmd = [
        "node", "generate-latex.mjs",
        f"--cv={cv_path}",
        f"--job={jd_file}",
        f"--stories={story_file}",
        f"--title={job.title or ''}",
        f"--company={job.company or ''}",
        f"--output={output_pdf}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(career_ops_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            logger.warning(
                "generate-latex exited %d for '%s': %s",
                proc.returncode, slug, (proc.stderr or "")[-500:],
            )
        if output_pdf.exists():
            logger.info("Generated tailored CV: %s", output_pdf)
            return output_pdf
        logger.warning("generate-latex produced no PDF for '%s'", slug)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("generate-latex timed out (%ss) for '%s'", timeout, slug)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate-latex error for '%s': %s", slug, exc)
        return None


# ------------------------------------------------------------
# 2. OSINT waterfall: Apollo -> Hunter.io
# ------------------------------------------------------------
def _apollo_search(domain: str, api_key: Optional[str], timeout: int = 20) -> Optional[dict]:
    if not api_key or not domain:
        return None
    url = "https://api.apollo.io/api/v1/mixed_people/search"
    payload = {
        "api_key": api_key,
        "q_organization_domains": domain,
        "person_titles": _RECRUITER_TITLES,
        "page": 1,
        "per_page": 5,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url, json=payload,
                headers={"Content-Type": "application/json", "Cache-Control": "no-cache"},
            )
            resp.raise_for_status()
            data = resp.json()
        for person in data.get("people", []):
            email = person.get("email")
            if email and "email_not_unlocked" not in str(email).lower():
                return {
                    "email": email,
                    "name": person.get("name", ""),
                    "title": person.get("title", ""),
                    "source": "apollo",
                }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Apollo lookup failed for %s: %s", domain, exc)
    return None


def _hunter_search(domain: str, api_key: Optional[str], timeout: int = 20) -> Optional[dict]:
    if not api_key or not domain:
        return None
    url = "https://api.hunter.io/v2/domain-search"
    params = {"domain": domain, "api_key": api_key, "department": "hr", "limit": 5}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        for e in data.get("data", {}).get("emails", []):
            if e.get("value"):
                name = f"{e.get('first_name', '')} {e.get('last_name', '')}".strip()
                return {
                    "email": e["value"],
                    "name": name,
                    "title": e.get("position", ""),
                    "source": "hunter",
                }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hunter lookup failed for %s: %s", domain, exc)
    return None


def find_recruiter_contact(
    job: JobRecord,
    apollo_key: Optional[str],
    hunter_key: Optional[str],
) -> Optional[dict]:
    """Apollo first, Hunter.io as fallback."""
    domain = _extract_domain(job)
    if not domain:
        return None
    return _apollo_search(domain, apollo_key) or _hunter_search(domain, hunter_key)


# ------------------------------------------------------------
# 3. Personalized outreach draft
# ------------------------------------------------------------
def _template_email(job: JobRecord, contact: dict) -> tuple[str, str]:
    name = contact.get("name") or "Hiring Team"
    subject = f"Application for {job.title} at {job.company}"
    body = (
        f"Hi {name},\n\n"
        f"I'm reaching out regarding the {job.title} role at {job.company}. "
        f"My background aligns closely with what you're looking for, and I've "
        f"attached a tailored CV highlighting the most relevant experience.\n\n"
        f"I'd welcome the chance to discuss how I can contribute.\n\n"
        f"Best regards"
    )
    return subject, body


def draft_outreach_email(
    job: JobRecord,
    contact: dict,
    story_bank: str,
    factory=None,
) -> tuple[str, str]:
    if factory is None:
        return _template_email(job, contact)

    system = (
        "You are the candidate writing a concise, warm, professional cold-outreach "
        "email to a recruiter. Ground specifics in the STORY BANK. 120-160 words. "
        "No placeholders or brackets. "
        'Return ONLY JSON: {"subject": "...", "body": "..."}'
    )
    user = (
        f"Role: {job.title} at {job.company}\n"
        f"Recruiter: {contact.get('name') or 'Hiring Team'} ({contact.get('title', '')})\n"
        f"Story bank:\n{(story_bank or '')[:2000]}\n"
    )
    try:
        data = factory.complete_json(system, user)
        subject = data.get("subject") or f"Application for {job.title}"
        body = data.get("body") or _template_email(job, contact)[1]
        return subject, body
    except Exception as exc:  # noqa: BLE001
        logger.warning("Outreach draft failed; using template: %s", exc)
        return _template_email(job, contact)


# ------------------------------------------------------------
# Orchestration entry point (per aligned job)
# ------------------------------------------------------------
def produce_assets(
    job: JobRecord,
    *,
    session_id: str,
    career_ops_dir: str | Path,
    cv_path: str | Path,
    story_bank: str,
    data_output_dir: str | Path,
    apollo_key: Optional[str],
    hunter_key: Optional[str],
    factory=None,
) -> JobRecord:
    """
    For one aligned job, run resume generation and the OSINT waterfall
    CONCURRENTLY, then enrich the JobRecord (cv_path, contact_email).
    """
    session_output_dir = Path(data_output_dir) / session_id

    def _resume_task() -> Optional[Path]:
        return generate_resume(
            job,
            career_ops_dir=career_ops_dir,
            cv_path=cv_path,
            story_bank=story_bank,
            session_output_dir=session_output_dir,
        )

    def _osint_task() -> Optional[dict]:
        contact = find_recruiter_contact(job, apollo_key, hunter_key)
        if not contact:
            return None
        subject, body = draft_outreach_email(job, contact, story_bank, factory)
        return {"contact": contact, "subject": subject, "body": body}

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_resume = pool.submit(_resume_task)
        f_osint = pool.submit(_osint_task)
        pdf_path = f_resume.result()
        osint = f_osint.result()

    # Store CV path RELATIVE to data_output_dir (the download endpoint's sandbox).
    if pdf_path:
        try:
            rel = Path(pdf_path).resolve().relative_to(Path(data_output_dir).resolve())
            job.cv_path = str(rel)
        except ValueError:
            job.cv_path = Path(pdf_path).name

    if osint:
        job.contact_email = osint["contact"].get("email")
        # Persist the drafted outreach next to the resume.
        try:
            slug = _slugify(job.company, job.title)
            (session_output_dir / f"{slug}_outreach.txt").write_text(
                f"To: {job.contact_email}\n"
                f"Subject: {osint['subject']}\n\n"
                f"{osint['body']}\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not persist outreach draft", exc_info=True)

    return job
