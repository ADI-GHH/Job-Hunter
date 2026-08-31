# ============================================================
# JobHunter - Phase 3/4: Sourcing Swarm + Zero-Token Gatekeeper
#
# Canonical home for shared models: SeniorityLevel, JobRecord,
# GatekeeperResult. cv_profiler.py imports SeniorityLevel from here.
# ============================================================
from __future__ import annotations

import csv
import logging
import math
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available, fallback scraper will be limited")

logger = logging.getLogger("jobhunter.sourcer")


# ------------------------------------------------------------
# Shared models
# ------------------------------------------------------------
class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: Optional[str]) -> "SeniorityLevel":
        """Tolerant parser for messy LLM / heuristic output."""
        if not value:
            return cls.UNKNOWN
        v = value.strip().lower()

        aliases = {
            "intern": cls.INTERN,
            "internship": cls.INTERN,
            "student": cls.INTERN,
            "undergraduate": cls.INTERN,
            "entry": cls.JUNIOR,
            "entry-level": cls.JUNIOR,
            "entry level": cls.JUNIOR,
            "junior": cls.JUNIOR,
            "graduate": cls.JUNIOR,
            "grad": cls.JUNIOR,
            "associate": cls.JUNIOR,
            "mid": cls.MID,
            "mid-level": cls.MID,
            "midlevel": cls.MID,
            "intermediate": cls.MID,
            "senior": cls.SENIOR,
            "sr": cls.SENIOR,
            "lead": cls.SENIOR,
            "principal": cls.SENIOR,
            "staff": cls.SENIOR,
            "manager": cls.SENIOR,
            "director": cls.SENIOR,
        }
        # exact match first
        if v in aliases:
            return aliases[v]
        # substring match
        for key, level in aliases.items():
            if key in v:
                return level
        try:
            return cls(v)
        except ValueError:
            return cls.UNKNOWN


class JobRecord(BaseModel):
    """A single job posting flowing through the pipeline."""

    title: str = ""
    company: str = ""
    url: str = ""
    description: str = ""
    location: str = ""
    source: str = ""                       # "jobspy" | "career-ops" | "jobspy+career-ops"
    source_channel: str = ""               # Exact origin: "linkedin", "indeed", "glassdoor", "greenhouse", "lever", "ashby", "the-trackr"
    ats_provider: Optional[str] = None     # greenhouse | lever | ashby
    ats_company_id: Optional[str] = None

    # Populated downstream (Phase 5/6)
    score: Optional[int] = None
    reasoning: str = ""
    cv_path: Optional[str] = None
    contact_email: Optional[str] = None

    # Gatekeeper metadata
    flagged: bool = False                   # survived via Career-Ops Exception Rule
    flag_reason: str = ""

    def dedup_key(self) -> str:
        """Stable key for cross-track deduplication."""
        if self.url:
            parsed = urlparse(self.url.strip().lower())
            path = parsed.path.rstrip("/")
            netloc = parsed.netloc
            if netloc or path:
                return f"{netloc}{path}"
        return f"{self.company.strip().lower()}::{self.title.strip().lower()}"


class GatekeeperResult(BaseModel):
    """Outcome of the zero-token regex filter."""

    survivors: list[JobRecord] = Field(default_factory=list)  # includes flagged
    dropped: list[JobRecord] = Field(default_factory=list)

    @property
    def flagged(self) -> list[JobRecord]:
        return [j for j in self.survivors if j.flagged]

    @property
    def summary(self) -> str:
        return (
            f"{len(self.survivors)} survived "
            f"({len(self.flagged)} flagged for Exception Rule), "
            f"{len(self.dropped)} dropped"
        )


# ------------------------------------------------------------
# ATS detection / slug extraction
# ------------------------------------------------------------
_GREENHOUSE_RE = re.compile(
    r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_app\?for=)?([a-z0-9_-]+)",
    re.IGNORECASE,
)
_LEVER_RE = re.compile(r"jobs\.lever\.co/([a-z0-9_-]+)", re.IGNORECASE)
_ASHBY_RE = re.compile(r"jobs\.ashbyhq\.com/([a-z0-9_-]+)", re.IGNORECASE)


def _detect_ats(url: str) -> tuple[Optional[str], Optional[str]]:
    """Return (provider, company_id) if the URL is a known ATS board."""
    if not url:
        return None, None
    for provider, regex in (
        ("greenhouse", _GREENHOUSE_RE),
        ("lever", _LEVER_RE),
        ("ashby", _ASHBY_RE),
    ):
        m = regex.search(url)
        if m:
            return provider, m.group(1).lower()
    return None, None


def extract_ats_company_ids(jobs: list[JobRecord]) -> dict[str, set[str]]:
    """
    Extract Greenhouse/Lever/Ashby company slugs from Track A URLs so
    Track B can hit their JSON APIs directly (Dynamic Link Extraction).
    """
    result: dict[str, set[str]] = {"greenhouse": set(), "lever": set(), "ashby": set()}
    for job in jobs:
        provider, company_id = job.ats_provider, job.ats_company_id
        if not (provider and company_id):
            provider, company_id = _detect_ats(job.url)
        if provider and company_id:
            result.setdefault(provider, set()).add(company_id)
    logger.info(
        "Extracted ATS slugs -> greenhouse:%d lever:%d ashby:%d",
        len(result["greenhouse"]),
        len(result["lever"]),
        len(result["ashby"]),
    )
    return result


# ------------------------------------------------------------
# Track A: Mass-Market Aggregation (python-jobspy)
# ------------------------------------------------------------
def _safe(value) -> str:
    """Coerce a pandas cell (which may be NaN) to a clean string."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def _fetch_fallback_description(url: str) -> str:
    """
    Fetch job description using Playwright as fallback to bypass Cloudflare/CAPTCHA.

    Args:
        url: Job posting URL

    Returns:
        Cleaned job description text (empty string on failure)
    """
    if not url or not PLAYWRIGHT_AVAILABLE:
        return ""

    try:
        with sync_playwright() as p:
            # Launch chromium browser in headless mode
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Set user agent to appear more like a real browser
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            # Navigate to the URL with timeout
            page.goto(url, timeout=15000, wait_until="domcontentloaded")

            # Wait for JavaScript and bot-checks to resolve
            page.wait_for_timeout(3000)

            # Extract the text content
            text = page.inner_text("body")

            # Close the browser
            browser.close()

            # Clean up text (similar to BeautifulSoup version)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

            return cleaned_text

    except Exception as e:
        logger.warning(f"Playwright fallback scraper failed for {url}: {e}")
        return ""


def run_jobspy_track_a(
    role: str,
    location: str,
    results_wanted: int = 50,
    site_names: Optional[list[str]] = None,
    hours_old: int = 72,
) -> list[JobRecord]:
    """Scrape dense aggregators (LinkedIn/Indeed/Glassdoor) via JobSpy."""
    try:
        from jobspy import scrape_jobs  # lazy: heavy import
    except ImportError as exc:
        logger.error("python-jobspy unavailable: %s", exc)
        return []

    site_names = site_names or ["linkedin", "indeed", "glassdoor"]

    try:
        df = scrape_jobs(
            site_name=site_names,
            search_term=role,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            description_format="markdown",
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline on scrape failure
        logger.exception("JobSpy scrape failed: %s", exc)
        return []

    records: list[JobRecord] = []
    if df is None or getattr(df, "empty", True):
        logger.warning("JobSpy returned no rows")
        return records

    for _, row in df.iterrows():
        url = _safe(row.get("job_url")) or _safe(row.get("job_url_direct"))
        provider, company_id = _detect_ats(url)
        description = _safe(row.get("description"))

        # Use fallback scraper if description is empty or too short
        if not description or len(description.strip()) < 50:
            fallback_desc = _fetch_fallback_description(url)
            if fallback_desc and len(fallback_desc.strip()) >= 50:
                description = fallback_desc
                logger.info(f"Used fallback scraper for job: {url}")

        # Determine the specific source channel for JobSpy results
        source_channel = "jobspy"  # fallback
        if site_names and len(site_names) == 1:
            # If only one site was requested, use that as the channel
            source_channel = site_names[0]
        else:
            # Try to infer from the URL or other clues
            if url:
                url_lower = url.lower()
                if "linkedin.com" in url_lower:
                    source_channel = "linkedin"
                elif "indeed.com" in url_lower:
                    source_channel = "indeed"
                elif "glassdoor.com" in url_lower:
                    source_channel = "glassdoor"

        records.append(
            JobRecord(
                title=_safe(row.get("title")),
                company=_safe(row.get("company")),
                url=url,
                description=description,
                location=_safe(row.get("location")),
                source="jobspy",
                source_channel=source_channel,
                ats_provider=provider,
                ats_company_id=company_id,
            )
        )

    logger.info("Track A (JobSpy) returned %d jobs", len(records))
    return records


# ------------------------------------------------------------
# Track B: Node.js ATS engine (Santiago's scan.mjs)
# ------------------------------------------------------------
def run_nodejs_track_b(
    career_ops_dir: Path,
    config_path: Optional[Path] = None,
    company_ids: Optional[list[str]] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> subprocess.Popen:
    """
    Asynchronously trigger Santiago's `node scan.mjs`.

    Returns the live Popen handle so the orchestrator can stream stdout
    into the WebSocket console (see stream_process_output) and await exit.
    """
    career_ops_dir = Path(career_ops_dir)
    scan_script = career_ops_dir / "scan.mjs"
    if not scan_script.exists():
        logger.error("scan.mjs not found in %s", career_ops_dir)
        raise FileNotFoundError(f"scan.mjs missing at {scan_script}")

    cmd: list[str] = ["node", "scan.mjs"]
    if company_ids:
        cmd += ["--company", ",".join(company_ids)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    logger.info("Launching Track B (Node): %s (cwd=%s)", " ".join(cmd), career_ops_dir)
    proc = subprocess.Popen(
        cmd,
        cwd=str(career_ops_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,               # line-buffered
        env=env,
    )
    return proc


def stream_process_output(
    proc: subprocess.Popen,
    log_fn: Optional[Callable[[str], None]] = None,
    timeout: Optional[int] = 300,
) -> int:
    """
    Blocking helper: drain a Popen's stdout line-by-line (forwarding each
    line to log_fn) and return its exit code. Intended to be run inside
    loop.run_in_executor(...) from the async orchestrator.
    """
    try:
        if proc.stdout is not None:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                if not line:
                    continue
                if log_fn:
                    log_fn(line)
                else:
                    logger.info("[node] %s", line)
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Track B timed out after %ss; terminating", timeout)
        proc.kill()
        return -1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


# ------------------------------------------------------------
# Merge tracks (dedup JobSpy + TSV)
# ------------------------------------------------------------
# Tolerant column resolution for Track B's TSV output.
_TSV_COLUMNS = {
    "title": ["title", "job_title", "role"],
    "company": ["company", "employer", "company_name"],
    "url": ["url", "job_url", "link", "apply_url"],
    "description": ["description", "desc", "job_description", "body"],
    "location": ["location", "loc", "city"],
    "source": ["source", "provider"],
}


def _lookup(row: dict, candidates: list[str]) -> str:
    lowered = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
    for c in candidates:
        if c in lowered and str(lowered[c]).strip():
            return str(lowered[c]).strip()
    return ""


def _read_tsv_jobs(tsv_path: Path) -> list[JobRecord]:
    """Read Track B's ATS jobs from the flat scan-history.tsv."""
    records: list[JobRecord] = []
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return records
    try:
        with tsv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                url = _lookup(row, _TSV_COLUMNS["url"])
                provider, company_id = _detect_ats(url)
                records.append(
                    JobRecord(
                        title=_lookup(row, _TSV_COLUMNS["title"]),
                        company=_lookup(row, _TSV_COLUMNS["company"]),
                        url=url,
                        description=_lookup(row, _TSV_COLUMNS["description"]),
                        location=_lookup(row, _TSV_COLUMNS["location"]),
                        source=_lookup(row, _TSV_COLUMNS["source"]) or "career-ops",
                        ats_provider=provider,
                        ats_company_id=company_id,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read TSV %s: %s", tsv_path, exc)
    return records


def _merge_two(a: JobRecord, b: JobRecord) -> JobRecord:
    """Combine two duplicate records, keeping the richest fields."""
    base, other = (a, b) if len(a.description) >= len(b.description) else (b, a)
    merged = base.model_copy(deep=True)

    if not merged.description and other.description:
        merged.description = other.description
    if not merged.location and other.location:
        merged.location = other.location
    if not merged.ats_provider and other.ats_provider:
        merged.ats_provider = other.ats_provider
        merged.ats_company_id = other.ats_company_id

    sources = {s for part in (a.source, b.source) for s in part.split("+") if s}
    merged.source = "+".join(sorted(sources))
    return merged


def merge_tracks(
    jobspy_jobs: list[JobRecord],
    tsv_path: Path,
) -> list[JobRecord]:
    """Deduplicate JobSpy (Track A) + TSV (Track B) into one unified pool."""
    tsv_jobs = _read_tsv_jobs(tsv_path)

    merged: dict[str, JobRecord] = {}
    for job in [*jobspy_jobs, *tsv_jobs]:
        key = job.dedup_key()
        merged[key] = _merge_two(merged[key], job) if key in merged else job

    result = list(merged.values())
    logger.info(
        "Merged %d JobSpy + %d TSV -> %d unique jobs",
        len(jobspy_jobs),
        len(tsv_jobs),
        len(result),
    )
    return result


# ------------------------------------------------------------
# Dynamic keyword rules (Phase 3 rule generation)
# ------------------------------------------------------------
def build_keyword_rules(seniority: SeniorityLevel) -> dict[str, list[str]]:
    """Auto-generate positive/negative keyword arrays from seniority."""
    rules = {
        SeniorityLevel.INTERN: (
            ["intern", "internship", "co-op", "placement", "entry level"],
            ["senior", "lead", "manager", "staff", "principal", "director", "head of"],
        ),
        SeniorityLevel.JUNIOR: (
            ["junior", "associate", "entry level", "graduate", "new grad"],
            ["senior", "lead", "principal", "staff", "director", "head of", "10+ years"],
        ),
        SeniorityLevel.MID: (
            ["mid", "intermediate", "engineer ii", "engineer 2"],
            ["principal", "head of", "director", "vp", "intern"],
        ),
        SeniorityLevel.SENIOR: (
            ["senior", "lead", "staff", "principal"],
            ["intern", "internship", "junior"],
        ),
    }
    positive, negative = rules.get(seniority, ([], []))
    return {"positive_keywords": positive, "negative_keywords": negative}


# ------------------------------------------------------------
# Phase 3: The Zero-Token Gatekeeper (Regex Filter)
# ------------------------------------------------------------
# Intern guardrails: advanced standing / credentials the doc lists.
_INTERN_DROP_RE = re.compile(
    r"\b(penultimate|rising\s+senior|graduating\s+in\s+20\d{2}|"
    r"master'?s|m\.?sc\b|mba\b|ph\.?d)\b",
    re.IGNORECASE,
)
# Intern guardrails: senior-level job titles.
_INTERN_TITLE_DROP_RE = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|manager|head\s+of|director|"
    r"vp|vice\s+president|architect|expert|specialist|consultant|forward\s+deployed)\b",
    re.IGNORECASE,
)
# Years-of-experience detector: "5+ years", "5 years", "5+years".
_YOE_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.IGNORECASE)
# Career-Ops Exception signals: the posting itself allows portfolio/equivalent.
_YOE_EXCEPTION_RE = re.compile(
    r"\b(or\s+equivalent(?:\s+(?:experience|practical\s+experience|qualification))?|"
    r"equivalent\s+combination|equivalent\s+practical|"
    r"portfolio|open[-\s]?source|"
    r"demonstrated\s+(?:ability|experience|mastery|expertise)|"
    r"proven\s+track\s+record|strong\s+projects?)\b",
    re.IGNORECASE,
)


def _extract_max_yoe(text: str) -> Optional[int]:
    """Highest plausible YoE requirement mentioned (1-40)."""
    years = [int(m.group(1)) for m in _YOE_RE.finditer(text)]
    years = [y for y in years if 0 < y <= 40]
    return max(years) if years else None


def _evaluate_job(
    job: JobRecord,
    haystack: str,
    seniority: SeniorityLevel,
    junior_yoe_threshold: int,
) -> str:
    """Return 'drop' or 'keep' (mutating job.flagged / flag_reason)."""
    # ---- Intern track ----
    if seniority is SeniorityLevel.INTERN:
        if _INTERN_TITLE_DROP_RE.search(job.title):
            job.flag_reason = "Senior-level title unsuitable for an intern"
            return "drop"
        m = _INTERN_DROP_RE.search(haystack)
        if m:
            job.flag_reason = f"Requires advanced standing/credential: '{m.group(0)}'"
            return "drop"
        return "keep"

    # ---- Junior / Mid track (with Career-Ops Exception Rule) ----
    if seniority in (SeniorityLevel.JUNIOR, SeniorityLevel.MID):
        max_yoe = _extract_max_yoe(haystack)
        if max_yoe is not None and max_yoe >= junior_yoe_threshold:
            # Check if this is an internship override - if so, disable the YoE Exception Rule
            # Interns do not get a pass based on portfolio mastery
            is_internship_override = False  # This would be set externally if needed

            if _YOE_EXCEPTION_RE.search(haystack) and not is_internship_override:
                # DO NOT drop — flag so the LLM can apply the Exception Rule.
                job.flagged = True
                job.flag_reason = (
                    f"Requires {max_yoe}+ YoE but posting allows equivalent "
                    f"project/portfolio mastery — flagged for LLM Exception Rule"
                )
                return "keep"
            job.flag_reason = (
                f"Requires {max_yoe}+ years experience "
                f"(exceeds {junior_yoe_threshold} threshold)"
            )
            return "drop"
        return "keep"

    # ---- Senior / Unknown: no aggressive local dropping ----
    return "keep"


def run_gatekeeper(
    jobs: list[JobRecord],
    seniority: SeniorityLevel,
    junior_yoe_threshold: int = 5,
    is_internship: bool = False,
) -> GatekeeperResult:
    """
    Zero-Token Gatekeeper: local, zero-cost regex disqualification so
    expensive LLM calls are reserved for high-probability targets.
    """
    result = GatekeeperResult()
    # Internship title filter regex
    internship_title_re = re.compile(
        r"\b(intern|internship|placement|co-?op)\b", re.IGNORECASE
    )

    for job in jobs:
        # NEW: Data sanitization for empty/broken job descriptions
        if not job.description or len(job.description.strip()) < 50:
            job.flagged = False
            job.flag_reason = "Empty or invalid job description"
            result.dropped.append(job)
            continue

        # Internship mode: strictly require internship-related titles
        if is_internship and not internship_title_re.search(job.title):
            job.flagged = False
            job.flag_reason = "Title does not indicate an internship/co-op/placement role"
            result.dropped.append(job)
            continue

        haystack = f"{job.title}\n{job.description}"
        if _evaluate_job(job, haystack, seniority, junior_yoe_threshold) == "drop":
            result.dropped.append(job)
        else:
            result.survivors.append(job)

    logger.info("Gatekeeper [%s]: %s", seniority.value, result.summary)
    return result


# ------------------------------------------------------------
# Track C: The-Trackr scraper
# ------------------------------------------------------------
def run_the_trackr_scraper(
    role: str,
    location: str,
    results_wanted: int = 50,
) -> list[JobRecord]:
    """
    Scrape The-Trackr aggregator for tech internships/jobs.
    Uses Playwright if available, falls back to direct URL approach.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not available, The-Trackr scraper will be limited")
        return []

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Set user agent to appear more like a real browser
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            # Navigate to The-Trackr with search parameters
            # Try to use URL parameters first, fall back to form filling
            search_url = f"https://app.the-trackr.com/?search={role}"
            if location and location.lower() != "remote":
                search_url += f"&location={location}"

            logger.info(f"Navigating to The-Trackr: {search_url}")
            page.goto(search_url, timeout=15000, wait_until="domcontentloaded")

            # Wait for JavaScript and potential bot-checks to resolve
            page.wait_for_timeout(3000)

            # Try to extract job listings from the page
            # The-Trackr structure may vary, so we look for common patterns
            job_elements = page.query_selector_all(".job-listing, .job-card, .position, [data-job-id], .listing")

            records: list[JobRecord] = []

            for element in job_elements[:results_wanted]:  # Limit results
                try:
                    # Extract title
                    title_elem = element.query_selector("h2, h3, .title, .job-title, [data-title]")
                    title = title_elem.inner_text().strip() if title_elem else ""

                    # Extract company
                    company_elem = element.query_selector(".company, .employer, [data-company]")
                    company = company_elem.inner_text().strip() if company_elem else ""

                    # Extract URL/link
                    link_elem = element.query_selector("a[href]")
                    url = link_elem.get_attribute("href") if link_elem else ""
                    if url and not url.startswith("http"):
                        url = f"https://app.the-trackr.com{url}"

                    # Extract location
                    location_elem = element.query_selector(".location, [data-location]")
                    job_location = location_elem.inner_text().strip() if location_elem else location

                    # Extract description (if available)
                    desc_elem = element.query_selector(".description, .summary, [data-description]")
                    description = desc_elem.inner_text().strip() if desc_elem else ""

                    if title and company and url:
                        records.append(
                            JobRecord(
                                title=title,
                                company=company,
                                url=url,
                                description=description,
                                location=job_location,
                                source="the-trackr",
                                source_channel="the-trackr",
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse The-Trackr job element: {e}")
                    continue

            browser.close()
            logger.info(f"Track C (The-Trackr) returned {len(records)} jobs")
            return records

    except Exception as e:
        logger.error(f"The-Trackr scraper failed: {e}")
        return []


# ------------------------------------------------------------
# Track B+: Dynamic company discovery for ATS targeting
# ------------------------------------------------------------
def _discover_ats_companies_duckduckgo(
    role: str,
    location: str,
    max_results: int = 10,
) -> list[str]:
    """
    Discover ATS company slugs using DuckDuckGo search.
    Returns a list of company slugs for Greenhouse/Lever/Ashby.
    """
    try:
        from duckduckgo_search import DDGS

        # Construct search dorks for ATS platforms
        dorks = [
            f'"{role}" "{location}" site:greenhouse.io',
            f'"{role}" "{location}" site:lever.co',
            f'"{role}" "{location}" site:ashbyhq.com',
            f'"{role}" internship site:greenhouse.io',
            f'"{role}" internship site:lever.co',
            f'"{role}" internship site:ashbyhq.com',
        ]

        company_slugs = []

        with DDGS() as ddgs:
            for dork in dorks:
                try:
                    results = list(ddgs.text(dork, max_results=3))
                    for result in results:
                        url = result.get('href', '')
                        provider, company_id = _detect_ats(url)
                        if provider and company_id:
                            company_slugs.append((provider, company_id))
                except Exception as e:
                    import traceback
                    logger.error(f"DuckDuckGo search failed for dork '{dork}': {e}")
                    logger.error(traceback.format_exc())
                    continue

        # Deduplicate and format for return
        seen = set()
        unique_slugs = []
        for provider, company_id in company_slugs:
            if (provider, company_id) not in seen:
                seen.add((provider, company_id))
                unique_slugs.append(company_id)

        logger.info(f"Discovered {len(unique_slugs)} unique ATS company slugs via DuckDuckGo")
        return unique_slugs[:max_results]  # Limit results

    except ImportError:
        logger.error("duckduckgo-search not installed")
        return []
    except Exception as e:
        import traceback
        logger.error(f"Dynamic company discovery failed: {e}")
        logger.error(traceback.format_exc())
        return []


def run_nodejs_track_b_with_discovery(
    career_ops_dir: Path,
    config_path: Optional[Path] = None,
    role: str = "",
    location: str = "",
    extra_env: Optional[dict[str, str]] = None,
) -> int:
    """
    Enhanced Track B runner that dynamically discovers companies via DuckDuckGo
    before launching Santiago's scan.mjs.

    Blocks until scan.mjs exits (draining stdout line-by-line) and returns the
    process exit code, so the caller's subsequent scan-history.tsv read sees a
    fully-written file.
    """
    career_ops_dir = Path(career_ops_dir)
    scan_script = career_ops_dir / "scan.mjs"
    if not scan_script.exists():
        logger.error("scan.mjs not found in %s", career_ops_dir)
        raise FileNotFoundError(f"scan.mjs missing at {scan_script}")

    # Discover companies dynamically if role and location provided
    company_ids = None
    if role and location:
        try:
            discovered_slugs = _discover_ats_companies_duckduckgo(role, location)
            if discovered_slugs:
                company_ids = discovered_slugs
                logger.info(f"Using {len(company_ids)} dynamically discovered company slugs for Track B")
            else:
                logger.warning("No companies discovered via DuckDuckGo, falling back to portals.yml")
        except Exception as e:
            import traceback
            logger.error(f"Failed to discover companies via DuckDuckGo: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Falling back to portals.yml due to discovery error")
    else:
        logger.info("Role/location not provided for discovery, using portals.yml only")

    # Fallback: parse portals.yml for company slugs if no dynamic discovery
    if not company_ids and config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                portals = yaml.safe_load(f)
            if portals:
                # Extract company slugs from portals.yml
                slugs = []
                for portal in portals.get('portals', []):
                    if 'company_id' in portal:
                        slugs.append(portal['company_id'])
                    elif 'slug' in portal:
                        slugs.append(portal['slug'])
                if slugs:
                    company_ids = slugs
                    logger.info(f"Using {len(company_ids)} company slugs from portals.yml")
        except Exception as e:
            logger.warning(f"Failed to parse portals.yml: {e}")

    cmd: list[str] = ["node", "scan.mjs"]
    if company_ids:
        cmd += ["--company", ",".join(company_ids)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    logger.info("Launching Track B (Node) with discovery: %s (cwd=%s)", " ".join(cmd), career_ops_dir)
    proc = subprocess.Popen(
        cmd,
        cwd=str(career_ops_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,               # line-buffered
        env=env,
    )
    # Block and drain stdout so scan-history.tsv is fully written before returning
    return stream_process_output(
        proc,
        log_fn=lambda line: logger.info(f"[node] {line}"),
        timeout=300,
    )
