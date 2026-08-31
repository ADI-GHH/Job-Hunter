# ============================================================
# JobHunter - Central Orchestrator (background pipeline)
#
# profile_cv -> [Track A + Track B] -> merge -> gatekeeper
#   -> evaluate_batch -> route_by_score -> action_layer
#   -> results delivered to the UI:
#        * aligned  -> results TSV -> file_watcher -> UI card
#        * rejected -> manager.result (direct)
# ============================================================
from __future__ import annotations

import asyncio
import csv
import functools
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from agents.action_layer import produce_assets
from agents.cv_profiler import profile_cv
from agents.evaluator import evaluate_batch, route_by_score
from agents.file_watcher import start_file_watcher
from agents.sourcer import (
    JobRecord,
    SeniorityLevel,
    _read_tsv_jobs,
    build_keyword_rules,
    extract_ats_company_ids,
    merge_tracks,
    run_gatekeeper,
    run_jobspy_track_a,
    run_nodejs_track_b,
    run_nodejs_track_b_with_discovery,
    run_the_trackr_scraper,
    stream_process_output,
)

logger = logging.getLogger("jobhunter.orchestrator")

# Canonical results-TSV schema (columns the file_watcher understands).
_RESULTS_HEADER = [
    "timestamp", "session_id", "company", "title", "url",
    "score", "reasoning", "status", "cv_path", "contact_email",
    "source_channel",
]


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _job_to_payload(job: JobRecord) -> dict:
    """JobRecord -> frontend result payload."""
    return {
        "title": job.title,
        "company": job.company,
        "url": job.url,
        "score": job.score if job.score is not None else 0,
        "reasoning": job.reasoning,
        "cv_path": job.cv_path,
        "contact_email": job.contact_email,
        "source_channel": job.source_channel,
    }


def _init_results_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerow(_RESULTS_HEADER)


def _append_result_row(path: Path, session_id: str, job: JobRecord, status: str) -> None:
    """Append an aligned row for the file_watcher to translate into a card."""
    clean_reason = (job.reasoning or "").replace("\t", " ").replace("\n", " ")
    with path.open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerow([
            datetime.now(timezone.utc).isoformat(),
            session_id,
            job.company,
            job.title,
            job.url,
            job.score if job.score is not None else "",
            clean_reason,
            status,
            job.cv_path or "",
            job.contact_email or "",
            job.source_channel or "",
        ])


def _run_track_b_blocking(career_ops_dir, config_path, company_ids, ts_log) -> int:
    """Blocking Track B runner (launch Node + stream stdout). For an executor."""
    try:
        proc = run_nodejs_track_b(
            career_ops_dir, config_path=config_path, company_ids=company_ids
        )
    except FileNotFoundError as exc:
        ts_log(f"Track B skipped: {exc}", "WARN")
        return -1
    except Exception as exc:  # noqa: BLE001
        ts_log(f"Track B launch failed: {exc}", "ERROR")
        return -1
    return stream_process_output(
        proc, log_fn=lambda line: ts_log(f"[node] {line}", "DEBUG"), timeout=300
    )


# ------------------------------------------------------------
# The background task imported by api/main.py
# ------------------------------------------------------------
async def run_orchestrator(
    session_id: str,
    upload_path: Path,
    target_role: str,
    location: str,
    is_internship: bool = False,
) -> None:
    # Lazy import avoids a circular import (api.main imports this module).
    from api.main import manager, settings

    # Issue 1 Fix: Append "Intern" to search queries if internship mode
    if is_internship and "intern" not in target_role.lower():
        target_role = f"{target_role} Intern"

    loop = asyncio.get_running_loop()
    watcher = None

    # Thread-safe log/result bridges usable from executor threads.
    def ts_log(message: str, level: str = "INFO") -> None:
        asyncio.run_coroutine_threadsafe(
            manager.log(session_id, message, level), loop
        )

    async def log(msg: str, level: str = "INFO") -> None:
        await manager.log(session_id, msg, level)

    try:
        await manager.status(session_id, "running")
        await log("Booting JobHunter pipeline…")

        # --- Session artifacts ---
        session_cfg = settings.config_dir / session_id
        portals_path = session_cfg / "portals.yml"
        story_path = session_cfg / "story_bank.txt"
        story_bank = story_path.read_text(encoding="utf-8") if story_path.exists() else ""

        # Track B (Node) writes to its own working-dir TSV — kept separate from
        # the UI results TSV to avoid schema collisions.
        track_b_tsv = settings.career_ops_dir / "scan-history.tsv"
        results_tsv = Path(settings.scan_history_path)

        # --- LLM factory (graceful if unavailable) ---
        factory = None
        try:
            from agents.llm_factory import LLMFactory

            factory = LLMFactory.from_env()
            await log(f"LLM ready: {factory.provider} ({factory.model})", "SUCCESS")
        except Exception as exc:  # noqa: BLE001
            await log(f"LLM unavailable ({exc}); using regex fallbacks", "WARN")

        # --- Phase 3a: CV profiling ---
        await log("Profiling CV…")
        profile = await loop.run_in_executor(None, profile_cv, upload_path, factory)
        seniority = profile.seniority

        # Apply internship override if requested
        if is_internship:
            original_seniority = seniority
            seniority = SeniorityLevel.INTERN
            await log(
                f"Internship override applied: {original_seniority.value} → {seniority.value}",
                "SUCCESS",
            )
        else:
            await log(
                f"Seniority: {seniority.value} "
                f"(via {profile.classification_source}, {profile.char_count} chars)",
                "SUCCESS",
            )

        rules = build_keyword_rules(seniority)
        await log(
            f"Dynamic rules → block: {', '.join(rules['negative_keywords']) or '—'}"
        )

        # --- Results TSV + file_watcher (Phase 6 translation engine) ---
        _init_results_tsv(results_tsv)
        watcher = start_file_watcher(
            tsv_path=results_tsv,
            session_id=session_id,
            broadcast_fn=manager.send,  # emits {"type":"result", ...} == manager.result
            loop=loop,
        )
        await log("Live results stream attached.")

        # --- Phase 4: Sequential sourcing with priority (Track C -> Track B -> Track A) ---
        await log(f"Sourcing '{target_role}' in '{location}'…")

        # Track C: The-Trackr scraper (highest priority)
        await log("Running Track C: The-Trackr scraper…")
        track_c_future = loop.run_in_executor(
            None, run_the_trackr_scraper, target_role, location, 50
        )
        track_c_res = await track_c_future
        track_c_jobs: list[JobRecord] = track_c_res if isinstance(track_c_res, list) else []
        if isinstance(track_c_res, Exception):
            await log(f"Track C error: {track_c_res}", "WARN")
        await log(f"Track C (The-Trackr) returned {len(track_c_jobs)} jobs.")

        # Track B: Node.js ATS engine with dynamic discovery
        await log("Running Track B: Node.js ATS engine with discovery…")
        try:
            track_b_exit_code = await loop.run_in_executor(
                None,
                run_nodejs_track_b_with_discovery,
                str(settings.career_ops_dir),
                str(portals_path),
                target_role,
                location,
            )
            if track_b_exit_code == 0:
                await log(f"Track B (Node.js) completed successfully.", "SUCCESS")
            else:
                await log(f"Track B (Node.js) exited with code {track_b_exit_code}", "WARN")
        except Exception as e:
            await log(f"Track B error: {e}", "WARN")

        # Track A: JobSpy for non-LinkedIn sites (Indeed, Glassdoor, etc.)
        await log("Running Track A: JobSpy (Indeed/Glassdoor/etc.)…")
        track_a_non_linkedin_future = loop.run_in_executor(
            None, run_jobspy_track_a, target_role, location, 50, ["indeed", "glassdoor"]
        )
        track_a_non_linkedin_res = await track_a_non_linkedin_future
        track_a_non_linkedin_jobs: list[JobRecord] = track_a_non_linkedin_res if isinstance(track_a_non_linkedin_res, list) else []
        if isinstance(track_a_non_linkedin_res, Exception):
            await log(f"Track A (non-LinkedIn) error: {track_a_non_linkedin_res}", "WARN")
        await log(f"Track A (Indeed/Glassdoor) returned {len(track_a_non_linkedin_jobs)} jobs.")

        # Track A: JobSpy for LinkedIn only (lowest priority)
        await log("Running Track A: JobSpy (LinkedIn only)…")
        track_a_linkedin_future = loop.run_in_executor(
            None, run_jobspy_track_a, target_role, location, 50, ["linkedin"]
        )
        track_a_linkedin_res = await track_a_linkedin_future
        track_a_linkedin_jobs: list[JobRecord] = track_a_linkedin_res if isinstance(track_a_linkedin_res, list) else []
        if isinstance(track_a_linkedin_res, Exception):
            await log(f"Track A (LinkedIn) error: {track_a_linkedin_res}", "WARN")
        await log(f"Track A (LinkedIn) returned {len(track_a_linkedin_jobs)} jobs.")

        # Combine all Track A results
        jobspy_jobs = track_a_non_linkedin_jobs + track_a_linkedin_jobs

        # --- Merge + dedup in priority order: C -> B -> A ---
        # Start with Track C (highest priority)
        merged: dict[str, JobRecord] = {}
        for job in track_c_jobs:
            key = job.dedup_key()
            merged[key] = job

        # Add Track B (medium priority) - only add if not already present from Track C
        tsv_jobs = _read_tsv_jobs(track_b_tsv)
        for job in tsv_jobs:
            key = job.dedup_key()
            if key not in merged:
                merged[key] = job

        # Add Track A (lowest priority) - only add if not already present from C or B
        for job in jobspy_jobs:
            key = job.dedup_key()
            if key not in merged:
                merged[key] = job

        job_list = list(merged.values())
        logger.info(
            "Merged %d Track C + %d TSV (Track B) + %d JobSpy (Track A) -> %d unique jobs",
            len(track_c_jobs),
            len(tsv_jobs),
            len(jobspy_jobs),
            len(job_list),
        )
        ats = extract_ats_company_ids(job_list)
        await log(
            f"Merged pool: {len(job_list)} unique jobs "
            f"(ATS slugs → gh:{len(ats['greenhouse'])} "
            f"lever:{len(ats['lever'])} ashby:{len(ats['ashby'])})"
        )

        if not job_list:
            await log("No jobs sourced. Nothing to evaluate.", "WARN")
            await manager.status(session_id, "done")
            return

        # --- Phase 3b: Zero-Token Gatekeeper ---
        gate = run_gatekeeper(job_list, seniority, is_internship=is_internship)
        await log(
            f"Gatekeeper: {len(gate.survivors)} survived "
            f"({len(gate.flagged)} flagged for Exception Rule), "
            f"{len(gate.dropped)} dropped.",
            "SUCCESS",
        )

        # Gatekeeper drops -> Rejected column (direct manager.result).
        for job in gate.dropped:
            job.score = 0
            job.reasoning = job.flag_reason or "Filtered by local gatekeeper"
            await manager.result(session_id, _job_to_payload(job))

        if not gate.survivors:
            await log("No survivors after gatekeeping.", "WARN")
            await manager.status(session_id, "done")
            return

        # --- Phase 5: LLM evaluation ---
        await log(f"Evaluating {len(gate.survivors)} jobs with the LLM…")

        def _on_eval(job: JobRecord) -> None:
            lvl = "SUCCESS" if (job.score or 0) >= settings.min_alignment_score else "INFO"
            ts_log(f"  ↳ {job.title} @ {job.company}: {job.score}/10", lvl)

        evaluated = await evaluate_batch(
            gate.survivors,
            seniority,
            profile.raw_text,
            story_bank,
            factory,
            max_concurrent=3,   # free-tier safe
            batch_delay=2.0,    # pause between batches of 3
            on_evaluated=_on_eval,
        )


        aligned, rejected = route_by_score(evaluated, threshold=settings.min_alignment_score)
        await log(f"{len(aligned)} aligned, {len(rejected)} rejected.", "SUCCESS")

        # Rejected -> Rejected column (direct manager.result).
        for job in rejected:
            await manager.result(session_id, _job_to_payload(job))

        # --- Phase 6: Asset production for aligned jobs (concurrent) ---
        if aligned:
            await log(f"Producing assets for {len(aligned)} aligned roles…")
            tasks = [
                loop.run_in_executor(
                    None,
                    functools.partial(
                        produce_assets,
                        job,
                        session_id=session_id,
                        career_ops_dir=str(settings.career_ops_dir),
                        cv_path=str(upload_path),
                        story_bank=story_bank,
                        data_output_dir=str(settings.data_output_dir),
                        apollo_key=os.getenv("APOLLO_API_KEY"),
                        hunter_key=os.getenv("HUNTER_API_KEY"),
                        factory=factory,
                    ),
                )
                for job in aligned
            ]
            enriched = await asyncio.gather(*tasks, return_exceptions=True)

            for original, res in zip(aligned, enriched):
                job = res if isinstance(res, JobRecord) else original
                if isinstance(res, Exception):
                    await log(
                        f"Asset production failed for {original.title}: {res}", "WARN"
                    )
                # Write to results TSV -> file_watcher streams it as an aligned card.
                _append_result_row(results_tsv, session_id, job, status="aligned")
                cv_note = "CV ✓" if job.cv_path else "CV ✗"
                contact_note = job.contact_email or "no contact"
                await log(f"  ✅ {job.title} @ {job.company} [{cv_note}, {contact_note}]", "SUCCESS")

            # Give the watcher a moment to flush the final appends.
            await asyncio.sleep(1.5)

        await log("Pipeline complete.", "SUCCESS")
        await manager.status(session_id, "done")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Orchestrator failed for %s", session_id)
        await manager.log(session_id, f"Fatal error: {exc}", "ERROR")
        await manager.status(session_id, "error")
    finally:
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:  # noqa: BLE001
                logger.debug("watcher.stop() raised", exc_info=True)
