# ============================================================
# JobHunter - Phase 5: Dual-Track Evaluator
# ============================================================
import asyncio
import logging
from typing import Callable, Optional

from pydantic import BaseModel, Field, field_validator

from agents.sourcer import JobRecord, SeniorityLevel


logger = logging.getLogger("jobhunter.evaluator")


# ------------------------------------------------------------
# Deterministic structured output
# ------------------------------------------------------------
class EvaluationResult(BaseModel):
    score: int = Field(..., ge=1, le=10)
    reasoning: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v):
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return 1
        return max(1, min(10, iv))  # clamp into 1..10

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v):
        return "" if v is None else str(v)


# ------------------------------------------------------------
# Context-driven system prompts
# ------------------------------------------------------------
_INTERN_SYSTEM = (
    "You are a meticulous early-career technical recruiter evaluating a candidate "
    "for an INTERNSHIP. Explicitly DISREGARD any lack of multi-year commercial "
    "experience — first-year students are expected to have none. Focus entirely on "
    "technical alignment: does this student possess the foundational competencies "
    "(e.g., Python, SQL, relevant tooling) and standalone project depth required to "
    "deliver immediate value in THIS internship?\n"
    "Score the fit from 1 (no alignment) to 10 (perfect alignment).\n"
    'Return ONLY JSON: {"score": <int 1-10>, "reasoning": "<one concise sentence>"}'
)

_PROFESSIONAL_SYSTEM = (
    "You are a meticulous senior technical recruiter applying the 'Career-Ops Method' "
    "to evaluate an experienced candidate. Check the role's Years-of-Experience (YoE) "
    "requirement against the candidate's background.\n"
    "EXCEPTION RULE: if the candidate's project portfolio and structural knowledge "
    "demonstrate undeniable mastery of the exact target architecture, you MAY override "
    "strict YoE minimums and pass the role.\n"
    "Score the fit from 1 (no alignment) to 10 (perfect alignment).\n"
    'Return ONLY JSON: {"score": <int 1-10>, "reasoning": "<one concise sentence>"}'
)

_FLAG_ADDENDUM = (
    "\n\nATTENTION — PRE-FILTER FLAG: A local gatekeeper flagged this posting because it "
    "requires MORE years of experience than the candidate nominally has, BUT the posting "
    "explicitly accepts equivalent project/portfolio mastery. You MUST weigh the "
    "candidate's demonstrated portfolio depth and architectural mastery heavily against "
    "the missing YoE, and apply the Exception Rule where the evidence justifies it."
)


def build_system_prompt(seniority: SeniorityLevel, flagged: bool = False) -> str:
    """Select the Intern vs Professional track prompt (+ Exception addendum)."""
    if seniority is SeniorityLevel.INTERN:
        return _INTERN_SYSTEM
    prompt = _PROFESSIONAL_SYSTEM
    if flagged:
        prompt += _FLAG_ADDENDUM
    return prompt


def build_user_prompt(cv_text: str, story_bank: str, job: JobRecord) -> str:
    cv = (cv_text or "")[:6000]
    stories = (story_bank or "").strip()[:3000] or "(none provided)"
    jd = (job.description or "")[:6000]
    return (
        f"=== CANDIDATE CV ===\n{cv}\n\n"
        f"=== CANDIDATE STORY BANK ===\n{stories}\n\n"
        f"=== TARGET ROLE ===\n"
        f"Title: {job.title}\nCompany: {job.company}\nLocation: {job.location}\n\n"
        f"=== JOB DESCRIPTION ===\n{jd}\n"
    )


# ------------------------------------------------------------
# Single + batch evaluation
# ------------------------------------------------------------
def evaluate_job(
    job: JobRecord,
    seniority: SeniorityLevel,
    cv_text: str,
    story_bank: str,
    factory,
) -> JobRecord:
    """Evaluate one job; mutates and returns the JobRecord."""
    system = build_system_prompt(seniority, job.flagged)
    user = build_user_prompt(cv_text, story_bank, job)
    try:
        result: EvaluationResult = factory.complete_model(system, user, EvaluationResult)
        job.score = result.score
        job.reasoning = result.reasoning or "(no reasoning provided)"
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline on one job
        logger.warning("Evaluation failed for '%s' @ %s: %s", job.title, job.company, exc)
        job.score = 0  # sentinel -> routed to rejected
        job.reasoning = f"Evaluation error: {exc}"
    return job


async def evaluate_batch(
    jobs: list[JobRecord],
    seniority: SeniorityLevel,
    cv_text: str,
    story_bank: str,
    factory,
    *,
    max_concurrent: int = 1,
    batch_delay: float = 35.0,
    on_evaluated: Optional[Callable[[JobRecord], None]] = None,
) -> list[JobRecord]:
    """
    Evaluate jobs with strict concurrency control for free-tier gateways.

    - asyncio.Semaphore(max_concurrent) hard-caps in-flight LLM calls at 1.
    - Each worker holds its slot for `batch_delay`s (asyncio.sleep) AFTER the
      call, so the gateway is never hit faster than ~max_concurrent/batch_delay.
    - The blocking (sync httpx) call is offloaded via run_in_executor so the
      event loop keeps streaming logs to the UI.

    Order is preserved. One job failing never sinks the run.
    """
    if not jobs:
        return []

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)          # <-- max 3 concurrent
    results: list[Optional[JobRecord]] = [None] * len(jobs)

    async def _worker(idx: int, job: JobRecord) -> None:
        async with sem:                              # never > 3 in flight
            evaluated = await loop.run_in_executor(
                None, evaluate_job, job, seniority, cv_text, story_bank, factory
            )
            results[idx] = evaluated

            if on_evaluated:
                try:
                    on_evaluated(evaluated)
                except Exception:  # noqa: BLE001
                    logger.exception("on_evaluated callback error")

            # Pace the free-tier API: keep the slot busy for batch_delay seconds
            # so we don't fire the next request the instant this one returns.
            await asyncio.sleep(batch_delay)         # <-- 2s between executions

    await asyncio.gather(*(_worker(i, job) for i, job in enumerate(jobs)))
    return [j for j in results if j is not None]

def route_by_score(
    jobs: list[JobRecord],
    threshold: int = 7,
) -> tuple[list[JobRecord], list[JobRecord]]:
    """Split evaluated jobs into (aligned >= threshold, rejected < threshold)."""
    aligned = [j for j in jobs if (j.score or 0) >= threshold]
    rejected = [j for j in jobs if (j.score or 0) < threshold]
    logger.info(
        "Routing @ threshold %d: %d aligned, %d rejected",
        threshold, len(aligned), len(rejected),
    )
    return aligned, rejected
