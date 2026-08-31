# ============================================================
# JobHunter - Phase 5: LLM Factory (litellm.Router, config-driven)
# ============================================================
#
# Replaces the previous adapter-pattern factory (OpenAI / Anthropic /
# Gemini / Groq / OmniRoute adapters) and the external OmniRoute
# gateway with a single `litellm.Router` that:
#
#   * Loads a dynamic, provider-agnostic model list from
#     `config/llm_models.yaml` (secrets via `${ENV_VAR}` substitution).
#   * Groups every deployment under one generic alias ("jobhunter")
#     so callers never need to know which concrete model is used.
#   * Load-balances across that group with
#     `routing_strategy="usage-based-routing-v2"` (true cooperative
#     sharing, not a rigid failover chain).
#   * Enforces per-deployment `rpm_limit` / `tpm_limit` (and optional
#     `rpd_limit` / `tpd_limit`) so saturated models are routed around
#     instead of crashing on HTTP 429.
#
# Public surface (unchanged for all existing callers):
#   LLMFactory.from_env()
#   factory.complete_text(system, user) -> str
#   factory.complete_json(system, user) -> dict
#   factory.complete_model(system, user, ModelCls) -> ModelCls
# ============================================================
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("jobhunter.llm_factory")

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT = 120
MAX_RETRIES = 2

CONFIG_PATH_ENV = "LLM_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_models.yaml"

# Fallback provider -> env-var mapping, used only when the YAML config
# is missing/unparseable. OmniRoute has been removed entirely.
_PROVIDER_FALLBACK: dict[str, tuple[str, str, str]] = {
    # provider   -> (api_key_env, model_env, default_model)
    "openai":    ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    "gemini":    ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.1-flash-lite"),
    "groq":      ("GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
}

# Lazy import so the rest of the app still imports cleanly even if
# `litellm` is not installed (e.g. during a partial pip install).
try:
    import litellm
    from litellm import Router
    _LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover - litellm is a hard runtime dep
    litellm = None
    Router = None
    _LITELLM_AVAILABLE = False


# ------------------------------------------------------------
# Robust JSON extraction (handles code fences / stray prose)
# ------------------------------------------------------------
def _extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if (start != -1 and end > start) else text

    return json.loads(candidate)


# ------------------------------------------------------------
# Config loading (YAML + ${ENV_VAR} substitution)
# ------------------------------------------------------------
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env(value):
    """Recursively resolve `${VAR}` placeholders in a config value."""
    if isinstance(value, str):
        def _repl(match):
            var = match.group(1)
            resolved = os.getenv(var)
            if resolved is None:
                raise ValueError(
                    f"Environment variable '{var}' referenced in llm_models.yaml "
                    "is not set."
                )
            return resolved
        return _ENV_PATTERN.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def _load_config() -> dict | None:
    """Load and resolve the router config, or return None if unavailable."""
    config_path = Path(os.getenv(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        logger.warning(
            "LLM config not found at %s; falling back to ACTIVE_LLM_PROVIDER.",
            config_path,
        )
        return None

    import yaml

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 - malformed config should fall back
        logger.warning("Failed to parse %s: %s", config_path, exc)
        return None

    if not isinstance(raw, dict) or not raw.get("model_list"):
        logger.warning("llm_models.yaml is missing a 'model_list' section.")
        return None

    try:
        return _substitute_env(raw)
    except ValueError as exc:
        logger.warning("Config resolution failed: %s", exc)
        return None


# ------------------------------------------------------------
# Fallback single-model router (env-driven)
# ------------------------------------------------------------
def _build_fallback_router() -> tuple[Router, str]:
    provider = (os.getenv("ACTIVE_LLM_PROVIDER") or "openai").strip().lower()

    if provider not in _PROVIDER_FALLBACK:
        raise ValueError(
            f"Unknown ACTIVE_LLM_PROVIDER '{provider}'. Expected one of: "
            + ", ".join(sorted(_PROVIDER_FALLBACK))
            + "."
        )

    key_env, model_env, default_model = _PROVIDER_FALLBACK[provider]
    api_key = os.getenv(key_env)
    model = os.getenv(model_env, default_model)

    if not api_key:
        raise ValueError(f"Missing API key in '{key_env}' for provider '{provider}'.")

    # Model may already be prefixed (e.g. "openai/gpt-oss-120b"); don't
    # double-prefix. Normalise to a bare name and let litellm resolve it.
    bare_model = model.split("/", 1)[1] if "/" in model else model
    litellm_model = f"{provider}/{bare_model}"

    model_list = [
        {
            "model_name": "jobhunter",
            "litellm_params": {
                "model": litellm_model,
                "api_key": api_key,
                "timeout": DEFAULT_TIMEOUT,
            },
        }
    ]

    logger.info("Fallback router: provider=%s model=%s", provider, litellm_model)
    return Router(model_list=model_list), "jobhunter"


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------
class LLMFactory:
    def __init__(self, router, alias: str):
        self.router = router
        self.alias = alias

    # -- construction --------------------------------------------------
    @classmethod
    def from_env(cls) -> "LLMFactory":
        if not _LITELLM_AVAILABLE:
            raise RuntimeError(
                "litellm is not installed; add it to requirements.txt and "
                "reinstall dependencies."
            )

        config = _load_config()

        if config is not None:
            model_list = config["model_list"]
            alias = config.get("default_alias") or "jobhunter"

            # Derive the alias from the model_list if the config omits it.
            if not alias:
                names = {m.get("model_name") for m in model_list if m.get("model_name")}
                alias = names.pop() if len(names) == 1 else "jobhunter"

            router = Router(
                model_list=model_list,
                routing_strategy="usage-based-routing-v2",
                num_retries=MAX_RETRIES,
                timeout=DEFAULT_TIMEOUT,
                cooldown_time=5.0,
            )
            logger.info(
                "LLMFactory initialised via router: alias=%s deployments=%d",
                alias, len(model_list),
            )
        else:
            router, alias = _build_fallback_router()
            logger.info("LLMFactory initialised via fallback router: alias=%s", alias)

        return cls(router, alias)

    # -- delegation ----------------------------------------------------
    def _completion(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text output for a single request."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.router.completion(
            model=self.alias,
            messages=messages,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or ""

    def _with_retries(self, fn: Callable):
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - litellm raises many types
                last_exc = exc
                logger.warning(
                    "LLM attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc,
                )
                time.sleep(0.8 * attempt)
        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        return self._with_retries(lambda: self._completion(system_prompt, user_prompt))

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        sys = (
            system_prompt.rstrip()
            + "\n\nRespond with a single valid JSON object and nothing else."
        )

        def _run() -> dict:
            raw = self._completion(sys, user_prompt)
            return _extract_json(raw)

        return self._with_retries(_run)

    def complete_model(self, system_prompt: str, user_prompt: str, model_cls: Type[T]) -> T:
        data = self.complete_json(system_prompt, user_prompt)
        return model_cls.model_validate(data)