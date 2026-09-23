"""
AI token usage + spend recording.

Why this exists
---------------
Azure returns token counts on every completion, but they are discarded unless
something captures them. This module captures the usage object, records it, and
lets the database compute cost using the split rate card.

Cost is NOT calculated here. gpt-4.1 prices output at four times input
($8.00 vs $2.00 per million) and cached input at a quarter of input ($0.50),
so a single blended rate is always wrong. The trigger fn_ai_usage_cost applies
the three rates separately, which means a rate change only needs a row update
in ai_model_rates rather than a code deploy.

Recording never raises. A billing failure must not break a dashboard.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from app.core.config import settings
from app.db.supabase import pharmacy


def _extract(usage: Any) -> Dict[str, int]:
    """Pull token counts off an Azure OpenAI usage object."""
    if usage is None:
        return {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}

    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)

    # Cached prompt tokens bill at ~25% of the normal input rate.
    # They arrive under prompt_tokens_details, which is absent on older
    # API versions — hence the defensive lookup.
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    elif isinstance(usage, dict):
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt,
        "cached_tokens": min(cached, prompt),
        "completion_tokens": completion,
    }


def record(
    response: Any,
    feature: str,
    *,
    scope: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    latency_ms: Optional[int] = None,
    model_name: Optional[str] = None,
    succeeded: bool = True,
    error_message: Optional[str] = None,
) -> None:
    """Write one usage row. Silent on failure by design."""
    try:
        tokens = _extract(getattr(response, "usage", None))

        # Skip no-op rows unless we are recording a failure worth seeing.
        if succeeded and tokens["prompt_tokens"] == 0 and tokens["completion_tokens"] == 0:
            return

        pharmacy("ai_usage_log").insert({
            "organization_id": settings.ORGANIZATION_ID,
            "user_id": user_id,
            "feature": feature,
            "scope": scope,
            "session_id": session_id,
            "model_name": model_name or settings.AZURE_OPENAI_DEPLOYMENT,
            "deployment_type": getattr(settings, "AZURE_DEPLOYMENT_TYPE", "global_standard"),
            "prompt_tokens": tokens["prompt_tokens"],
            "cached_tokens": tokens["cached_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "latency_ms": latency_ms,
            "succeeded": succeeded,
            "error_message": (error_message or "")[:300] or None,
        }).execute()

    except Exception as exc:                                   # pragma: no cover
        print(f"[usage] could not record AI usage: {exc}")


@contextmanager
def track(feature: str, **kw):
    """
    Time a completion and record it.

        with usage.track("strip", scope=scope) as t:
            resp = client.chat.completions.create(...)
            t.response = resp
    """
    class _Tracker:
        response: Any = None

    t = _Tracker()
    started = time.perf_counter()
    error: Optional[str] = None
    try:
        yield t
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        record(
            t.response,
            feature,
            latency_ms=int((time.perf_counter() - started) * 1000),
            succeeded=error is None,
            error_message=error,
            **kw,
        )


# --------------------------------------------------------------- reporting
def summary() -> Dict[str, Any]:
    """Billing totals for the current organisation, aggregated in SQL."""
    from app.db.supabase import rows

    org = settings.ORGANIZATION_ID
    head = rows(
        pharmacy("v_ai_billing_summary").select("*")
        .eq("organization_id", org).limit(1).execute()
    )

    if not head:
        return {
            "total_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0,
            "completion_tokens": 0, "total_amount": 0.0,
            "month_to_date_amount": 0.0, "today_amount": 0.0,
            "total_calls": 0, "failed_calls": 0, "avg_cost_per_call": 0.0,
            "avg_latency_ms": 0, "first_call_at": None, "last_call_at": None,
            "currency": "USD", "model": settings.AZURE_OPENAI_DEPLOYMENT,
            "by_feature": [], "daily": [], "rate_card": _rate_card(),
        }

    data = dict(head[0])
    data["currency"] = "USD"
    data["model"] = settings.AZURE_OPENAI_DEPLOYMENT
    data["by_feature"] = rows(
        pharmacy("v_ai_billing_by_feature").select("*")
        .eq("organization_id", org).order("total_amount", desc=True).execute()
    )
    data["daily"] = rows(
        pharmacy("v_ai_billing_daily").select("*")
        .eq("organization_id", org).order("usage_date").execute()
    )
    data["rate_card"] = _rate_card()
    return data


def _rate_card() -> Optional[dict]:
    """The rates actually being charged, so the UI can show its working."""
    from app.db.supabase import rows
    found = rows(
        pharmacy("ai_model_rates").select("*")
        .eq("model_name", settings.AZURE_OPENAI_DEPLOYMENT).limit(1).execute()
    )
    return found[0] if found else None
