"""AI insight surfaces: cross-domain strip, morning brief, metric explainer, billing."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUser, get_current_user, require_capability
from app.db.supabase import ok, pharmacy, rows
from app.services import ai_engine, usage

router = APIRouter(prefix="/api/ai", tags=["AI Insights"])


@router.get("/strip")
def strip(scope: str = Query("dashboard"), limit: int = Query(4, ge=1, le=6),
          refresh: bool = False, user: CurrentUser = Depends(get_current_user)):
    """Cross-domain AI strip shown on every page."""
    return {"success": True, **ai_engine.cross_domain_strip(scope, limit=limit, refresh=refresh)}


@router.get("/brief")
def brief(refresh: bool = False, user: CurrentUser = Depends(get_current_user)):
    """Morning brief for the Command Center."""
    return {"success": True, **ai_engine.morning_brief(refresh=refresh)}


@router.get("/explain")
def explain(metric: str, value: Optional[str] = None, scope: str = "dashboard",
            user: CurrentUser = Depends(get_current_user)):
    """Explain any KPI on demand."""
    return {"success": True, **ai_engine.explain_metric(metric, scope=scope, value=value)}


@router.get("/status")
def status(user: CurrentUser = Depends(get_current_user)):
    from app.core.config import settings
    return ok({
        "ai_enabled": settings.AI_ENABLED,
        "azure_configured": settings.azure_configured,
        "deployment": settings.AZURE_OPENAI_DEPLOYMENT if settings.azure_configured else None,
        "api_version": settings.AZURE_OPENAI_API_VERSION,
        "mode": "model" if ai_engine.ai_available() else "rule-engine",
        "cache_minutes": settings.AI_CACHE_MINUTES,
    })


@router.post("/cache/clear")
def clear_cache(user: CurrentUser = Depends(get_current_user)):
    ai_engine.clear_cache()
    return {"success": True, "message": "AI insight cache cleared"}


@router.get("/insights/stored")
def stored(scope: Optional[str] = None, limit: int = Query(50, le=200),
           user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("ai_insights").select("*").eq("is_dismissed", False) \
        .order("generated_at", desc=True)
    if scope:
        q = q.eq("scope", scope)
    return ok(rows(q.limit(limit).execute()))


# ================================================================
# BILLING
#
# Spend is a finance concern, so this is admin-only rather than
# visible to every role.
#
# Totals are aggregated in SQL (v_ai_billing_summary) rather than
# summed in Python. Supabase caps a plain select at 1000 rows, so a
# Python-side SUM silently understates the bill once the log grows
# past that — and the wrong number still looks plausible.
# ================================================================
@router.get("/billing")
def billing(user: CurrentUser = Depends(require_capability("can_admin"))):
    """
    Token usage and spend for this organisation.

    Cost is computed per call by the database using separate input,
    cached-input and output rates, because gpt-4.1 prices output at
    four times input. A blended rate applied to total_tokens would
    materially misstate the bill.
    """
    return {"success": True, "data": usage.summary()}


@router.get("/billing/rates")
def billing_rates(user: CurrentUser = Depends(require_capability("can_admin"))):
    """The rate card currently being applied, for transparency."""
    return ok(rows(
        pharmacy("ai_model_rates").select("*").order("model_name").execute()
    ))


@router.get("/billing/recent")
def billing_recent(limit: int = Query(50, le=200),
                   user: CurrentUser = Depends(require_capability("can_admin"))):
    """Most recent individual calls — useful for spotting a runaway feature."""
    from app.core.config import settings
    data = rows(
        pharmacy("ai_usage_log")
        .select("usage_id, feature, scope, model_name, prompt_tokens, cached_tokens, "
                "completion_tokens, total_tokens, cost_usd, latency_ms, succeeded, created_at")
        .eq("organization_id", settings.ORGANIZATION_ID)
        .order("created_at", desc=True).limit(limit).execute()
    )
    return ok(data)
