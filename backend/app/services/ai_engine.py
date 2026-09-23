"""
Azure OpenAI reasoning layer.

Division of responsibility:
  * Numbers come from SQL views and the forecasting service.
  * The model reads those numbers and writes the narrative, the cross-domain
    connection and the recommended action.
The model is never asked to invent a figure.

Route safety: action_route is model-generated, so it is validated against a
whitelist of real application routes before it ever reaches the browser.
Anything unresolvable is dropped rather than shipped as a dead link.

Metering: every call goes through complete(), which records prompt, cached and
completion tokens to ai_usage_log. Cost is computed by a database trigger using
a three-part rate card, because input, cached input and output are priced
differently — a single blended rate materially understates the bill.

If Azure is not configured the module degrades gracefully to deterministic
rule-based insights so the dashboard still demos end to end.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.supabase import equipment, pharmacy, rows
from app.services import usage

_client = None
_cache: Dict[str, Any] = {}

# --------------------------------------------------------------- routes
# Must stay in sync with the ROUTES table in frontend/src/App.tsx
VALID_ROUTES = {
    "/dashboard", "/inventory", "/forecast", "/operations", "/risk-center",
    "/expiry-alerts", "/ai-copilot", "/purchase-orders", "/goods-receipts",
    "/transfers", "/dispensing", "/products", "/suppliers", "/equipment",
    "/locations", "/admin", "/audit-logs", "/settings",
}

ROUTE_ALIASES = {
    "risk": "/risk-center", "riskcenter": "/risk-center", "stockout": "/risk-center",
    "cold-chain": "/risk-center", "coldchain": "/risk-center",
    "procurement-risk": "/risk-center", "impact-center": "/risk-center",
    "expiry": "/expiry-alerts", "alerts": "/expiry-alerts", "expiry-alert": "/expiry-alerts",
    "po": "/purchase-orders", "purchase-order": "/purchase-orders",
    "procurement": "/purchase-orders", "orders": "/purchase-orders",
    "reorder": "/purchase-orders", "replenishment": "/purchase-orders",
    "goods-receipt": "/goods-receipts", "receipts": "/goods-receipts", "grn": "/goods-receipts",
    "transfer": "/transfers", "rebalance": "/transfers",
    "equipment-readiness": "/equipment", "assets": "/equipment", "asset": "/equipment",
    "readiness": "/equipment", "maintenance": "/equipment", "calibration": "/equipment",
    "supplier": "/suppliers", "supplier-360": "/suppliers", "vendors": "/suppliers",
    "stock": "/inventory", "inventory-overview": "/inventory", "batches": "/inventory",
    "forecasts": "/forecast", "demand": "/forecast", "demand-forecast": "/forecast",
    "command-center": "/dashboard", "home": "/dashboard", "overview": "/dashboard",
    "copilot": "/ai-copilot", "chat": "/ai-copilot",
    "audit": "/audit-logs", "logs": "/audit-logs",
    "product": "/products", "location": "/locations",
}


def normalise_route(raw: Optional[str]) -> Optional[str]:
    """Map a model-generated route onto a real one, or return None."""
    if not raw:
        return None
    key = str(raw).strip().lower().split("?")[0].split("#")[0]
    key = key.replace(" ", "-").strip("/")
    if not key:
        return None

    candidate = f"/{key}"
    if candidate in VALID_ROUTES:
        return candidate
    if key in ROUTE_ALIASES:
        return ROUTE_ALIASES[key]

    first = key.split("/")[0]
    if f"/{first}" in VALID_ROUTES:
        return f"/{first}"
    if first in ROUTE_ALIASES:
        return ROUTE_ALIASES[first]

    for route in VALID_ROUTES:
        bare = route.lstrip("/")
        if bare in key or key in bare:
            return route
    return None


def sanitise_insights(insights: List[dict]) -> List[dict]:
    """Repair model output: valid routes, bounded fields, sane defaults."""
    cleaned: List[dict] = []
    for item in insights or []:
        if not isinstance(item, dict) or not item.get("title"):
            continue

        route = normalise_route(item.get("action_route"))
        label = (item.get("action_label") or "").strip()

        # a label without a usable route is a dead button — drop both
        if not route:
            label, route = None, None
        elif not label:
            label = "Open"

        severity = str(item.get("severity", "info")).lower()
        if severity not in ("critical", "warning", "info", "success"):
            severity = "info"

        domains = [
            d for d in (item.get("linked_domains") or [])
            if str(d).lower() in ("medicines", "equipment", "suppliers")
        ]

        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8

        cleaned.append({
            "title": str(item["title"])[:80],
            "body": str(item.get("body", ""))[:280],
            "severity": severity,
            "linked_domains": domains or ["medicines"],
            "evidence": [str(e)[:90] for e in (item.get("evidence") or [])][:3],
            "action_label": label[:24] if label else None,
            "action_route": route,
            "confidence": max(0.0, min(1.0, confidence)),
        })
    return cleaned


def get_client():
    global _client
    if _client is None and settings.azure_configured:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    return _client


def ai_available() -> bool:
    return settings.AI_ENABLED and settings.azure_configured


# --------------------------------------------------------------- low level
def complete(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 900,
    tools: Optional[list] = None,
    json_mode: bool = False,
    feature: str = "general",
    scope: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Single entry point to Azure OpenAI.

    Every call is metered here. Token counts arrive on response.usage and are
    written to ai_usage_log; the database applies the split rate card. Failed
    calls are logged too, with zero tokens, so a spike in errors is visible on
    the billing page rather than silent.
    """
    client = get_client()
    if client is None:
        return None

    kwargs: Dict[str, Any] = {
        "model": settings.AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        usage.record(
            None, feature, scope=scope, user_id=user_id, session_id=session_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            succeeded=False, error_message=str(exc),
        )
        raise

    usage.record(
        response, feature, scope=scope, user_id=user_id, session_id=session_id,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return response


def complete_json(
    system: str,
    user: str,
    temperature: float = 0.25,
    feature: str = "general",
    scope: Optional[str] = None,
) -> Optional[dict]:
    try:
        resp = complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            json_mode=True,
            feature=feature,
            scope=scope,
        )
        if resp is None:
            return None
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:  # pragma: no cover
        print(f"[ai_engine] JSON completion failed: {exc}")
        return None


def _cached(key: str, builder):
    entry = _cache.get(key)
    ttl = settings.AI_CACHE_MINUTES * 60
    if entry and (time.time() - entry["at"]) < ttl:
        return entry["value"]
    value = builder()
    _cache[key] = {"at": time.time(), "value": value}
    return value


def clear_cache():
    _cache.clear()


# --------------------------------------------------------------- context
def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def estate_snapshot() -> dict:
    data = rows(pharmacy("v_estate_summary").select("*").limit(1).execute())
    return data[0] if data else {}


def build_context(scope: str, focus_id: Optional[str] = None) -> dict:
    """Assemble the compact, factual payload the model reasons over."""
    ctx: Dict[str, Any] = {
        "scope": scope,
        "today": date.today().isoformat(),
        "organization": settings.ORGANIZATION_NAME,
        "currency": settings.CURRENCY,
        "estate": estate_snapshot(),
    }

    if scope in ("dashboard", "inventory", "risk-center", "operations"):
        ctx["top_stockout_risk"] = rows(
            pharmacy("inventory_ai_features")
            .select("product_name, location_name, current_stock, forecast_days_of_stock, "
                    "avg_daily_demand, stockout_risk, demand_pattern, confidence_score")
            .eq("stockout_risk", "HIGH")
            .limit(8)
            .execute()
        )
        ctx["cold_chain_risk"] = rows(
            pharmacy("v_cold_chain_risk")
            .select("product_name, location_name, current_stock, stock_value, "
                    "asset_numbers, asset_types, cold_chain_risk")
            .eq("cold_chain_risk", "HIGH")
            .limit(6)
            .execute()
        )
        ctx["procurement_risk"] = rows(
            pharmacy("v_procurement_risk")
            .select("product_name, location_name, supplier_name, on_time_delivery_rate, "
                    "average_lead_time_days, forecast_days_of_stock, lead_time_risk")
            .eq("lead_time_risk", "HIGH")
            .limit(6)
            .execute()
        )

    if scope in ("expiry-alerts", "dashboard", "risk-center", "transfers"):
        ctx["transfer_opportunities"] = rows(
            pharmacy("v_expiry_transfer_opportunities")
            .select("product_name, source_location_name, target_location_name, days_to_expiry, "
                    "suggested_transfer_qty, value_preserved, target_stockout_risk")
            .order("value_preserved", desc=True)
            .limit(6)
            .execute()
        )

    if scope in ("purchase-orders", "suppliers", "goods-receipts", "dashboard"):
        ctx["supplier_health"] = rows(
            pharmacy("v_supplier_360")
            .select("supplier_name, on_time_delivery_rate, fill_rate, defect_rate, "
                    "average_lead_time_days, certs_expiring, certs_expired, supplier_risk, "
                    "performance_status, po_count")
            .order("on_time_delivery_rate")
            .limit(8)
            .execute()
        )

    if scope in ("equipment", "dashboard", "risk-center"):
        ctx["equipment_risk"] = rows(
            equipment("v_equipment_readiness")
            .select("asset_number, equipment_type_name, location_name, status, "
                    "calibration_status, days_to_calibration, readiness_score, "
                    "readiness_risk, is_cold_chain, open_downtime_events")
            .eq("readiness_risk", "HIGH")
            .limit(8)
            .execute()
        )

    if scope == "forecast":
        ctx["forecast_leaders"] = rows(
            pharmacy("inventory_ai_features")
            .select("product_name, location_name, demand_pattern, model_name, confidence_score, "
                    "forecast_daily_demand, forecast_days_of_stock, forecast_stock_gap, stockout_risk")
            .order("forecast_stock_gap", desc=True)
            .limit(10)
            .execute()
        )

    if focus_id:
        ctx["focus_id"] = focus_id
    return ctx


# --------------------------------------------------------------- strip
STRIP_SYSTEM = """You are the MediFlow cross-domain intelligence engine for a hospital pharmacy network.

You connect THREE domains that share one database:
  1. MEDICINES  — inventory, batches, expiry, demand forecasts
  2. EQUIPMENT  — refrigerators, freezers, analysers: status, calibration, downtime
  3. SUPPLIERS  — on-time delivery, fill rate, lead time, compliance certificates

Your job is to surface insights that a single-domain dashboard CANNOT produce.
The strongest insights link at least two domains. Examples of the pattern:
  - a cold-chain medicine at a location whose refrigerator is out of service
  - a reorder whose supplier lead time exceeds the remaining days of cover
  - expiring stock at one site that another site urgently needs

STRICT RULES
- Use ONLY numbers present in the supplied context. Never invent a figure.
- Every insight must cite its evidence as short "label: value" strings.
- Be specific and operational. Name the product, location, supplier or asset.
- Keep each body to 1-2 sentences, under 40 words.
- If the context does not support an insight, return fewer cards. Never pad.

ACTION ROUTES — CRITICAL
action_route MUST be copied EXACTLY from this list. Do not invent, abbreviate
or add sub-paths. If none fits, set BOTH action_route and action_label to null.

  /dashboard        /inventory        /forecast         /operations
  /risk-center      /expiry-alerts    /ai-copilot       /purchase-orders
  /goods-receipts   /transfers        /dispensing       /products
  /suppliers        /equipment        /locations        /audit-logs

Choose by intent:
  stock-out or cold-chain risk .......... /risk-center
  batch expiring ........................ /expiry-alerts
  move stock between sites .............. /transfers
  order more stock ...................... /purchase-orders
  supplier performance or certificates .. /suppliers
  asset, calibration, downtime .......... /equipment
  demand prediction ..................... /forecast

Return JSON: {"insights":[{
  "title": str,
  "body": str,
  "severity": "critical"|"warning"|"info"|"success",
  "linked_domains": [str],
  "evidence": [str],
  "action_label": str|null,
  "action_route": str|null,
  "confidence": float
}]}

Field rules: title under 60 characters with no trailing period; linked_domains
a subset of ["medicines","equipment","suppliers"]; evidence 2-3 items each
formatted "Label: value"; action_label under 24 characters; action_route an
exact match from the list above; confidence between 0.0 and 1.0."""


def _fallback_strip(ctx: dict) -> List[dict]:
    """Deterministic insights when Azure is unavailable."""
    out: List[dict] = []
    est = ctx.get("estate", {})

    for c in (ctx.get("cold_chain_risk") or [])[:1]:
        out.append({
            "title": "Cold chain exposure detected",
            "body": f"{c.get('product_name')} is held at {c.get('location_name')} where "
                    f"{c.get('asset_types')} {c.get('asset_numbers')} is impaired.",
            "severity": "critical",
            "linked_domains": ["medicines", "equipment"],
            "evidence": [
                f"Stock at risk: {c.get('current_stock')} units",
                f"Stock value: {settings.CURRENCY} {c.get('stock_value')}",
                f"Asset: {c.get('asset_numbers')}",
            ],
            "action_label": "View equipment",
            "action_route": "/equipment",
            "confidence": 0.88,
        })

    for p in (ctx.get("procurement_risk") or [])[:1]:
        out.append({
            "title": "Lead time exceeds remaining cover",
            "body": f"{p.get('product_name')} at {p.get('location_name')} has "
                    f"{p.get('forecast_days_of_stock')} days of cover against a "
                    f"{p.get('average_lead_time_days')}-day lead time from {p.get('supplier_name')}.",
            "severity": "warning",
            "linked_domains": ["medicines", "suppliers"],
            "evidence": [
                f"Supplier on-time: {p.get('on_time_delivery_rate')}%",
                f"Lead time: {p.get('average_lead_time_days')} days",
                f"Cover: {p.get('forecast_days_of_stock')} days",
            ],
            "action_label": "Create PO",
            "action_route": "/purchase-orders",
            "confidence": 0.84,
        })

    for t in (ctx.get("transfer_opportunities") or [])[:1]:
        out.append({
            "title": "Write-off avoidable by transfer",
            "body": f"{t.get('product_name')} expires in {t.get('days_to_expiry')} days at "
                    f"{t.get('source_location_name')}; {t.get('target_location_name')} is short.",
            "severity": "warning",
            "linked_domains": ["medicines"],
            "evidence": [
                f"Suggested qty: {t.get('suggested_transfer_qty')} units",
                f"Value preserved: {settings.CURRENCY} {t.get('value_preserved')}",
                f"Target risk: {t.get('target_stockout_risk')}",
            ],
            "action_label": "Draft transfer",
            "action_route": "/transfers",
            "confidence": 0.86,
        })

    for s in (ctx.get("supplier_health") or [])[:1]:
        if _num(s.get("on_time_delivery_rate"), 100) < 88 or s.get("certs_expired"):
            out.append({
                "title": "Supplier reliability below tolerance",
                "body": f"{s.get('supplier_name')} is running "
                        f"{s.get('on_time_delivery_rate')}% on-time with a "
                        f"{s.get('average_lead_time_days')}-day average lead time.",
                "severity": "warning",
                "linked_domains": ["suppliers", "medicines"],
                "evidence": [
                    f"On-time delivery: {s.get('on_time_delivery_rate')}%",
                    f"Fill rate: {s.get('fill_rate')}%",
                    f"Certificates expiring: {s.get('certs_expiring')}",
                ],
                "action_label": "Review supplier",
                "action_route": "/suppliers",
                "confidence": 0.82,
            })

    for e in (ctx.get("equipment_risk") or [])[:1]:
        out.append({
            "title": "Asset readiness below threshold",
            "body": f"{e.get('asset_number')} ({e.get('equipment_type_name')}) at "
                    f"{e.get('location_name')} is {e.get('status')} with "
                    f"calibration {e.get('calibration_status')}.",
            "severity": "critical" if e.get("is_cold_chain") else "warning",
            "linked_domains": ["equipment", "medicines"] if e.get("is_cold_chain") else ["equipment"],
            "evidence": [
                f"Readiness score: {e.get('readiness_score')}",
                f"Calibration: {e.get('calibration_status')}",
                f"Open downtime: {e.get('open_downtime_events')}",
            ],
            "action_label": "Open asset",
            "action_route": "/equipment",
            "confidence": 0.85,
        })

    if est and not out:
        out.append({
            "title": "Estate operating within tolerance",
            "body": f"{est.get('high_stockout_risk', 0)} high stockout risks and "
                    f"{est.get('equipment_high_risk', 0)} equipment risks across the network.",
            "severity": "success",
            "linked_domains": ["medicines", "equipment", "suppliers"],
            "evidence": [
                f"Positions: {est.get('inventory_positions', 0)}",
                f"Equipment readiness: {est.get('equipment_avg_readiness', 0)}",
                f"Supplier on-time: {est.get('supplier_avg_on_time', 0)}%",
            ],
            "action_label": None,
            "action_route": None,
            "confidence": 0.9,
        })
    return out


def cross_domain_strip(scope: str, limit: int = 4, refresh: bool = False) -> dict:
    def build() -> dict:
        ctx = build_context(scope)
        insights: List[dict] = []
        source = "rules"

        if ai_available():
            result = complete_json(
                STRIP_SYSTEM,
                f"Produce at most {limit} cross-domain insights for the '{scope}' page.\n\n"
                f"CONTEXT:\n{json.dumps(ctx, default=str)[:12000]}",
                feature="strip",
                scope=scope,
            )
            if result and isinstance(result.get("insights"), list):
                insights = sanitise_insights(result["insights"])[:limit]
                source = "model"

        if not insights:
            insights = sanitise_insights(_fallback_strip(ctx))[:limit]

        return {
            "scope": scope,
            "generated_at": datetime.utcnow().isoformat(),
            "model": settings.AZURE_OPENAI_DEPLOYMENT if source == "model" else "rule-engine",
            "source": source,
            "insights": insights,
        }

    if refresh:
        _cache.pop(f"strip:{scope}", None)
    return _cached(f"strip:{scope}", build)


# --------------------------------------------------------------- morning brief
BRIEF_SYSTEM = """You are the duty intelligence officer for a hospital pharmacy network.
Write a concise operational brief for the team starting their shift.

Format exactly:
- One opening sentence stating overall estate posture.
- Then exactly three bullet lines, each beginning with "• ", each naming a
  specific product, location, supplier or asset and the action to take.

Use only figures from the context. Under 110 words total. No headings, no preamble."""


def morning_brief(refresh: bool = False) -> dict:
    def build() -> dict:
        ctx = build_context("dashboard")
        est = ctx.get("estate", {})
        text, source = None, "rules"

        if ai_available():
            try:
                resp = complete(
                    [
                        {"role": "system", "content": BRIEF_SYSTEM},
                        {"role": "user", "content": json.dumps(ctx, default=str)[:11000]},
                    ],
                    temperature=0.4,
                    max_tokens=320,
                    feature="brief",
                    scope="dashboard",
                )
                if resp:
                    text = resp.choices[0].message.content.strip()
                    source = "model"
            except Exception as exc:  # pragma: no cover
                print(f"[ai_engine] morning brief failed: {exc}")

        if not text:
            text = (
                f"Estate is stable with {est.get('high_stockout_risk', 0)} high stockout risks "
                f"and {est.get('equipment_high_risk', 0)} equipment assets needing attention.\n"
                f"• Review {est.get('high_stockout_risk', 0)} high-risk items on the Risk Center.\n"
                f"• {est.get('cold_chain_exposures', 0)} cold-chain exposures need equipment follow-up.\n"
                f"• {settings.CURRENCY} {est.get('rebalance_value_opportunity', 0)} recoverable "
                f"through expiry rebalancing transfers."
            )

        return {
            "brief": text,
            "estate": est,
            "source": source,
            "model": settings.AZURE_OPENAI_DEPLOYMENT if source == "model" else "rule-engine",
            "generated_at": datetime.utcnow().isoformat(),
        }

    if refresh:
        _cache.pop("brief", None)
    return _cached("brief", build)


# --------------------------------------------------------------- explain a KPI
EXPLAIN_SYSTEM = """You explain a single dashboard metric to a pharmacy operations user.
Structure your answer as:
  1. What the number means in one plain sentence.
  2. How it is calculated (name the underlying fields).
  3. What is driving it right now, citing specific records from the context.
  4. One recommended next action.
Use only supplied figures. Under 130 words. No markdown headings."""


def explain_metric(metric: str, scope: str = "dashboard", value: Optional[str] = None) -> dict:
    ctx = build_context(scope)
    text, source = None, "rules"

    if ai_available():
        try:
            resp = complete(
                [
                    {"role": "system", "content": EXPLAIN_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Metric: {metric}\nDisplayed value: {value}\n\n"
                                   f"CONTEXT:\n{json.dumps(ctx, default=str)[:11000]}",
                    },
                ],
                temperature=0.3,
                max_tokens=380,
                feature="explain",
                scope=scope,
            )
            if resp:
                text = resp.choices[0].message.content.strip()
                source = "model"
        except Exception as exc:  # pragma: no cover
            print(f"[ai_engine] explain failed: {exc}")

    if not text:
        est = ctx.get("estate", {})
        text = (
            f"'{metric}' currently reads {value}. It is derived from the inventory_ai_features "
            f"view, which joins live stock against 90-day demand and the active 30-day forecast. "
            f"Across the estate there are {est.get('high_stockout_risk', 0)} high stockout risks "
            f"and {est.get('high_expiry_risk', 0)} high expiry risks. "
            f"Next action: open the Risk Center and triage the high-risk positions first."
        )

    return {
        "metric": metric,
        "value": value,
        "explanation": text,
        "source": source,
        "model": settings.AZURE_OPENAI_DEPLOYMENT if source == "model" else "rule-engine",
        "generated_at": datetime.utcnow().isoformat(),
    }


# --------------------------------------------------------------- supplier verdict
def supplier_verdict(supplier: dict) -> dict:
    system = (
        "You are a procurement analyst. In at most 45 words give a blunt, decisive verdict on "
        "this supplier: what they are reliable for and what to avoid using them for. "
        "Use only the supplied metrics."
    )
    text, source = None, "rules"

    if ai_available():
        try:
            resp = complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(supplier, default=str)[:4000]},
                ],
                temperature=0.35,
                max_tokens=160,
                feature="verdict",
                scope="suppliers",
            )
            if resp:
                text = resp.choices[0].message.content.strip()
                source = "model"
        except Exception as exc:  # pragma: no cover
            print(f"[ai_engine] supplier verdict failed: {exc}")

    if not text:
        otd = _num(supplier.get("on_time_delivery_rate"), 0)
        defect = _num(supplier.get("defect_rate"), 0)
        strong = "quality" if defect < 2 else "availability"
        weak = "delivery reliability" if otd < 90 else "pricing"
        text = (
            f"{supplier.get('supplier_name')} scores {otd}% on-time with a {defect}% defect rate. "
            f"Dependable on {strong}; weaker on {weak}. "
            f"{'Avoid for urgent replenishment.' if otd < 88 else 'Suitable for routine replenishment.'}"
        )

    return {
        "verdict": text,
        "source": source,
        "model": settings.AZURE_OPENAI_DEPLOYMENT if source == "model" else "rule-engine",
    }
