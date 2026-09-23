"""
Copilot tool layer.

Write operations follow a strict DRAFT -> CONFIRM contract:
the model may only *propose* a document. Nothing is committed to the database
until the user clicks Confirm in the UI, which calls the /confirm endpoint with
the draft payload. The model's output is never trusted as a write on its own.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.supabase import equipment, pharmacy, rows

ORG = settings.ORGANIZATION_ID


# --------------------------------------------------------------- resolution
def _score(needle: str, hay: str) -> float:
    needle, hay = needle.lower().strip(), (hay or "").lower()
    if not needle or not hay:
        return 0.0
    if needle == hay:
        return 1.0
    if hay.startswith(needle):
        return 0.92
    if needle in hay:
        return 0.82
    tokens = [t for t in needle.split() if len(t) > 2]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in hay)
    return 0.6 * (hits / len(tokens)) if hits else 0.0


def resolve_entity(kind: str, query: str, limit: int = 5) -> List[dict]:
    """Fuzzy-match a human phrase to a real record. Never guesses silently."""
    query = (query or "").strip()
    if not query:
        return []

    table_map = {
        "product": ("products", "product_id", ["product_name", "generic_name", "brand_name", "product_code"]),
        "supplier": ("suppliers", "supplier_id", ["supplier_name", "supplier_code"]),
        "location": ("locations", "location_id", ["location_name", "location_code"]),
    }
    if kind not in table_map:
        return []

    table, pk, fields = table_map[kind]
    data = rows(pharmacy(table).select("*").limit(500).execute())

    scored = []
    for row in data:
        best = max((_score(query, str(row.get(f) or ""))) for f in fields)
        if best > 0.35:
            scored.append({
                "id": row[pk],
                "label": row.get(fields[0]),
                "code": row.get(fields[-1]),
                "score": round(best, 3),
                "record": row,
            })
    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


def _one(kind: str, query: str) -> Optional[dict]:
    matches = resolve_entity(kind, query, limit=1)
    return matches[0] if matches else None


def _parse_date(value: Optional[str], default_offset: int = 0) -> str:
    if not value:
        return (date.today() + timedelta(days=default_offset)).isoformat()
    text = str(value).strip().lower()
    today = date.today()
    if text in ("today", "now"):
        return today.isoformat()
    if text == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, name in enumerate(weekdays):
        if name in text:
            delta = (i - today.weekday()) % 7 or 7
            return (today + timedelta(days=delta)).isoformat()
    try:
        return datetime.fromisoformat(text[:10]).date().isoformat()
    except Exception:
        return (today + timedelta(days=default_offset)).isoformat()


# --------------------------------------------------------------- read tools
def query_inventory(risk: Optional[str] = None, product: Optional[str] = None,
                    location: Optional[str] = None, limit: int = 15) -> dict:
    q = pharmacy("inventory_ai_features").select(
        "inventory_id, product_id, product_name, location_name, batch_number, current_stock, "
        "avg_daily_demand, forecast_daily_demand, days_of_stock, forecast_days_of_stock, "
        "days_to_expiry, stockout_risk, expiry_risk, demand_pattern, confidence_score, stock_value"
    )
    if risk:
        q = q.eq("stockout_risk", risk.upper())
    if product:
        match = _one("product", product)
        if match:
            q = q.eq("product_id", match["id"])
    if location:
        match = _one("location", location)
        if match:
            q = q.eq("location_id", match["id"])
    data = rows(q.limit(limit).execute())
    return {"kind": "inventory", "count": len(data), "rows": data}


def query_expiry(level: Optional[str] = None, limit: int = 15) -> dict:
    q = pharmacy("expiry_alerts").select("*").neq("alert_status", "resolved")
    if level:
        q = q.eq("alert_level", level.lower())
    data = rows(q.order("days_to_expiry").limit(limit).execute())
    return {"kind": "expiry", "count": len(data), "rows": data}


def query_forecast(product: Optional[str] = None, limit: int = 20) -> dict:
    q = pharmacy("inventory_ai_features").select(
        "product_name, location_name, demand_pattern, model_name, confidence_score, "
        "forecast_daily_demand, predicted_demand_qty, forecast_days_of_stock, "
        "forecast_stock_gap, stockout_risk"
    )
    if product:
        match = _one("product", product)
        if match:
            q = q.eq("product_id", match["id"])
    data = rows(q.order("forecast_stock_gap", desc=True).limit(limit).execute())
    return {"kind": "forecast", "count": len(data), "rows": data}


def check_supplier(supplier: Optional[str] = None, limit: int = 10) -> dict:
    q = pharmacy("v_supplier_360").select("*")
    if supplier:
        match = _one("supplier", supplier)
        if match:
            q = q.eq("supplier_id", match["id"])
    data = rows(q.limit(limit).execute())
    return {"kind": "supplier", "count": len(data), "rows": data}


def check_equipment(location: Optional[str] = None, risk_only: bool = True, limit: int = 12) -> dict:
    q = equipment("v_equipment_readiness").select("*")
    if risk_only:
        q = q.in_("readiness_risk", ["HIGH", "MEDIUM"])
    if location:
        match = _one("location", location)
        if match:
            q = q.eq("location_id", match["id"])
    data = rows(q.order("readiness_score").limit(limit).execute())
    return {"kind": "equipment", "count": len(data), "rows": data}


def check_cold_chain(limit: int = 12) -> dict:
    data = rows(pharmacy("v_cold_chain_risk").select("*").limit(limit).execute())
    return {"kind": "cold_chain", "count": len(data), "rows": data}


def transfer_opportunities(limit: int = 10) -> dict:
    data = rows(
        pharmacy("v_expiry_transfer_opportunities")
        .select("*").order("value_preserved", desc=True).limit(limit).execute()
    )
    return {"kind": "transfer_opportunity", "count": len(data), "rows": data}


# --------------------------------------------------------------- draft tools
def draft_purchase_order(product: str, quantity: float, supplier: Optional[str] = None,
                         location: Optional[str] = None,
                         expected_delivery_date: Optional[str] = None,
                         notes: Optional[str] = None) -> dict:
    p = _one("product", product)
    if not p:
        return {"ok": False, "error": f"No product matched '{product}'.",
                "suggestions": [m["label"] for m in resolve_entity("product", product, 5)]}

    loc = _one("location", location) if location else None
    if not loc:
        loc_rows = rows(pharmacy("locations").select("*").eq("location_type", "PHARMACY").limit(1).execute())
        loc = {"id": loc_rows[0]["location_id"], "label": loc_rows[0]["location_name"]} if loc_rows else None
    if not loc:
        return {"ok": False, "error": "No location could be resolved."}

    sup = _one("supplier", supplier) if supplier else None
    sp = None
    if sup:
        sp_rows = rows(
            pharmacy("supplier_products").select("*")
            .eq("product_id", p["id"]).eq("supplier_id", sup["id"]).limit(1).execute()
        )
        sp = sp_rows[0] if sp_rows else None
    if sp is None:
        sp_rows = rows(
            pharmacy("supplier_products").select("*")
            .eq("product_id", p["id"]).order("is_preferred", desc=True).limit(1).execute()
        )
        if not sp_rows:
            return {"ok": False, "error": f"No supplier is set up for {p['label']}."}
        sp = sp_rows[0]
        sup_rec = rows(pharmacy("suppliers").select("*").eq("supplier_id", sp["supplier_id"]).limit(1).execute())
        sup = {"id": sp["supplier_id"], "label": sup_rec[0]["supplier_name"] if sup_rec else "Preferred supplier"}

    s360 = rows(pharmacy("v_supplier_360").select("*").eq("supplier_id", sup["id"]).limit(1).execute())
    s360 = s360[0] if s360 else {}

    qty = float(quantity or 0)
    unit = float(sp.get("unit_price") or 0)
    subtotal = round(unit * qty, 2)
    tax = round(subtotal * 0.0825, 2)
    lead = int(sp.get("lead_time_days") or s360.get("default_lead_time_days") or 5)
    requested = _parse_date(expected_delivery_date, lead)

    warnings: List[str] = []
    promised = (datetime.fromisoformat(requested).date() - date.today()).days
    if promised < lead:
        warnings.append(
            f"Requested delivery is {promised} days out but {sup['label']} averages a "
            f"{lead}-day lead time — delivery is unlikely to be met."
        )
    otd = s360.get("on_time_delivery_rate")
    if otd is not None and float(otd) < 88:
        alt = rows(
            pharmacy("v_supplier_360").select("supplier_name, on_time_delivery_rate")
            .gt("on_time_delivery_rate", float(otd)).order("on_time_delivery_rate", desc=True)
            .limit(1).execute()
        )
        msg = f"{sup['label']} is running {otd}% on-time delivery."
        if alt:
            msg += f" {alt[0]['supplier_name']} is at {alt[0]['on_time_delivery_rate']}%."
        warnings.append(msg)
    if s360.get("certs_expired"):
        warnings.append(f"{sup['label']} has {s360['certs_expired']} expired compliance certificate(s).")
    if s360.get("certs_expiring"):
        warnings.append(f"{sup['label']} has {s360['certs_expiring']} certificate(s) expiring within 60 days.")

    return {
        "ok": True,
        "draft_type": "purchase_order",
        "draft_id": str(uuid.uuid4()),
        "title": "Draft Purchase Order",
        "fields": [
            {"label": "Supplier", "value": sup["label"]},
            {"label": "Product", "value": p["label"]},
            {"label": "Location", "value": loc["label"]},
            {"label": "Quantity", "value": f"{qty:,.0f}"},
            {"label": "Unit price", "value": f"{settings.CURRENCY} {unit:,.2f}"},
            {"label": "Total", "value": f"{settings.CURRENCY} {subtotal + tax:,.2f}"},
            {"label": "Expected delivery", "value": requested},
        ],
        "warnings": warnings,
        "payload": {
            "organization_id": ORG,
            "supplier_id": sup["id"],
            "location_id": loc["id"],
            "product_id": p["id"],
            "quantity": qty,
            "order_date": date.today().isoformat(),
            "expected_delivery_date": requested,
            "currency_code": settings.CURRENCY,
            "subtotal": subtotal,
            "tax_amount": tax,
            "total_amount": round(subtotal + tax, 2),
            "status": "draft",
            "source": "ai_copilot",
            "notes": notes or "Drafted by MediFlow AI Copilot",
        },
    }


def draft_goods_receipt(po_number: Optional[str] = None, notes: Optional[str] = None) -> dict:
    q = pharmacy("purchase_orders").select("*")
    if po_number:
        q = q.ilike("po_number", f"%{po_number}%")
    else:
        q = q.eq("status", "approved")
    po_rows = rows(q.limit(1).execute())
    if not po_rows:
        return {"ok": False, "error": "No matching purchase order found to receive against."}

    po = po_rows[0]
    sup = rows(pharmacy("suppliers").select("supplier_name").eq("supplier_id", po["supplier_id"]).limit(1).execute())
    loc = rows(pharmacy("locations").select("location_name").eq("location_id", po["location_id"]).limit(1).execute())

    return {
        "ok": True,
        "draft_type": "goods_receipt",
        "draft_id": str(uuid.uuid4()),
        "title": "Draft Goods Receipt",
        "fields": [
            {"label": "Purchase order", "value": po["po_number"]},
            {"label": "Supplier", "value": sup[0]["supplier_name"] if sup else "—"},
            {"label": "Location", "value": loc[0]["location_name"] if loc else "—"},
            {"label": "Quantity", "value": f"{float(po.get('quantity') or 0):,.0f}"},
            {"label": "Receipt date", "value": date.today().isoformat()},
        ],
        "warnings": ([] if po["status"] in ("approved", "submitted")
                     else [f"PO {po['po_number']} is currently '{po['status']}' — normally receipts "
                           f"are raised against approved orders."]),
        "payload": {
            "organization_id": ORG,
            "purchase_order_id": po["purchase_order_id"],
            "supplier_id": po["supplier_id"],
            "location_id": po["location_id"],
            "receipt_date": date.today().isoformat(),
            "status": "pending",
            "source": "ai_copilot",
            "notes": notes or f"Receipt drafted by AI Copilot against {po['po_number']}",
        },
    }


def draft_transfer(product: str, source: Optional[str] = None, destination: Optional[str] = None,
                   quantity: Optional[float] = None, notes: Optional[str] = None) -> dict:
    p = _one("product", product)
    if not p:
        return {"ok": False, "error": f"No product matched '{product}'.",
                "suggestions": [m["label"] for m in resolve_entity("product", product, 5)]}

    opp = rows(
        pharmacy("v_expiry_transfer_opportunities").select("*")
        .eq("product_id", p["id"]).order("value_preserved", desc=True).limit(1).execute()
    )

    src = _one("location", source) if source else None
    dst = _one("location", destination) if destination else None
    qty = float(quantity) if quantity else None
    warnings, reason = [], "routine"

    if opp:
        o = opp[0]
        src = src or {"id": o["source_location_id"], "label": o["source_location_name"]}
        dst = dst or {"id": o["target_location_id"], "label": o["target_location_name"]}
        qty = qty or float(o["suggested_transfer_qty"] or 0)
        reason = "expiry_rebalance"
        warnings.append(
            f"Source batch expires in {o['days_to_expiry']} days. This transfer preserves "
            f"{settings.CURRENCY} {o['value_preserved']} that would otherwise be written off."
        )

    if not src or not dst:
        return {"ok": False, "error": "Both a source and destination location are required."}
    if src["id"] == dst["id"]:
        return {"ok": False, "error": "Source and destination locations must differ."}

    return {
        "ok": True,
        "draft_type": "transfer",
        "draft_id": str(uuid.uuid4()),
        "title": "Draft Stock Transfer",
        "fields": [
            {"label": "Product", "value": p["label"]},
            {"label": "From", "value": src["label"]},
            {"label": "To", "value": dst["label"]},
            {"label": "Quantity", "value": f"{(qty or 0):,.0f}"},
            {"label": "Reason", "value": reason.replace("_", " ").title()},
        ],
        "warnings": warnings,
        "payload": {
            "organization_id": ORG,
            "source_location_id": src["id"],
            "destination_location_id": dst["id"],
            "product_id": p["id"],
            "quantity": qty or 0,
            "transfer_date": date.today().isoformat(),
            "expected_arrival_date": (date.today() + timedelta(days=2)).isoformat(),
            "status": "pending",
            "reason": reason,
            "source": "ai_copilot",
            "notes": notes or "Transfer drafted by MediFlow AI Copilot",
        },
    }


# --------------------------------------------------------------- commit
TABLE_FOR_DRAFT = {
    "purchase_order": ("purchase_orders", "purchase_order_id", "po_number"),
    "goods_receipt": ("goods_receipts", "goods_receipt_id", "receipt_number"),
    "transfer": ("inventory_transfers", "transfer_id", "transfer_number"),
}


def commit_draft(draft_type: str, payload: dict, user_id: Optional[str] = None) -> dict:
    """Persist a user-confirmed draft. Called only from the /confirm endpoint."""
    if draft_type not in TABLE_FOR_DRAFT:
        return {"ok": False, "error": f"Unknown draft type '{draft_type}'."}

    table, pk, number_field = TABLE_FOR_DRAFT[draft_type]
    body = dict(payload)
    body.setdefault("organization_id", ORG)
    body[pk] = str(uuid.uuid4())
    if draft_type == "purchase_order" and user_id:
        body["created_by"] = user_id

    result = rows(pharmacy(table).insert(body).execute())
    if not result:
        return {"ok": False, "error": "Insert returned no rows."}

    created = result[0]
    return {
        "ok": True,
        "draft_type": draft_type,
        "record_id": created.get(pk),
        "reference": created.get(number_field),
        "record": created,
    }


# --------------------------------------------------------------- schema
TOOL_SCHEMA: List[dict] = [
    {"type": "function", "function": {
        "name": "query_inventory",
        "description": "Look up live stock positions with demand, expiry and risk classification.",
        "parameters": {"type": "object", "properties": {
            "risk": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "product": {"type": "string"},
            "location": {"type": "string"},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "query_expiry",
        "description": "List open expiry alerts, optionally filtered by level.",
        "parameters": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["critical", "warning", "info"]},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "query_forecast",
        "description": "Retrieve 30-day demand forecasts with model, pattern and confidence.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "check_supplier",
        "description": "Supplier 360: on-time delivery, fill rate, defects, lead time, compliance.",
        "parameters": {"type": "object", "properties": {
            "supplier": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "check_equipment",
        "description": "Equipment readiness: status, calibration, downtime, readiness score.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "risk_only": {"type": "boolean"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "check_cold_chain",
        "description": "Cold-chain medicines exposed by impaired refrigeration equipment.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "transfer_opportunities",
        "description": "Expiring stock that another location needs — write-offs avoidable by transfer.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "draft_purchase_order",
        "description": "Propose a purchase order. This DRAFTS only; the user must confirm to commit.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string", "description": "Product name or code"},
            "quantity": {"type": "number"},
            "supplier": {"type": "string"},
            "location": {"type": "string"},
            "expected_delivery_date": {"type": "string",
                                       "description": "ISO date or natural language e.g. 'Friday'"},
            "notes": {"type": "string"}},
            "required": ["product", "quantity"]}}},
    {"type": "function", "function": {
        "name": "draft_goods_receipt",
        "description": "Propose a goods receipt against a purchase order. Drafts only.",
        "parameters": {"type": "object", "properties": {
            "po_number": {"type": "string"}, "notes": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "draft_transfer",
        "description": "Propose a stock transfer between locations. Drafts only.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "source": {"type": "string"},
            "destination": {"type": "string"}, "quantity": {"type": "number"},
            "notes": {"type": "string"}},
            "required": ["product"]}}},
]

EXECUTORS = {
    "query_inventory": query_inventory,
    "query_expiry": query_expiry,
    "query_forecast": query_forecast,
    "check_supplier": check_supplier,
    "check_equipment": check_equipment,
    "check_cold_chain": check_cold_chain,
    "transfer_opportunities": transfer_opportunities,
    "draft_purchase_order": draft_purchase_order,
    "draft_goods_receipt": draft_goods_receipt,
    "draft_transfer": draft_transfer,
}

DRAFT_TOOLS = {"draft_purchase_order", "draft_goods_receipt", "draft_transfer"}

CAPABILITY_FOR_TOOL = {
    "draft_purchase_order": "can_create_po",
    "draft_goods_receipt": None,
    "draft_transfer": None,
}
