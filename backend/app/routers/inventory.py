"""Inventory intelligence — backed by the inventory_ai_features view."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.db.supabase import ok, pharmacy, rows
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/api/inventory", tags=["Inventory"])

SELECT_COLS = (
    "inventory_id, organization_id, product_id, batch_id, location_id, product_name, product_code, "
    "abc_class, cold_chain, location_name, batch_number, lot_number, manufacturing_date, expiry_date, "
    "unit_cost, current_stock, available_qty, reserved_qty, days_to_expiry, total_dispensed_qty, "
    "transaction_count, observation_days, avg_daily_demand, demand_stddev, last_dispensed_at, "
    "forecast_date, forecast_horizon_days, predicted_demand_qty, lower_bound_qty, upper_bound_qty, "
    "model_name, model_version, confidence_score, demand_pattern, forecast_status, "
    "forecast_daily_demand, reorder_point_qty, safety_stock_qty, days_of_stock, "
    "forecast_days_of_stock, forecast_stock_gap, stock_value, expiry_risk, stockout_risk, "
    "projected_writeoff_value"
)


@router.get("/")
def get_inventory(location_id: Optional[str] = None, product_id: Optional[str] = None,
                  risk: Optional[str] = None, cold_chain: Optional[bool] = None,
                  limit: int = Query(600, le=2000),
                  user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("inventory_ai_features").select(SELECT_COLS)
    if location_id:
        q = q.eq("location_id", location_id)
    if product_id:
        q = q.eq("product_id", product_id)
    if risk:
        q = q.eq("stockout_risk", risk.upper())
    if cold_chain is not None:
        q = q.eq("cold_chain", cold_chain)
    return ok(rows(q.limit(limit).execute()))


@router.get("/risk-analysis/")
def risk_analysis(limit: int = Query(600, le=2000),
                  user: CurrentUser = Depends(get_current_user)):
    data = rows(
        pharmacy("inventory_ai_features").select(SELECT_COLS)
        .in_("stockout_risk", ["HIGH", "MEDIUM", "LOW"]).limit(limit).execute()
    )
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "EXPIRED": 0}
    data.sort(key=lambda r: (order.get(r.get("stockout_risk"), 3),
                             order.get(r.get("expiry_risk"), 3)))
    return ok(data)


@router.get("/cold-chain/")
def cold_chain(user: CurrentUser = Depends(get_current_user)):
    data = rows(
        pharmacy("v_cold_chain_risk")
        .select("*")
        .limit(200)
        .execute()
    )
    return ok(data)


@router.get("/transfer-opportunities/")
def transfer_ops(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(
        pharmacy("v_expiry_transfer_opportunities").select("*")
        .order("value_preserved", desc=True).limit(50).execute()
    ))


@router.get("/procurement-risk/")
def procurement_risk(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(pharmacy("v_procurement_risk").select("*").limit(200).execute()))


@router.get("/summary/")
def summary(user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("v_estate_summary").select("*").limit(1).execute())
    return ok(data[0] if data else {})
