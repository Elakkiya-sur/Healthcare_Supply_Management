"""Demand forecasting endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUser, get_current_user, require_capability
from app.db.supabase import ok, pharmacy, rows
from app.services import forecasting

router = APIRouter(prefix="/api/forecast", tags=["Forecast"])


@router.get("/")
def get_forecasts(location_id: Optional[str] = None, product_id: Optional[str] = None,
                  limit: int = Query(1200, le=4000),
                  user: CurrentUser = Depends(get_current_user)):
    """Forecast rows enriched with product name and live stock risk."""
    q = pharmacy("demand_forecasts").select("*").gte(
        "forecast_date", __import__("datetime").date.today().isoformat()
    ).order("forecast_date")
    if location_id:
        q = q.eq("location_id", location_id)
    if product_id:
        q = q.eq("product_id", product_id)
    forecasts = rows(q.limit(limit).execute())

    names = {p["product_id"]: p["product_name"]
             for p in rows(pharmacy("products").select("product_id, product_name").execute())}
    feats = {}
    for f in rows(pharmacy("inventory_ai_features").select(
            "product_id, location_id, current_stock, days_of_stock, forecast_days_of_stock, "
            "stockout_risk, expiry_risk, days_to_expiry, demand_pattern").execute()):
        feats[(f["product_id"], f["location_id"])] = f

    for f in forecasts:
        feat = feats.get((f["product_id"], f["location_id"]), {})
        f["product_name"] = names.get(f["product_id"], "Unknown product")
        f["current_stock"] = feat.get("current_stock", 0)
        f["days_of_stock"] = feat.get("days_of_stock")
        f["stockout_risk"] = feat.get("stockout_risk", "LOW")
        f["expiry_risk"] = feat.get("expiry_risk", "LOW")
        f["days_to_expiry"] = feat.get("days_to_expiry")
    return ok(forecasts)


@router.get("/summary/")
def forecast_summary(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(pharmacy("v_forecast_summary").select("*").execute()))


@router.post("/run")
def run_forecast(horizon: int = Query(30, ge=7, le=90),
                 history_days: int = Query(180, ge=60, le=365),
                 user: CurrentUser = Depends(get_current_user)):
    """Recompute every forecast from live dispensing history."""
    result = forecasting.run_forecast(horizon=horizon, history_days=history_days)
    return {"success": True, "message": "Forecast regenerated", "data": result}


@router.get("/backtest")
def backtest(sample: int = Query(40, ge=5, le=200), holdout: int = Query(14, ge=7, le=30),
             user: CurrentUser = Depends(get_current_user)):
    """Hold-out accuracy score — evidence the model works."""
    return {"success": True, "data": forecasting.backtest(sample=sample, holdout=holdout)}


@router.get("/recommendations/")
def recommendations(user: CurrentUser = Depends(get_current_user)):
    data = rows(
        pharmacy("replenishment_recommendations").select("*")
        .order("recommendation_score", desc=True).limit(100).execute()
    )
    names = {p["product_id"]: p["product_name"]
             for p in rows(pharmacy("products").select("product_id, product_name").execute())}
    sups = {s["supplier_id"]: s["supplier_name"]
            for s in rows(pharmacy("suppliers").select("supplier_id, supplier_name").execute())}
    for r in data:
        r["product_name"] = names.get(r["product_id"])
        r["supplier_name"] = sups.get(r.get("supplier_id"))
    return ok(data)
