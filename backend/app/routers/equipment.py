"""Equipment domain — the cross-domain differentiator."""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUser, get_current_user
from app.db.supabase import equipment, ok, rows

router = APIRouter(prefix="/api/equipment", tags=["Equipment"])


@router.get("/")
def list_assets(risk: Optional[str] = None, location_id: Optional[str] = None,
                cold_chain: Optional[bool] = None, limit: int = Query(200, le=500),
                user: CurrentUser = Depends(get_current_user)):
    q = equipment("v_equipment_readiness").select("*").order("readiness_score")
    if risk:
        q = q.eq("readiness_risk", risk.upper())
    if location_id:
        q = q.eq("location_id", location_id)
    if cold_chain is not None:
        q = q.eq("is_cold_chain", cold_chain)
    return ok(rows(q.limit(limit).execute()))


@router.get("/summary/")
def summary(user: CurrentUser = Depends(get_current_user)):
    data = rows(equipment("v_equipment_readiness").select("*").execute())
    total = len(data) or 1
    return ok({
        "total_assets": len(data),
        "avg_readiness": round(sum(float(d.get("readiness_score") or 0) for d in data) / total, 1),
        "high_risk": sum(1 for d in data if d.get("readiness_risk") == "HIGH"),
        "medium_risk": sum(1 for d in data if d.get("readiness_risk") == "MEDIUM"),
        "out_of_service": sum(1 for d in data if d.get("status") == "out_of_service"),
        "in_maintenance": sum(1 for d in data if d.get("status") == "maintenance"),
        "calibration_overdue": sum(1 for d in data if d.get("calibration_status") == "overdue"),
        "calibration_due_soon": sum(1 for d in data if d.get("calibration_status") == "due_soon"),
        "cold_chain_assets": sum(1 for d in data if d.get("is_cold_chain")),
        "open_downtime": sum(int(d.get("open_downtime_events") or 0) for d in data),
    })


@router.get("/alerts/")
def alerts(limit: int = Query(100, le=300), user: CurrentUser = Depends(get_current_user)):
    return ok(rows(equipment("equipment_alerts").select("*")
                   .order("due_date").limit(limit).execute()))


@router.get("/{asset_id}")
def asset_detail(asset_id: str, user: CurrentUser = Depends(get_current_user)):
    asset = rows(equipment("v_equipment_readiness").select("*")
                 .eq("equipment_asset_id", asset_id).limit(1).execute())
    return {
        "success": True,
        "data": asset[0] if asset else None,
        "calibrations": rows(equipment("equipment_calibrations").select("*")
                             .eq("equipment_asset_id", asset_id)
                             .order("calibration_due_date", desc=True).execute()),
        "maintenance": rows(equipment("equipment_maintenance").select("*")
                            .eq("equipment_asset_id", asset_id)
                            .order("scheduled_date", desc=True).execute()),
        "downtime": rows(equipment("equipment_downtime").select("*")
                         .eq("equipment_asset_id", asset_id)
                         .order("downtime_start", desc=True).execute()),
        "warranties": rows(equipment("equipment_warranties").select("*")
                           .eq("equipment_asset_id", asset_id).execute()),
    }
