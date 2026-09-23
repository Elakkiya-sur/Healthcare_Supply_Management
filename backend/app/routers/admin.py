"""Administration — users, roles and maintenance jobs. ADMIN only."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

from app.core.config import settings
from app.core.deps import CurrentUser, require_capability
from app.core.security import hash_password
from app.db.supabase import equipment, ok, pharmacy, rows, supabase

router = APIRouter(prefix="/api/admin", tags=["Administration"])
ADMIN = require_capability("can_admin")


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role_name: str
    job_title: Optional[str] = None
    location_id: Optional[str] = None


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    job_title: Optional[str] = None
    role_name: Optional[str] = None
    location_id: Optional[str] = None
    status: Optional[str] = None
    password: Optional[str] = None


@router.get("/users")
def list_users(user: CurrentUser = Depends(ADMIN)):
    data = rows(pharmacy("user_profiles").select(
        "user_id, employee_code, email, first_name, last_name, job_title, role_name, "
        "location_id, avatar_initials, last_login_at, status, created_at"
    ).order("employee_code").execute())
    locs = {l["location_id"]: l["location_name"]
            for l in rows(pharmacy("locations").select("location_id, location_name").execute())}
    for u in data:
        u["location_name"] = locs.get(u.get("location_id"))
    return ok(data)


@router.post("/users")
def create_user(body: UserCreate, user: CurrentUser = Depends(ADMIN)):
    if rows(pharmacy("user_profiles").select("user_id").eq("email", body.email.lower()).execute()):
        raise HTTPException(409, "A user with that email already exists")

    existing = rows(pharmacy("user_profiles").select("employee_code").execute())
    next_no = max([int(e["employee_code"].split("-")[-1]) for e in existing] or [0]) + 1

    import uuid as _uuid
    payload = {
        "user_id": str(_uuid.uuid4()),
        "organization_id": settings.ORGANIZATION_ID,
        "location_id": body.location_id,
        "employee_code": f"EMP-{next_no:04d}",
        "email": body.email.lower(),
        "password_hash": hash_password(body.password),
        "first_name": body.first_name,
        "last_name": body.last_name,
        "job_title": body.job_title,
        "role_name": body.role_name,
        "avatar_initials": (body.first_name[:1] + body.last_name[:1]).upper(),
        "status": "active",
    }
    created = rows(pharmacy("user_profiles").insert(payload).execute())
    for c in created:
        c.pop("password_hash", None)
    return ok(created, "User created")


@router.put("/users/{user_id}")
def update_user(user_id: str, body: UserUpdate, user: CurrentUser = Depends(ADMIN)):
    payload = body.model_dump(exclude_unset=True)
    if payload.pop("password", None):
        payload["password_hash"] = hash_password(body.password)
    if not payload:
        return ok([], "No changes to apply")
    data = rows(pharmacy("user_profiles").update(payload).eq("user_id", user_id).execute())
    for d in data:
        d.pop("password_hash", None)
    return ok(data, "User updated")


@router.delete("/users/{user_id}")
def deactivate_user(user_id: str, user: CurrentUser = Depends(ADMIN)):
    if user_id == user.user_id:
        raise HTTPException(400, "You cannot deactivate your own account")
    data = rows(pharmacy("user_profiles").update({"status": "inactive"})
                .eq("user_id", user_id).execute())
    return ok(data, "User deactivated")


@router.post("/jobs/refresh-expiry-alerts")
def refresh_expiry(days: int = 90, user: CurrentUser = Depends(ADMIN)):
    result = supabase.rpc("fn_refresh_expiry_alerts", {"p_days": days}).execute()
    return {"success": True, "message": "Expiry alerts regenerated", "data": result.data}


@router.post("/jobs/refresh-compliance")
def refresh_compliance(user: CurrentUser = Depends(ADMIN)):
    result = supabase.rpc("fn_refresh_compliance_status", {}).execute()
    return {"success": True, "message": "Compliance and calibration statuses refreshed",
            "data": result.data}


@router.get("/health")
def system_health(user: CurrentUser = Depends(ADMIN)):
    from app.services.ai_engine import ai_available
    counts = {}
    for label, table in [("products", "products"), ("inventory", "inventory"),
                         ("transactions", "inventory_transactions"),
                         ("forecasts", "demand_forecasts"),
                         ("purchase_orders", "purchase_orders"),
                         ("expiry_alerts", "expiry_alerts")]:
        try:
            counts[label] = len(rows(pharmacy(table).select("*", count="exact")
                                     .limit(1).execute())) or 0
        except Exception:
            counts[label] = None
    return ok({
        "organization": settings.ORGANIZATION_NAME,
        "currency": settings.CURRENCY,
        "ai_mode": "gpt-4.1" if ai_available() else "rule-engine",
        "equipment_assets": len(rows(equipment("equipment_assets").select("id").execute())),
        "tables": counts,
    })
