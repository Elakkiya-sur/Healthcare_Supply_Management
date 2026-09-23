"""Login, session and role metadata."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.core.security import create_access_token, verify_password
from app.db.supabase import ok, pharmacy, rows

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _role(role_name: str) -> dict:
    data = rows(pharmacy("roles").select("*").eq("role_name", role_name).limit(1).execute())
    return data[0] if data else {}


def _profile(user: dict) -> dict:
    role = _role(user["role_name"])
    location = None
    if user.get("location_id"):
        loc = rows(
            pharmacy("locations").select("location_id, location_name, location_code")
            .eq("location_id", user["location_id"]).limit(1).execute()
        )
        location = loc[0] if loc else None
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "full_name": f"{user.get('first_name','')} {user.get('last_name','')}".strip(),
        "job_title": user.get("job_title"),
        "initials": user.get("avatar_initials"),
        "role": user["role_name"],
        "role_display": role.get("display_name"),
        "role_description": role.get("description"),
        "allowed_pages": role.get("allowed_pages", []),
        "capabilities": {
            "can_approve": role.get("can_approve", False),
            "can_create_po": role.get("can_create_po", False),
            "can_dispense": role.get("can_dispense", False),
            "can_admin": role.get("can_admin", False),
        },
        "location": location,
        "organization": {
            "id": settings.ORGANIZATION_ID,
            "name": settings.ORGANIZATION_NAME,
            "currency": settings.CURRENCY,
        },
    }


@router.post("/login")
def login(body: LoginRequest):
    data = rows(
        pharmacy("user_profiles").select("*").eq("email", body.email.lower()).limit(1).execute()
    )
    if not data or not verify_password(body.password, data[0]["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    user = data[0]
    if user.get("status") != "active":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account is not active")

    pharmacy("user_profiles").update(
        {"last_login_at": datetime.utcnow().isoformat()}
    ).eq("user_id", user["user_id"]).execute()

    token = create_access_token({"sub": user["user_id"], "role": user["role_name"]})
    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "user": _profile(user),
    }


@router.get("/me")
def me(user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("user_profiles").select("*").eq("user_id", user.user_id).limit(1).execute())
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return {"success": True, "user": _profile(data[0])}


@router.get("/roles")
def list_roles(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(pharmacy("roles").select("*").execute()))
