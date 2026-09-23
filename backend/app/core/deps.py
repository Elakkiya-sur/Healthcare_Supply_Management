"""Authentication + role-based access dependencies."""
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings
from app.core.security import decode_access_token
from app.db.supabase import pharmacy, rows

_ROLE_CACHE: dict = {}


def _load_roles() -> dict:
    if not _ROLE_CACHE:
        for r in rows(pharmacy("roles").select("*").execute()):
            _ROLE_CACHE[r["role_name"]] = r
    return _ROLE_CACHE


class CurrentUser(dict):
    """Thin dict wrapper with convenience accessors."""

    @property
    def role(self) -> str:
        return self.get("role_name", "")

    @property
    def user_id(self) -> str:
        return self.get("user_id", "")

    def allows(self, page: str) -> bool:
        role = _load_roles().get(self.role, {})
        return page in (role.get("allowed_pages") or [])

    def can(self, capability: str) -> bool:
        role = _load_roles().get(self.role, {})
        return bool(role.get(capability, False))


async def get_current_user(authorization: Optional[str] = Header(None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")

    claims = decode_access_token(authorization.split(" ", 1)[1].strip())
    if not claims:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    result = rows(
        pharmacy("user_profiles")
        .select("user_id, organization_id, location_id, email, first_name, last_name, "
                "job_title, role_name, avatar_initials, status")
        .eq("user_id", claims.get("sub"))
        .limit(1)
        .execute()
    )
    if not result:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")

    user = result[0]
    if user.get("status") != "active":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "User account is not active")

    return CurrentUser(user)


def require_capability(capability: str):
    """Guard an endpoint behind a role capability (can_approve, can_create_po, ...)."""

    async def _guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not user.can(capability):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user.role}' is not permitted to perform this action.",
            )
        return user

    return _guard


def require_page(page: str):
    """Guard an endpoint behind page-level access."""

    async def _guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not user.allows(page):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user.role}' does not have access to '{page}'.",
            )
        return user

    return _guard


def org_id() -> str:
    return settings.ORGANIZATION_ID
