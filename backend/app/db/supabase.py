"""Supabase client helpers, scoped to the two MediFlow schemas."""

import httpx
from supabase import Client, ClientOptions, create_client

from app.core.config import settings

if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set in backend/.env — "
        "copy .env.example first."
    )


# Use HTTP/1.1 instead of HTTP/2.
# This avoids the Windows socket error seen in concurrent requests.
http_client = httpx.Client(
    http2=False,
    timeout=httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=30.0,
        pool=10.0,
    ),
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
    ),
    follow_redirects=True,
)

options = ClientOptions(
    httpx_client=http_client,
    postgrest_client_timeout=30,
)

supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY,
    options=options,
)


def pharmacy(table: str):
    """Query builder against the pharmacy_inventory schema."""
    return supabase.schema(settings.PHARMACY_SCHEMA).from_(table)


def equipment(table: str):
    """Query builder against the equipmentsdeets schema."""
    return supabase.schema(settings.EQUIPMENT_SCHEMA).from_(table)


def rows(response) -> list:
    return response.data or []


def ok(data, message: str = None, extra: dict = None) -> dict:
    payload = {
        "success": True,
        "count": len(data) if isinstance(data, list) else 1,
        "data": data,
    }

    if message:
        payload["message"] = message

    if extra:
        payload.update(extra)

    return payload