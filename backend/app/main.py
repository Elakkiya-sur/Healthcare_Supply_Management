from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routers import (
    admin,
    auth,
    catalog,
    chat,
    equipment,
    forecast,
    insights,
    inventory,
    procurement,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description=(
        "Cross-domain pharmacy intelligence: medicines, equipment and "
        "suppliers in one model, with GPT-4.1 reasoning on top of "
        "statistical forecasting."
    ),
)


# ============================================================
# CORS Configuration
# ============================================================

# settings.CORS_ORIGINS comes from the CORS_ORIGINS environment variable.
#
# Example:
# CORS_ORIGINS=https://your-app.web.app,https://your-app.firebaseapp.com,http://localhost:5173
#
# Do NOT use "*" together with allow_credentials=True.

cors_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS
    if origin.strip() and origin.strip() != "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routers
# ============================================================

app.include_router(auth.router)
app.include_router(catalog.router)
app.include_router(inventory.router)
app.include_router(forecast.router)
app.include_router(procurement.router)
app.include_router(equipment.router)
app.include_router(insights.router)
app.include_router(chat.router)
app.include_router(admin.router)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/", tags=["Meta"])
def root():
    try:
        from app.services.ai_engine import ai_available

        ai_mode = "gpt-4.1" if ai_available() else "rule-engine"

    except Exception:
        ai_mode = "unavailable"

    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "organization": settings.ORGANIZATION_NAME,
        "currency": settings.CURRENCY,
        "ai_mode": ai_mode,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health", tags=["Meta"])
def health():
    """
    Health endpoint for Render and general monitoring.

    This currently checks the Supabase connection because your
    existing project code uses app.db.supabase.
    """

    try:
        from app.db.supabase import pharmacy, rows

        rows(
            pharmacy("organizations")
            .select("organization_id")
            .limit(1)
            .execute()
        )

        return {
            "status": "ok",
            "database": "connected",
        }

    except Exception as exc:
        return {
            "status": "ok",
            "database": "error",
            "detail": str(exc),
        }


# ============================================================
# Global Exception Handler
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """
    Catch unexpected application errors and return a JSON response.
    """

    print(
        f"[unhandled] {request.method} "
        f"{request.url.path}: {exc}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error",
        },
    )


# ============================================================
# Local Development Entry Point
# ============================================================

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
