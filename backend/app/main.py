"""MediFlow Intelligence API — application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routers import (admin, auth, catalog, chat, equipment, forecast,
                         insights, inventory, procurement)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description=(
        "Cross-domain pharmacy intelligence: medicines, equipment and suppliers "
        "in one model, with gpt-4.1 reasoning on top of statistical forecasting."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, catalog.router, inventory.router, forecast.router,
          procurement.router, equipment.router, insights.router,
          chat.router, admin.router):
    app.include_router(r)


@app.get("/", tags=["Meta"])
def root():
    from app.services.ai_engine import ai_available
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "organization": settings.ORGANIZATION_NAME,
        "currency": settings.CURRENCY,
        "ai_mode": "gpt-4.1" if ai_available() else "rule-engine",
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    try:
        from app.db.supabase import pharmacy, rows
        rows(pharmacy("organizations").select("organization_id").limit(1).execute())
        db = "connected"
    except Exception as exc:
        db = f"error: {exc}"
    return {"status": "ok", "database": db}


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover
    print(f"[unhandled] {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"success": False, "detail": str(exc)})
