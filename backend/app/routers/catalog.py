"""Products, categories, locations, suppliers, batches."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.deps import CurrentUser, get_current_user, require_capability
from app.db.supabase import ok, pharmacy, rows
from app.services.ai_engine import supplier_verdict

router = APIRouter(tags=["Catalog"])


@router.get("/api/products/")
def get_products(search: Optional[str] = None, category_id: Optional[str] = None,
                 status: Optional[str] = None, limit: int = Query(500, le=1000),
                 user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("products").select("*").order("product_name")
    if search:
        q = q.ilike("product_name", f"%{search}%")
    if category_id:
        q = q.eq("category_id", category_id)
    if status:
        q = q.eq("status", status)
    return ok(rows(q.limit(limit).execute()))


@router.get("/api/products/{product_id}")
def get_product(product_id: str, user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("products").select("*").eq("product_id", product_id).limit(1).execute())
    if not data:
        raise HTTPException(404, "Product not found")
    return ok(data[0])


@router.post("/api/products/")
def create_product(body: dict, user: CurrentUser = Depends(require_capability("can_admin"))):
    return ok(rows(pharmacy("products").insert(body).execute()), "Product created")


@router.put("/api/products/{product_id}")
def update_product(product_id: str, body: dict,
                   user: CurrentUser = Depends(require_capability("can_admin"))):
    data = rows(pharmacy("products").update(body).eq("product_id", product_id).execute())
    return ok(data, "Product updated")


@router.get("/api/categories/")
def get_categories(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(pharmacy("product_categories").select("*").order("category_name").execute()))


@router.get("/api/locations/")
def get_locations(location_type: Optional[str] = None, status: Optional[str] = None,
                  user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("locations").select("*").order("location_code")
    if location_type:
        q = q.eq("location_type", location_type)
    if status:
        q = q.eq("status", status)
    return ok(rows(q.execute()))


@router.get("/api/suppliers/")
def get_suppliers(status: Optional[str] = None, user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("suppliers").select("*").order("supplier_name")
    if status:
        q = q.eq("status", status)
    return ok(rows(q.execute()))


@router.get("/api/suppliers/360")
def suppliers_360(user: CurrentUser = Depends(get_current_user)):
    return ok(rows(pharmacy("v_supplier_360").select("*").order("supplier_name").execute()))


@router.get("/api/suppliers/360/{supplier_id}")
def supplier_detail(supplier_id: str, user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("v_supplier_360").select("*").eq("supplier_id", supplier_id).limit(1).execute())
    if not data:
        raise HTTPException(404, "Supplier not found")
    supplier = data[0]
    from app.db.supabase import equipment
    history = rows(
        equipment("supplier_performance").select("*")
        .eq("supplier_id", supplier_id).order("measurement_date").execute()
    )
    compliance = rows(
        equipment("supplier_compliance").select("*")
        .eq("supplier_id", supplier_id).order("expiry_date").execute()
    )
    return {
        "success": True,
        "data": supplier,
        "performance_history": history,
        "compliance": compliance,
        "ai": supplier_verdict(supplier),
    }


@router.get("/api/batches/")
def get_batches(product_id: Optional[str] = None, expiring_days: Optional[int] = None,
                limit: int = Query(300, le=1000), user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("batches").select("*").order("expiry_date")
    if product_id:
        q = q.eq("product_id", product_id)
    if expiring_days:
        from datetime import date, timedelta
        q = q.lte("expiry_date", (date.today() + timedelta(days=expiring_days)).isoformat())
    return ok(rows(q.limit(limit).execute()))
