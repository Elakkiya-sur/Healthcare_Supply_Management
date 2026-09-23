"""Purchase orders, goods receipts, transfers, dispensing, expiry, audit."""
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user, require_capability
from app.db.supabase import ok, pharmacy, rows

router = APIRouter(tags=["Procurement"])
ORG = settings.ORGANIZATION_ID


# ---------------------------------------------------------------- models
class POCreate(BaseModel):
    supplier_id: str
    location_id: str
    product_id: str
    quantity: float
    order_date: Optional[date] = None
    expected_delivery_date: Optional[date] = None
    notes: Optional[str] = None


class POUpdate(BaseModel):
    supplier_id: Optional[str] = None
    location_id: Optional[str] = None
    product_id: Optional[str] = None
    quantity: Optional[float] = None
    order_date: Optional[date] = None
    expected_delivery_date: Optional[date] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class GRCreate(BaseModel):
    purchase_order_id: Optional[str] = None
    supplier_id: str
    location_id: str
    receipt_date: Optional[date] = None
    notes: Optional[str] = None


class TransferCreate(BaseModel):
    source_location_id: str
    destination_location_id: str
    product_id: Optional[str] = None
    quantity: Optional[float] = None
    transfer_date: Optional[date] = None
    expected_arrival_date: Optional[date] = None
    reason: Optional[str] = "routine"
    notes: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str


class DispensingLine(BaseModel):
    product_id: str
    quantity: float
    batch_id: Optional[str] = None       # resolved by FEFO when omitted
    unit_price: Optional[float] = None


class DispensingCreate(BaseModel):
    location_id: str
    lines: List[DispensingLine] = []
    prescription_reference: Optional[str] = None
    patient_reference: Optional[str] = None
    prescriber_reference: Optional[str] = None
    notes: Optional[str] = None


class DispensingUpdate(BaseModel):
    dispensing_status: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------- purchase orders
@router.get("/api/purchase-orders/")
def list_pos(status: Optional[str] = None, limit: int = Query(300, le=1000),
             user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("purchase_orders").select("*").order("order_date", desc=True)
    if status:
        q = q.eq("status", status.lower())
    return ok(rows(q.limit(limit).execute()))


@router.post("/api/purchase-orders/")
def create_po(body: POCreate, user: CurrentUser = Depends(require_capability("can_create_po"))):
    for kind, table, pk, value in [
        ("Supplier", "suppliers", "supplier_id", body.supplier_id),
        ("Product", "products", "product_id", body.product_id),
        ("Location", "locations", "location_id", body.location_id),
    ]:
        if not rows(pharmacy(table).select(pk).eq(pk, value).limit(1).execute()):
            raise HTTPException(404, f"{kind} not found")
    if body.quantity <= 0:
        raise HTTPException(400, "Quantity must be greater than zero")

    payload = {
        "organization_id": ORG,
        "supplier_id": body.supplier_id,
        "location_id": body.location_id,
        "product_id": body.product_id,
        "quantity": body.quantity,
        "order_date": (body.order_date or date.today()).isoformat(),
        "expected_delivery_date": body.expected_delivery_date.isoformat()
        if body.expected_delivery_date else None,
        "currency_code": settings.CURRENCY,
        "status": "draft",
        "created_by": user.user_id,
        "source": "manual",
        "notes": body.notes,
    }
    return ok(rows(pharmacy("purchase_orders").insert(payload).execute()), "Purchase order created")


@router.put("/api/purchase-orders/{purchase_order_id}")
def update_po(purchase_order_id: str, body: POUpdate,
              user: CurrentUser = Depends(get_current_user)):
    payload = {k: (v.isoformat() if isinstance(v, date) else v)
               for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        return ok([], "No changes to apply")
    if payload.get("status") in ("approved", "received") and not user.can("can_approve"):
        raise HTTPException(403, f"Role '{user.role}' cannot approve purchase orders")
    if "status" in payload:
        payload["status"] = payload["status"].lower()
    data = rows(pharmacy("purchase_orders").update(payload)
                .eq("purchase_order_id", purchase_order_id).execute())
    return ok(data, "Purchase order updated")


# ---------------------------------------------------------------- goods receipts
@router.get("/api/goods-receipts/")
def list_grs(status: Optional[str] = None, limit: int = Query(300, le=1000),
             user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("goods_receipts").select("*").order("receipt_date", desc=True)
    if status:
        q = q.eq("status", status.lower())
    return ok(rows(q.limit(limit).execute()))


@router.post("/api/goods-receipts/")
def create_gr(body: GRCreate, user: CurrentUser = Depends(get_current_user)):
    payload = {
        "organization_id": ORG,
        "purchase_order_id": body.purchase_order_id,
        "supplier_id": body.supplier_id,
        "location_id": body.location_id,
        "receipt_date": (body.receipt_date or date.today()).isoformat(),
        "received_by": user.user_id,
        "status": "pending",
        "source": "manual",
        "notes": body.notes,
    }
    return ok(rows(pharmacy("goods_receipts").insert(payload).execute()), "Goods receipt created")


@router.put("/api/goods-receipts/{goods_receipt_id}")
def update_gr(goods_receipt_id: str, body: StatusUpdate,
              user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("goods_receipts").update({"status": body.status.lower()})
                .eq("goods_receipt_id", goods_receipt_id).execute())
    return ok(data, f"Goods receipt marked {body.status.lower()}")


# ---------------------------------------------------------------- transfers
@router.get("/api/transfers/")
def list_transfers(status: Optional[str] = None, limit: int = Query(300, le=1000),
                   user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("inventory_transfers").select("*").order("transfer_date", desc=True)
    if status:
        q = q.eq("status", status.lower())
    return ok(rows(q.limit(limit).execute()))


@router.post("/api/transfers/")
def create_transfer(body: TransferCreate, user: CurrentUser = Depends(get_current_user)):
    if body.source_location_id == body.destination_location_id:
        raise HTTPException(400, "Source and destination must differ")
    payload = {
        "organization_id": ORG,
        "source_location_id": body.source_location_id,
        "destination_location_id": body.destination_location_id,
        "product_id": body.product_id,
        "quantity": body.quantity,
        "transfer_date": (body.transfer_date or date.today()).isoformat(),
        "expected_arrival_date": body.expected_arrival_date.isoformat()
        if body.expected_arrival_date else None,
        "status": "pending",
        "reason": body.reason or "routine",
        "source": "manual",
        "notes": body.notes,
    }
    return ok(rows(pharmacy("inventory_transfers").insert(payload).execute()), "Transfer created")


@router.put("/api/transfers/{transfer_id}")
def update_transfer(transfer_id: str, body: StatusUpdate,
                    user: CurrentUser = Depends(get_current_user)):
    data = rows(pharmacy("inventory_transfers").update({"status": body.status.lower()})
                .eq("transfer_id", transfer_id).execute())
    return ok(data, f"Transfer marked {body.status.lower()}")


# ================================================================
# DISPENSING
# Header rows live in dispensing_transactions; the products dispensed
# live in dispensing_transaction_lines. The list endpoint therefore
# joins the lines back on, so the UI can show what was actually given
# to the patient rather than just a reference number.
# ================================================================
def _product_lookup() -> Dict[str, dict]:
    return {
        p["product_id"]: p
        for p in rows(pharmacy("products")
                      .select("product_id, product_name, product_code, strength, "
                              "dosage_form, unit_of_measure")
                      .execute())
    }


def _batch_lookup(batch_ids: List[str]) -> Dict[str, dict]:
    if not batch_ids:
        return {}
    return {
        b["batch_id"]: b
        for b in rows(pharmacy("batches")
                      .select("batch_id, batch_number, expiry_date")
                      .in_("batch_id", batch_ids).execute())
    }


def _attach_lines(txns: List[dict]) -> List[dict]:
    """Attach product lines + a readable summary to each dispensing header."""
    if not txns:
        return txns

    ids = [t["dispensing_transaction_id"] for t in txns]
    lines: List[dict] = []
    for i in range(0, len(ids), 100):                    # keep the IN list sane
        lines += rows(pharmacy("dispensing_transaction_lines")
                      .select("*").in_("dispensing_transaction_id", ids[i:i + 100])
                      .execute())

    products = _product_lookup()
    batches = _batch_lookup(list({l["batch_id"] for l in lines if l.get("batch_id")}))

    by_txn: Dict[str, List[dict]] = {}
    for l in lines:
        prod = products.get(l.get("product_id"), {})
        batch = batches.get(l.get("batch_id"), {})
        l["product_name"] = prod.get("product_name", "Unknown product")
        l["product_code"] = prod.get("product_code")
        l["strength"] = prod.get("strength")
        l["dosage_form"] = prod.get("dosage_form")
        l["unit_of_measure"] = prod.get("unit_of_measure")
        l["batch_number"] = batch.get("batch_number")
        l["expiry_date"] = batch.get("expiry_date")
        by_txn.setdefault(l["dispensing_transaction_id"], []).append(l)

    for t in txns:
        items = by_txn.get(t["dispensing_transaction_id"], [])
        t["items"] = items
        t["item_count"] = len(items)
        t["total_quantity"] = sum(float(i.get("quantity") or 0) for i in items)
        if items:
            first = items[0]["product_name"]
            extra = len(items) - 1
            t["primary_product"] = first
            t["product_summary"] = f"{first} +{extra} more" if extra else first
        else:
            t["primary_product"] = None
            t["product_summary"] = "No items recorded"
    return txns


def _fefo_batch(product_id: str, location_id: str) -> Optional[dict]:
    """First-expiry-first-out: the correct batch to dispense from."""
    found = rows(pharmacy("inventory_ai_features")
                 .select("batch_id, batch_number, expiry_date, current_stock, unit_cost")
                 .eq("product_id", product_id).eq("location_id", location_id)
                 .gt("current_stock", 0).order("expiry_date").limit(1).execute())
    if found:
        return found[0]
    # fall back to any batch of this product so the record can still be written
    any_batch = rows(pharmacy("batches").select("batch_id, batch_number, expiry_date")
                     .eq("product_id", product_id).order("expiry_date").limit(1).execute())
    return any_batch[0] if any_batch else None


@router.get("/api/dispensing/")
def list_dispensing(status: Optional[str] = None, location_id: Optional[str] = None,
                    product_id: Optional[str] = None, limit: int = Query(200, le=500),
                    user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("dispensing_transactions").select("*").order("dispensing_date", desc=True)
    if status:
        q = q.eq("dispensing_status", status.lower())
    if location_id:
        q = q.eq("location_id", location_id)

    data = _attach_lines(rows(q.limit(limit).execute()))

    if product_id:
        data = [t for t in data
                if any(i.get("product_id") == product_id for i in t.get("items", []))]
    return ok(data)


@router.get("/api/dispensing/{dispensing_transaction_id}")
def get_dispensing(dispensing_transaction_id: str,
                   user: CurrentUser = Depends(get_current_user)):
    found = rows(pharmacy("dispensing_transactions").select("*")
                 .eq("dispensing_transaction_id", dispensing_transaction_id)
                 .limit(1).execute())
    if not found:
        raise HTTPException(404, "Dispensing transaction not found")

    txn = _attach_lines(found)[0]
    loc = rows(pharmacy("locations").select("location_name")
               .eq("location_id", txn["location_id"]).limit(1).execute())
    txn["location_name"] = loc[0]["location_name"] if loc else None
    return ok(txn)


@router.post("/api/dispensing/")
def create_dispensing(body: DispensingCreate,
                      user: CurrentUser = Depends(require_capability("can_dispense"))):
    if not rows(pharmacy("locations").select("location_id")
                .eq("location_id", body.location_id).limit(1).execute()):
        raise HTTPException(404, "Location not found")
    if not body.lines:
        raise HTTPException(400, "At least one product line is required")

    products = _product_lookup()
    prepared: List[dict] = []
    total = 0.0

    for line in body.lines:
        if line.product_id not in products:
            raise HTTPException(404, f"Product {line.product_id} not found")
        if line.quantity <= 0:
            raise HTTPException(400, "Quantity must be greater than zero")

        batch_id = line.batch_id
        if not batch_id:
            batch = _fefo_batch(line.product_id, body.location_id)
            if not batch:
                raise HTTPException(
                    400,
                    f"No batch with stock available for "
                    f"{products[line.product_id]['product_name']} at this location",
                )
            batch_id = batch["batch_id"]

        unit_price = line.unit_price
        if unit_price is None:
            sp = rows(pharmacy("supplier_products").select("unit_price")
                      .eq("product_id", line.product_id)
                      .order("is_preferred", desc=True).limit(1).execute())
            unit_price = round(float(sp[0]["unit_price"]) * 1.45, 4) if sp else 0.0

        line_total = round(float(unit_price) * float(line.quantity), 2)
        total += line_total
        prepared.append({
            "dispensing_transaction_line_id": str(uuid.uuid4()),
            "product_id": line.product_id,
            "batch_id": batch_id,
            "quantity": line.quantity,
            "unit_price": unit_price,
            "discount_amount": 0,
            "tax_amount": round(line_total * 0.0825, 2),
            "line_total": line_total,
        })

    txn_id = str(uuid.uuid4())
    header = {
        "dispensing_transaction_id": txn_id,
        "organization_id": ORG,
        "location_id": body.location_id,
        "dispensing_date": datetime.utcnow().isoformat(),
        "prescription_reference": body.prescription_reference,
        "patient_reference": body.patient_reference,
        "prescriber_reference": body.prescriber_reference,
        "dispensing_status": "pending",
        "total_amount": round(total * 1.0825, 2),
        "notes": body.notes,
    }
    created = rows(pharmacy("dispensing_transactions").insert(header).execute())

    for p in prepared:
        p["dispensing_transaction_id"] = txn_id
    pharmacy("dispensing_transaction_lines").insert(prepared).execute()

    result = _attach_lines(created)
    return ok(result, f"Dispensing created with {len(prepared)} product line(s)")


@router.put("/api/dispensing/{dispensing_transaction_id}")
def update_dispensing(dispensing_transaction_id: str, body: DispensingUpdate,
                      user: CurrentUser = Depends(require_capability("can_dispense"))):
    payload: Dict[str, Any] = {}
    if body.dispensing_status:
        payload["dispensing_status"] = body.dispensing_status.lower()
    if body.notes is not None:
        payload["notes"] = body.notes
    if not payload:
        return ok([], "No changes to apply")

    existing = rows(pharmacy("dispensing_transactions").select("*")
                    .eq("dispensing_transaction_id", dispensing_transaction_id)
                    .limit(1).execute())
    if not existing:
        raise HTTPException(404, "Dispensing transaction not found")
    previous = existing[0]

    data = rows(pharmacy("dispensing_transactions").update(payload)
                .eq("dispensing_transaction_id", dispensing_transaction_id).execute())

    # Completing a dispense removes stock and writes the demand signal
    # that the forecasting service reads.
    if (payload.get("dispensing_status") == "dispensed"
            and previous.get("dispensing_status") != "dispensed"):
        lines = rows(pharmacy("dispensing_transaction_lines").select("*")
                     .eq("dispensing_transaction_id", dispensing_transaction_id).execute())
        movements = [{
            "transaction_id": str(uuid.uuid4()),
            "organization_id": ORG,
            "product_id": l["product_id"],
            "batch_id": l["batch_id"],
            "location_id": previous["location_id"],
            "transaction_type": "dispense",
            "quantity": -abs(float(l.get("quantity") or 0)),
            "unit_cost": l.get("unit_price") or 0,
            "reference_type": "dispensing",
            "reference_id": dispensing_transaction_id,
            "notes": f"Dispensed on {previous.get('dispensing_number')}",
        } for l in lines]
        if movements:
            pharmacy("inventory_transactions").insert(movements).execute()

    return ok(data, f"Dispensing marked {payload.get('dispensing_status', 'updated')}")


# ---------------------------------------------------------------- expiry + audit
@router.get("/api/expiry-alerts/")
def list_expiry(level: Optional[str] = None, limit: int = Query(300, le=1000),
                user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("expiry_alerts").select("*").order("days_to_expiry")
    if level:
        q = q.eq("alert_level", level.lower())
    alerts = rows(q.limit(limit).execute())

    names = {p["product_id"]: p["product_name"]
             for p in rows(pharmacy("products").select("product_id, product_name").execute())}
    locs = {l["location_id"]: l["location_name"]
            for l in rows(pharmacy("locations").select("location_id, location_name").execute())}
    batches = {b["batch_id"]: b["batch_number"]
               for b in rows(pharmacy("batches").select("batch_id, batch_number").execute())}
    for a in alerts:
        a["product_name"] = names.get(a["product_id"])
        a["location_name"] = locs.get(a["location_id"])
        a["batch_number"] = batches.get(a["batch_id"])
    return ok(alerts)


@router.put("/api/expiry-alerts/{expiry_alert_id}")
def update_expiry(expiry_alert_id: str, body: dict,
                  user: CurrentUser = Depends(get_current_user)):
    if "alert_status" in body:
        body["alert_status"] = body["alert_status"].lower()
    data = rows(pharmacy("expiry_alerts").update(body)
                .eq("expiry_alert_id", expiry_alert_id).execute())
    return ok(data, "Alert updated")


@router.get("/api/audit-logs/")
def list_audit(action_type: Optional[str] = None, table_name: Optional[str] = None,
               limit: int = Query(300, le=1000),
               user: CurrentUser = Depends(get_current_user)):
    q = pharmacy("audit_logs").select("*").order("action_timestamp", desc=True)
    if action_type:
        q = q.eq("action_type", action_type.lower())
    if table_name:
        q = q.eq("table_name", table_name)
    logs = rows(q.limit(limit).execute())
    users = {u["user_id"]: f"{u.get('first_name','')} {u.get('last_name','')}".strip()
             for u in rows(pharmacy("user_profiles")
                           .select("user_id, first_name, last_name").execute())}
    for entry in logs:
        entry["user_name"] = users.get(entry.get("user_id"), "System")
    return ok(logs)
