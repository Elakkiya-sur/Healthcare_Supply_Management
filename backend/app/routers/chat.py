"""
MediFlow AI Copilot.

Flow:
  1. User message -> gpt-4.1 with the MediFlow tool schema.
  2. Read tools execute immediately and feed results back to the model.
  3. Draft tools return a DRAFT CARD. Nothing is written.
  4. The UI renders the card; the user clicks Confirm, which calls /confirm
     and commits the exact payload that was shown on screen.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.db.supabase import pharmacy, rows
from app.services import ai_engine, tools

router = APIRouter(prefix="/api/chat", tags=["AI Copilot"])

SESSIONS: Dict[str, Dict[str, Any]] = {}
MAX_TURNS = 20

SYSTEM_PROMPT = f"""You are the MediFlow Copilot for {settings.ORGANIZATION_NAME}, a hospital
pharmacy network. Currency is {settings.CURRENCY}. Today is {{today}}.

You have live access to three connected domains:
  MEDICINES  — stock, batches, expiry, 30-day demand forecasts
  EQUIPMENT  — refrigerators, freezers, analysers: status, calibration, downtime
  SUPPLIERS  — on-time delivery, fill rate, lead times, compliance certificates

HOW TO WORK
- Always call a tool before stating any figure. Never estimate from memory.
- Prefer cross-domain reasoning: if a reorder is discussed, check the supplier's
  lead time and on-time rate; if a cold-chain product is discussed, check the
  refrigeration equipment at that location.
- When the user asks to create a purchase order, goods receipt or transfer, call
  the matching draft_* tool. These DRAFT only — the user must confirm on screen.
  After drafting, reply with one short sentence telling them to review and confirm.
  Do not restate every field; the card already shows them.
- If a draft returns warnings, mention the most important one in your reply.
- If an entity cannot be resolved, say so and offer the closest matches.

STYLE
- Direct and operational. Lead with the answer.
- Short paragraphs or tight bullets. No preamble, no filler.
- Name specific products, locations, suppliers and assets.
- Under 130 words unless the user asks for depth."""

QUICK_ACTIONS = [
    {"label": "Critical stock risks", "query": "Which items are at high stockout risk right now?"},
    {"label": "Expiring soon", "query": "What is expiring in the next 30 days and can any of it be transferred?"},
    {"label": "Supplier reliability", "query": "Which suppliers are underperforming on delivery?"},
    {"label": "Cold chain check", "query": "Are any cold-chain medicines exposed by equipment problems?"},
    {"label": "Draft a PO", "query": "Create a purchase order for 500 units of Amoxicillin 500 mg Capsule"},
    {"label": "What should I do today?", "query": "Give me my top three priorities today"},
]


# --------------------------------------------------------------- models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    scope: Optional[str] = "dashboard"


class ConfirmRequest(BaseModel):
    draft_type: str
    payload: Dict[str, Any]
    session_id: Optional[str] = None


# --------------------------------------------------------------- sessions
@router.post("/session")
def create_session(user: CurrentUser = Depends(get_current_user)):
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    SESSIONS[session_id] = {
        "user_id": user.user_id,
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
    }
    return {
        "success": True,
        "session_id": session_id,
        "quick_actions": QUICK_ACTIONS,
        "ai_mode": "gpt-4.1" if ai_engine.ai_available() else "rule-engine",
    }


@router.get("/session/{session_id}")
def get_session(session_id: str, user: CurrentUser = Depends(get_current_user)):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"success": True, "session_id": session_id, "messages": session["messages"]}


@router.delete("/session/{session_id}")
def clear_session(session_id: str, user: CurrentUser = Depends(get_current_user)):
    SESSIONS.pop(session_id, None)
    return {"success": True, "message": "Session cleared"}


# --------------------------------------------------------------- fallback
def _fallback(message: str) -> dict:
    """Keyword routing so the copilot still works without Azure configured."""
    text = message.lower()
    if any(k in text for k in ("expire", "expiry", "expiring")):
        result = tools.query_expiry(limit=8)
        body = (f"There are {result['count']} open expiry alerts. "
                f"The nearest batch expires in "
                f"{result['rows'][0]['days_to_expiry'] if result['rows'] else 'n/a'} days.")
    elif any(k in text for k in ("supplier", "vendor", "delivery")):
        result = tools.check_supplier(limit=6)
        body = f"Retrieved performance for {result['count']} suppliers, weakest on-time first."
    elif any(k in text for k in ("equipment", "fridge", "refrigerat", "calibration")):
        result = tools.check_equipment(limit=8)
        body = f"{result['count']} assets are currently at elevated readiness risk."
    elif any(k in text for k in ("forecast", "predict", "demand")):
        result = tools.query_forecast(limit=8)
        body = f"Showing the {result['count']} products with the largest forecast-to-stock gap."
    elif any(k in text for k in ("transfer", "rebalance")):
        result = tools.transfer_opportunities(limit=8)
        body = f"{result['count']} transfer opportunities could avoid write-offs."
    else:
        result = tools.query_inventory(risk="HIGH", limit=8)
        body = f"{result['count']} inventory positions are at high stockout risk."

    return {
        "success": True,
        "response": body + "\n\n(Azure OpenAI is not configured, so this is the rule-based "
                           "fallback. Add your gpt-4.1 credentials to backend/.env for full "
                           "conversational reasoning.)",
        "data": result.get("rows", [])[:5],
        "draft": None,
        "suggestions": [a["query"] for a in QUICK_ACTIONS[:4]],
        "query_type": result.get("kind", "general"),
        "tools_used": [],
        "ai_mode": "rule-engine",
    }


# --------------------------------------------------------------- main
@router.post("/")
def chat(body: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    session = SESSIONS.get(body.session_id or "")
    if session is None:
        body.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        session = SESSIONS[body.session_id] = {
            "user_id": user.user_id, "messages": [],
            "created_at": datetime.utcnow().isoformat(),
        }

    if not ai_engine.ai_available():
        result = _fallback(body.message)
        session["messages"].append({"role": "user", "content": body.message})
        session["messages"].append({"role": "assistant", "content": result["response"]})
        result["session_id"] = body.session_id
        return result

    system = SYSTEM_PROMPT.format(today=datetime.utcnow().date().isoformat())
    convo: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    convo += session["messages"][-MAX_TURNS:]
    convo.append({"role": "user", "content": body.message})

    draft: Optional[dict] = None
    payload_rows: List[dict] = []
    tools_used: List[str] = []

    try:
        for _hop in range(4):
            response = ai_engine.complete(
                convo, temperature=0.25, max_tokens=800, tools=tools.TOOL_SCHEMA
            )
            if response is None:
                raise RuntimeError("Azure OpenAI returned no response")

            choice = response.choices[0].message
            if not getattr(choice, "tool_calls", None):
                answer = (choice.content or "").strip()
                break

            convo.append({
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in choice.tool_calls
                ],
            })

            for call in choice.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tools_used.append(name)
                executor = tools.EXECUTORS.get(name)
                if executor is None:
                    output = {"ok": False, "error": f"Unknown tool '{name}'"}
                else:
                    capability = tools.CAPABILITY_FOR_TOOL.get(name)
                    if capability and not user.can(capability):
                        output = {"ok": False,
                                  "error": f"Your role ({user.role}) is not permitted to "
                                           f"create this document."}
                    else:
                        try:
                            output = executor(**args)
                        except TypeError as exc:
                            output = {"ok": False, "error": f"Invalid arguments: {exc}"}
                        except Exception as exc:  # pragma: no cover
                            output = {"ok": False, "error": str(exc)}

                if name in tools.DRAFT_TOOLS and output.get("ok"):
                    draft = output
                    pharmacy("ai_action_log").insert({
                        "organization_id": settings.ORGANIZATION_ID,
                        "user_id": user.user_id,
                        "session_id": body.session_id,
                        "user_message": body.message,
                        "tool_name": name,
                        "tool_arguments": args,
                        "outcome": "drafted",
                    }).execute()
                elif isinstance(output.get("rows"), list):
                    payload_rows = output["rows"][:8]

                convo.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(output, default=str)[:6000],
                })
        else:
            answer = "I gathered the data but could not finish composing a reply. Please rephrase."

    except Exception as exc:
        print(f"[chat] error: {exc}")
        result = _fallback(body.message)
        result["session_id"] = body.session_id
        return result

    session["messages"].append({"role": "user", "content": body.message})
    session["messages"].append({"role": "assistant", "content": answer})

    return {
        "success": True,
        "session_id": body.session_id,
        "response": answer,
        "data": payload_rows,
        "draft": draft,
        "suggestions": [] if draft else [a["query"] for a in QUICK_ACTIONS[:3]],
        "query_type": "draft" if draft else ("data" if payload_rows else "general"),
        "tools_used": tools_used,
        "ai_mode": "gpt-4.1",
    }


@router.post("/confirm")
def confirm(body: ConfirmRequest, user: CurrentUser = Depends(get_current_user)):
    """Commit a draft the user explicitly approved on screen."""
    capability = {"purchase_order": "can_create_po"}.get(body.draft_type)
    if capability and not user.can(capability):
        raise HTTPException(403, f"Role '{user.role}' cannot create this document.")

    result = tools.commit_draft(body.draft_type, body.payload, user_id=user.user_id)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Could not create the record"))

    pharmacy("ai_action_log").insert({
        "organization_id": settings.ORGANIZATION_ID,
        "user_id": user.user_id,
        "session_id": body.session_id,
        "tool_name": f"commit_{body.draft_type}",
        "tool_arguments": body.payload,
        "outcome": "confirmed",
        "result_ref_id": result.get("record_id"),
    }).execute()

    return {
        "success": True,
        "message": f"{body.draft_type.replace('_', ' ').title()} "
                   f"{result.get('reference') or ''} created",
        "reference": result.get("reference"),
        "record_id": result.get("record_id"),
        "data": result.get("record"),
    }


@router.post("/reject")
def reject(body: ConfirmRequest, user: CurrentUser = Depends(get_current_user)):
    pharmacy("ai_action_log").insert({
        "organization_id": settings.ORGANIZATION_ID,
        "user_id": user.user_id,
        "session_id": body.session_id,
        "tool_name": f"reject_{body.draft_type}",
        "tool_arguments": body.payload,
        "outcome": "rejected",
    }).execute()
    return {"success": True, "message": "Draft discarded"}


@router.get("/quick-actions")
def quick_actions(user: CurrentUser = Depends(get_current_user)):
    return {"success": True, "data": QUICK_ACTIONS}
