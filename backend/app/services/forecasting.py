"""
MediFlow demand forecasting service.

Pharmacy demand is dominated by intermittent and lumpy series — many products
dispense in bursts with long zero-demand gaps. Classical ARIMA/Prophet handle
that badly, so this service classifies each product/location series and routes
it to the appropriate estimator:

    Syntetos-Boylan-Approximation (Croston, de-biased)  -> intermittent / lumpy / erratic
    Simple Exponential Smoothing with day-of-week index -> smooth / fast movers

Classification follows the standard ADI / CV^2 quadrant:
    ADI  = average demand interval (periods / non-zero periods)
    CV^2 = squared coefficient of variation of non-zero demand

    ADI < 1.32, CV2 < 0.49  -> smooth
    ADI < 1.32, CV2 >= 0.49 -> erratic
    ADI >= 1.32, CV2 < 0.49 -> intermittent
    ADI >= 1.32, CV2 >= 0.49-> lumpy

Everything is pure NumPy/stdlib: CPU-only, milliseconds per series, and fully
explainable — which matters when a pharmacist asks why the number moved.
The LLM never produces the numbers; it only narrates them.
"""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

from app.core.config import settings
from app.db.supabase import pharmacy, rows

HISTORY_DAYS = 180
DEFAULT_HORIZON = 30
MODEL_VERSION = "v1.2"


# --------------------------------------------------------------- classification
def classify_series(series: List[float]) -> Tuple[str, float, float]:
    nz = [v for v in series if v > 0]
    if len(nz) < 2:
        return "intermittent", 999.0, 0.0
    adi = len(series) / len(nz)
    mean = sum(nz) / len(nz)
    var = sum((v - mean) ** 2 for v in nz) / len(nz)
    cv2 = (var / (mean ** 2)) if mean else 0.0
    if adi < 1.32 and cv2 < 0.49:
        return "smooth", adi, cv2
    if adi < 1.32:
        return "erratic", adi, cv2
    if cv2 < 0.49:
        return "intermittent", adi, cv2
    return "lumpy", adi, cv2


# --------------------------------------------------------------- estimators
def croston_sba(series: List[float], alpha: float = 0.1) -> Tuple[float, float]:
    """Croston's method with the Syntetos-Boylan de-biasing factor."""
    nz = [(i, v) for i, v in enumerate(series) if v > 0]
    if len(nz) < 2:
        mean = sum(series) / max(1, len(series))
        return mean, 0.0

    z = float(nz[0][1])
    p = float(nz[1][0] - nz[0][0]) or 1.0
    prev = nz[0][0]
    for i, v in nz[1:]:
        gap = float(i - prev) or 1.0
        z = alpha * v + (1 - alpha) * z
        p = alpha * gap + (1 - alpha) * p
        prev = i

    rate = (1 - alpha / 2.0) * (z / max(p, 1e-6))
    window = series[-60:] or series
    mae = sum(abs(v - rate) for v in window) / max(1, len(window))
    return rate, mae


def ses(series: List[float], alpha: float = 0.25) -> Tuple[float, float]:
    """Simple exponential smoothing — used for smooth, fast-moving products."""
    level = float(series[0])
    for v in series[1:]:
        level = alpha * v + (1 - alpha) * level
    window = series[-60:] or series
    mae = sum(abs(v - level) for v in window) / max(1, len(window))
    return level, mae


def day_of_week_profile(series: List[float], start: date) -> List[float]:
    buckets: List[List[float]] = [[] for _ in range(7)]
    for offset, value in enumerate(series):
        buckets[(start + timedelta(days=offset)).weekday()].append(value)
    overall = (sum(series) / len(series)) if series else 0.0
    if overall <= 0:
        return [1.0] * 7
    return [((sum(b) / len(b)) / overall) if b else 1.0 for b in buckets]


# --------------------------------------------------------------- data loading
def load_history(days: int = HISTORY_DAYS) -> Tuple[Dict[Tuple[str, str], List[float]], date]:
    """Pull dispense transactions and bucket them into daily series."""
    start = date.today() - timedelta(days=days)
    daily: Dict[Tuple[str, str], Dict[date, float]] = defaultdict(lambda: defaultdict(float))

    page, page_size = 0, 1000
    while True:
        batch = rows(
            pharmacy("inventory_transactions")
            .select("product_id, location_id, quantity, transaction_datetime")
            .eq("transaction_type", "dispense")
            .gte("transaction_datetime", start.isoformat())
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        if not batch:
            break
        for t in batch:
            day = datetime.fromisoformat(
                t["transaction_datetime"].replace("Z", "+00:00")
            ).date()
            daily[(t["product_id"], t["location_id"])][day] += abs(float(t["quantity"] or 0))
        if len(batch) < page_size:
            break
        page += 1

    series_map: Dict[Tuple[str, str], List[float]] = {}
    for key, by_day in daily.items():
        series_map[key] = [
            by_day.get(start + timedelta(days=d), 0.0) for d in range(days)
        ]
    return series_map, start


# --------------------------------------------------------------- pipeline
def forecast_series(series: List[float], start: date, horizon: int) -> dict:
    pattern, adi, cv2 = classify_series(series)
    if pattern == "smooth":
        rate, mae = ses(series)
        model = "ets_ses"
    else:
        rate, mae = croston_sba(series)
        model = "croston_sba"

    profile = day_of_week_profile(series, start)
    denominator = rate if rate > 0 else 1.0
    confidence = max(0.55, min(0.97, 1.0 - (mae / (denominator * 2.2))))

    points = []
    today = date.today()
    for h in range(1, horizon + 1):
        day = today + timedelta(days=h)
        point = rate * profile[day.weekday()] * (1 + 0.0011 * (len(series) + h))
        spread = mae * (1 + h / 45.0) * 1.28
        points.append(
            {
                "forecast_date": day,
                "predicted": round(max(0.0, point), 3),
                "lower": round(max(0.0, point - spread), 3),
                "upper": round(point + spread, 3),
            }
        )

    return {
        "model_name": model,
        "model_version": MODEL_VERSION,
        "demand_pattern": pattern,
        "adi": round(adi, 3),
        "cv2": round(cv2, 3),
        "daily_rate": round(rate, 3),
        "mae": round(mae, 3),
        "confidence": round(confidence, 4),
        "points": points,
    }


def run_forecast(horizon: int = DEFAULT_HORIZON, history_days: int = HISTORY_DAYS) -> dict:
    """Recompute every product/location forecast and persist to demand_forecasts."""
    series_map, start = load_history(history_days)
    if not series_map:
        return {"series": 0, "rows_written": 0, "message": "No dispensing history found."}

    payload, summary = [], []
    for (product_id, location_id), series in series_map.items():
        result = forecast_series(series, start, horizon)
        summary.append(
            {
                "product_id": product_id,
                "location_id": location_id,
                "model_name": result["model_name"],
                "demand_pattern": result["demand_pattern"],
                "daily_rate": result["daily_rate"],
                "confidence": result["confidence"],
            }
        )
        for p in result["points"]:
            payload.append(
                {
                    "forecast_id": str(uuid.uuid4()),
                    "organization_id": settings.ORGANIZATION_ID,
                    "location_id": location_id,
                    "product_id": product_id,
                    "forecast_date": p["forecast_date"].isoformat(),
                    "forecast_horizon_days": horizon,
                    "predicted_demand_qty": p["predicted"],
                    "lower_bound_qty": p["lower"],
                    "upper_bound_qty": p["upper"],
                    "model_name": result["model_name"],
                    "model_version": MODEL_VERSION,
                    "confidence_score": result["confidence"],
                    "demand_pattern": result["demand_pattern"],
                    "forecast_status": "completed",
                    "generated_at": datetime.utcnow().isoformat(),
                }
            )

    # replace the forward-looking window only; history stays intact
    pharmacy("demand_forecasts").delete().gte(
        "forecast_date", date.today().isoformat()
    ).execute()

    written = 0
    for i in range(0, len(payload), 500):
        chunk = payload[i : i + 500]
        pharmacy("demand_forecasts").upsert(
            chunk, on_conflict="location_id,product_id,forecast_date"
        ).execute()
        written += len(chunk)

    patterns: Dict[str, int] = defaultdict(int)
    models: Dict[str, int] = defaultdict(int)
    for s in summary:
        patterns[s["demand_pattern"]] += 1
        models[s["model_name"]] += 1

    return {
        "series": len(series_map),
        "rows_written": written,
        "horizon_days": horizon,
        "history_days": history_days,
        "pattern_mix": dict(patterns),
        "model_mix": dict(models),
        "avg_confidence": round(
            sum(s["confidence"] for s in summary) / max(1, len(summary)), 4
        ),
        "generated_at": datetime.utcnow().isoformat(),
    }


def backtest(sample: int = 40, holdout: int = 14) -> dict:
    """Hold out the last N days and score the model — proof it actually works."""
    series_map, start = load_history(HISTORY_DAYS)
    scored, errors = 0, []
    by_pattern: Dict[str, List[float]] = defaultdict(list)

    for (_pid, _lid), series in list(series_map.items())[:sample]:
        train, test = series[:-holdout], series[-holdout:]
        if len(train) < 40 or sum(test) == 0:
            continue
        pattern, _, _ = classify_series(train)
        rate, _ = ses(train) if pattern == "smooth" else croston_sba(train)
        profile = day_of_week_profile(train, start)

        predicted, actual = 0.0, sum(test)
        for h in range(holdout):
            day = start + timedelta(days=len(train) + h)
            predicted += rate * profile[day.weekday()]

        if actual > 0:
            err = abs(predicted - actual) / actual
            errors.append(err)
            by_pattern[pattern].append(err)
            scored += 1

    if not errors:
        return {"scored_series": 0, "message": "Not enough history to backtest."}

    return {
        "scored_series": scored,
        "holdout_days": holdout,
        "mape": round(sum(errors) / len(errors) * 100, 2),
        "accuracy_pct": round(max(0.0, 100 - (sum(errors) / len(errors) * 100)), 2),
        "by_pattern": {
            k: {"series": len(v), "mape": round(sum(v) / len(v) * 100, 2)}
            for k, v in by_pattern.items()
        },
    }
