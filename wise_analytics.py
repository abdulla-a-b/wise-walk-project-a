#!/usr/bin/env python3
"""
WISE Walk™ — Nightly Analytics Pipeline
Developed by MBBC & Company · Technology Partner: guulba

Pulls observation + walk data from the Google Apps Script backend,
computes management statistics, and writes analytics/analytics.json,
which the WISE Walk web app loads to power the Management Insights
section of the dashboard.

Usage:
    WISE_SCRIPT_URL="https://script.google.com/macros/s/XXXX/exec" python wise_analytics.py

Runs nightly via GitHub Actions (.github/workflows/wise-analytics.yml).
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

DAY_MS = 86_400_000
PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}
OPEN_STATUSES = {"Inspected", "Solving"}
CLOSED_STATUSES = {"Executed", "Verified"}
CERTIFICATION = [  # mirrors WISE_FRAMEWORK.certification
    ("Platinum", 1500, "WISE Master Practitioner"),
    ("Gold", 800, "WISE Excellence Champion"),
    ("Silver", 400, "WISE Improvement Leader"),
    ("Bronze", 100, "Certified WISE Walker"),
]

OUT_PATH = os.path.join("analytics", "analytics.json")


def fetch_data(script_url: str):
    r = requests.get(script_url, params={"action": "all"}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    obs = pd.DataFrame(payload.get("rows", []))
    walks = pd.DataFrame(payload.get("walks", []))
    return obs, walks


def prepare(obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return obs
    obs = obs.copy()
    obs["ts"] = pd.to_numeric(obs["ts"], errors="coerce")
    obs["closedTs"] = pd.to_numeric(obs.get("closedTs"), errors="coerce")
    obs = obs.dropna(subset=["ts"])
    obs["dt"] = pd.to_datetime(obs["ts"], unit="ms", utc=True)
    obs["closure_days"] = (obs["closedTs"] - obs["ts"]) / DAY_MS
    obs.loc[obs["closure_days"] <= 0, "closure_days"] = np.nan
    return obs


def closure_stats(obs: pd.DataFrame) -> dict:
    d = obs["closure_days"].dropna() if not obs.empty else pd.Series(dtype=float)
    if d.empty:
        return {"medianDays": None, "p90Days": None, "meanDays": None, "stdDays": None, "n": 0}
    return {
        "medianDays": round(float(d.median()), 2),
        "p90Days": round(float(d.quantile(0.9)), 2),
        "meanDays": round(float(d.mean()), 2),
        "stdDays": round(float(d.std(ddof=0)), 2),
        "n": int(d.count()),
    }


def weekly_trend(obs: pd.DataFrame, walks: pd.DataFrame, weeks: int = 12) -> dict:
    now = datetime.now(timezone.utc)
    rows, series = [], []
    for i in range(weeks - 1, -1, -1):
        b = now - timedelta(days=7 * i)
        a = b - timedelta(days=7)
        n_obs = 0 if obs.empty else int(((obs["dt"] > a) & (obs["dt"] <= b)).sum())
        n_walks = 0
        if not walks.empty and "ts" in walks:
            wdt = pd.to_datetime(pd.to_numeric(walks["ts"], errors="coerce"), unit="ms", utc=True)
            n_walks = int(((wdt > a) & (wdt <= b)).sum())
        rows.append({"weekStart": a.date().isoformat(), "findings": n_obs, "walks": n_walks})
        series.append(n_obs)

    y = np.array(series, dtype=float)
    slope = 0.0
    if len(y) >= 2:
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
    last = y[-1] if len(y) else 0.0
    forecast = [max(0.0, last + slope * k) for k in range(1, 5)]
    return {
        "weekly": rows,
        "slopePerWeek": round(slope, 2),
        "forecastNext4": [round(v, 1) for v in forecast],
        "forecastNext4Total": int(round(sum(forecast))),
    }


def pareto(obs: pd.DataFrame) -> list:
    if obs.empty:
        return []
    counts = obs["category"].value_counts()
    total = int(counts.sum()) or 1
    out, cum = [], 0
    for cat, n in counts.items():
        cum += int(n)
        out.append({"category": str(cat), "count": int(n), "cumPct": round(cum / total * 100, 1)})
    return out


def area_risk(obs: pd.DataFrame) -> list:
    if obs.empty:
        return []
    open_df = obs[obs["status"].isin(OPEN_STATUSES)].copy()
    if open_df.empty:
        return []
    open_df["w"] = open_df["priority"].map(PRIORITY_WEIGHT).fillna(1)
    g = open_df.groupby("area").agg(score=("w", "sum"), open=("id", "count"),
                                    high=("priority", lambda s: int((s == "High").sum())))
    g = g.sort_values("score", ascending=False)
    return [{"area": str(a), "score": int(r.score), "open": int(r.open), "high": int(r.high)}
            for a, r in g.iterrows()]


def aging(obs: pd.DataFrame, top: int = 8) -> list:
    if obs.empty:
        return []
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    df = obs[(obs["status"].isin(OPEN_STATUSES)) & (obs["priority"] == "High")].copy()
    if df.empty:
        return []
    df["daysOpen"] = (now_ms - df["ts"]) / DAY_MS
    df = df.sort_values("daysOpen", ascending=False).head(top)
    return [{"id": str(r.id), "area": str(r.area), "category": str(r.category),
             "assignee": str(r.get("assignee", "") or ""), "daysOpen": round(float(r.daysOpen), 1)}
            for r in df.itertuples()]


def walker_stats(walks: pd.DataFrame) -> list:
    if walks.empty:
        return []
    w = walks.copy()
    w["total"] = pd.to_numeric(w.get("total"), errors="coerce").fillna(0)
    w["findings"] = pd.to_numeric(w.get("findings"), errors="coerce").fillna(0)
    g = w.groupby("walker").agg(walks=("id", "count"), findings=("findings", "sum"),
                                points=("total", "sum"), avgScore=("total", "mean"))
    g = g.sort_values("points", ascending=False)
    out = []
    for name, r in g.iterrows():
        tier, title = None, None
        for t, minp, ttl in CERTIFICATION:
            if r.points >= minp:
                tier, title = t, ttl
                break
        out.append({"walker": str(name), "walks": int(r.walks), "findings": int(r.findings),
                    "points": int(r.points), "avgScore": round(float(r.avgScore), 1),
                    "tier": tier, "title": title})
    return out


def category_area_matrix(obs: pd.DataFrame) -> dict:
    """Cross-tab of findings by category x area — where each problem type lives."""
    if obs.empty:
        return {"areas": [], "categories": [], "matrix": []}
    ct = pd.crosstab(obs["category"], obs["area"])
    return {
        "areas": [str(a) for a in ct.columns],
        "categories": [str(c) for c in ct.index],
        "matrix": ct.values.astype(int).tolist(),
    }


def main():
    script_url = os.environ.get("WISE_SCRIPT_URL", "").strip()
    if not script_url:
        print("ERROR: set WISE_SCRIPT_URL to the Apps Script Web App URL.", file=sys.stderr)
        sys.exit(1)

    obs, walks = fetch_data(script_url)
    obs = prepare(obs)

    total = int(len(obs))
    closed = int(obs["status"].isin(CLOSED_STATUSES).sum()) if total else 0

    analytics = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "python-pipeline",
        "totals": {
            "findings": total,
            "walks": int(len(walks)),
            "open": int(obs["status"].isin(OPEN_STATUSES).sum()) if total else 0,
            "closureRatePct": round(closed / total * 100, 1) if total else 0.0,
        },
        "closure": closure_stats(obs),
        "trend": weekly_trend(obs, walks),
        "pareto": pareto(obs),
        "areaRisk": area_risk(obs),
        "aging": aging(obs),
        "walkers": walker_stats(walks),
        "categoryAreaMatrix": category_area_matrix(obs),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}: {total} findings, {len(walks)} walks.")


if __name__ == "__main__":
    main()
