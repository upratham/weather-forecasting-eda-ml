from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import models, schemas
from .database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="WeatherFlow API", version="1.0.0", description="Real-time weather with CRUD persistence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WMO_CODES: dict[int, tuple[str, str]] = {
    0: ("☀️", "Clear sky"),
    1: ("🌤️", "Mainly clear"),
    2: ("⛅", "Partly cloudy"),
    3: ("☁️", "Overcast"),
    45: ("🌫️", "Fog"),
    48: ("🌫️", "Rime fog"),
    51: ("🌦️", "Light drizzle"),
    53: ("🌦️", "Moderate drizzle"),
    55: ("🌦️", "Dense drizzle"),
    61: ("🌧️", "Slight rain"),
    63: ("🌧️", "Moderate rain"),
    65: ("🌧️", "Heavy rain"),
    71: ("❄️", "Slight snow"),
    73: ("❄️", "Moderate snow"),
    75: ("❄️", "Heavy snow"),
    77: ("🌨️", "Snow grains"),
    80: ("🌦️", "Slight showers"),
    81: ("🌦️", "Moderate showers"),
    82: ("🌧️", "Violent showers"),
    85: ("🌨️", "Slight snow showers"),
    86: ("🌨️", "Heavy snow showers"),
    95: ("⛈️", "Thunderstorm"),
    96: ("⛈️", "Thunderstorm + hail"),
    99: ("⛈️", "Thunderstorm + heavy hail"),
}


def wmo(code: int) -> tuple[str, str]:
    return WMO_CODES.get(code, ("🌡️", f"Code {code}"))


# ── Geocoding ──────────────────────────────────────────────────────────────

async def _geocode(query: str) -> dict[str, Any]:
    """Resolve a location string to coordinates via Open-Meteo geocoding."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en", "format": "json"},
        )
        r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise HTTPException(status_code=404, detail=f"Location '{query}' not found. Try a city name or postal code.")
    hit = results[0]
    parts = [hit["name"], hit.get("admin1", ""), hit.get("country", "")]
    display = ", ".join(p for p in parts if p)
    return {
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
        "display_name": display,
        "timezone": hit.get("timezone", "auto"),
    }


@app.get("/api/geocode")
async def geocode_endpoint(query: str = Query(..., description="City name, postal code, or landmark")):
    return await _geocode(query)


# ── Weather ────────────────────────────────────────────────────────────────

async def _fetch_open_meteo(lat: float, lon: float, timezone: str) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": 7,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "weather_code,wind_speed_10m,wind_direction_10m,"
            "pressure_msl,precipitation,cloud_cover,uv_index"
        ),
        "daily": (
            "temperature_2m_max,temperature_2m_min,weather_code,"
            "precipitation_sum,wind_speed_10m_max,uv_index_max"
        ),
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
    return r.json()


@app.get("/api/weather")
async def get_weather(
    location: str | None = Query(None),
    lat: float | None = Query(None),
    lon: float | None = Query(None),
):
    if location:
        geo = await _geocode(location)
        lat, lon = geo["latitude"], geo["longitude"]
        display = geo["display_name"]
        timezone = geo.get("timezone", "auto")
    elif lat is not None and lon is not None:
        display = f"{lat:.4f}°, {lon:.4f}°"
        timezone = "auto"
    else:
        raise HTTPException(status_code=400, detail="Provide 'location' or both 'lat' and 'lon'.")

    raw = await _fetch_open_meteo(lat, lon, timezone)
    cur = raw["current"]
    daily = raw["daily"]

    code = cur.get("weather_code", 0)
    emoji, desc = wmo(code)

    forecast = []
    for i in range(len(daily["time"])):
        fc_code = daily["weather_code"][i]
        fe, fd = wmo(fc_code)
        forecast.append({
            "date": daily["time"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "weather_code": fc_code,
            "emoji": fe,
            "description": fd,
            "precipitation": daily["precipitation_sum"][i],
            "wind_max": daily["wind_speed_10m_max"][i],
            "uv_max": daily["uv_index_max"][i],
        })

    return {
        "location": display,
        "latitude": lat,
        "longitude": lon,
        "current": {
            "temperature": cur["temperature_2m"],
            "feels_like": cur["apparent_temperature"],
            "humidity": cur["relative_humidity_2m"],
            "weather_code": code,
            "emoji": emoji,
            "description": desc,
            "wind_speed": cur["wind_speed_10m"],
            "wind_direction": cur["wind_direction_10m"],
            "pressure": cur["pressure_msl"],
            "precipitation": cur["precipitation"],
            "cloud_cover": cur["cloud_cover"],
            "uv_index": cur["uv_index"],
            "time": cur["time"],
        },
        "forecast": forecast,
    }


# ── CRUD ───────────────────────────────────────────────────────────────────

@app.post("/api/queries", response_model=schemas.QueryOut, status_code=201)
def create_query(body: schemas.QueryCreate, db: Session = Depends(get_db)):
    row = models.WeatherQuery(**body.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@app.get("/api/queries", response_model=list[schemas.QueryOut])
def list_queries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return (
        db.query(models.WeatherQuery)
        .order_by(models.WeatherQuery.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.get("/api/queries/{query_id}", response_model=schemas.QueryOut)
def get_query(query_id: int, db: Session = Depends(get_db)):
    row = db.query(models.WeatherQuery).filter(models.WeatherQuery.id == query_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Query not found.")
    return row


@app.put("/api/queries/{query_id}", response_model=schemas.QueryOut)
def update_query(query_id: int, body: schemas.QueryUpdate, db: Session = Depends(get_db)):
    row = db.query(models.WeatherQuery).filter(models.WeatherQuery.id == query_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Query not found.")
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(row, field, value)
    row.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return row


@app.delete("/api/queries/{query_id}", status_code=204)
def delete_query(query_id: int, db: Session = Depends(get_db)):
    row = db.query(models.WeatherQuery).filter(models.WeatherQuery.id == query_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Query not found.")
    db.delete(row)
    db.commit()


# ── Export ─────────────────────────────────────────────────────────────────

def _rows_as_dicts(db: Session) -> list[dict[str, Any]]:
    rows = (
        db.query(models.WeatherQuery)
        .order_by(models.WeatherQuery.created_at.desc())
        .all()
    )
    return [schemas.QueryOut.model_validate(r).model_dump() for r in rows]


@app.get("/api/export")
def export_data(
    fmt: str = Query("json", description="json | csv | xml | markdown"),
    db: Session = Depends(get_db),
):
    data = _rows_as_dicts(db)

    if fmt == "json":
        content = json.dumps(data, indent=2, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=weather_queries.json"},
        )

    if fmt == "csv":
        buf = io.StringIO()
        if data:
            writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=weather_queries.csv"},
        )

    if fmt == "xml":
        root = ET.Element("weather_queries")
        for row in data:
            item = ET.SubElement(root, "query")
            for k, v in row.items():
                el = ET.SubElement(item, str(k))
                el.text = str(v) if v is not None else ""
        xml_str = ET.tostring(root, encoding="unicode")
        content = f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'
        return Response(
            content=content,
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=weather_queries.xml"},
        )

    if fmt == "markdown":
        if not data:
            content = "# Weather Queries\n\nNo records found."
        else:
            keys = list(data[0].keys())
            header = "| " + " | ".join(keys) + " |"
            sep = "| " + " | ".join("---" for _ in keys) + " |"
            body_rows = [
                "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |"
                for row in data
            ]
            content = "# Weather Queries\n\n" + "\n".join([header, sep] + body_rows)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=weather_queries.md"},
        )

    raise HTTPException(status_code=400, detail=f"Unknown format '{fmt}'. Use: json, csv, xml, markdown.")


# ── Serve built React frontend ─────────────────────────────────────────────

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        index = _FRONTEND_DIST / "index.html"
        return FileResponse(index)
