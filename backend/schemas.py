from __future__ import annotations
from datetime import date, datetime
from pydantic import BaseModel, field_validator, model_validator


class QueryCreate(BaseModel):
    location_query: str
    location_name: str
    latitude: float
    longitude: float
    start_date: date | None = None
    end_date: date | None = None
    temperature_celsius: float | None = None
    weather_description: str | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "QueryCreate":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        return self


class QueryUpdate(BaseModel):
    notes: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "QueryUpdate":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        return self


class QueryOut(BaseModel):
    id: int
    location_query: str
    location_name: str
    latitude: float
    longitude: float
    start_date: date | None
    end_date: date | None
    temperature_celsius: float | None
    weather_description: str | None
    humidity: float | None
    wind_speed: float | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
