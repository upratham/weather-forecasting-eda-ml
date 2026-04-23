from datetime import date, datetime
from sqlalchemy import String, Float, Date, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class WeatherQuery(Base):
    __tablename__ = "weather_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    location_query: Mapped[str] = mapped_column(String(200))
    location_name: Mapped[str] = mapped_column(String(300))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    temperature_celsius: Mapped[float | None] = mapped_column(Float, nullable=True)
    weather_description: Mapped[str | None] = mapped_column(String(100), nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
