""""The one JSON shape this project produces. Baseline and model both target it"""
from __future__ import annotations
from pydantic import BaseModel, Field


class Item(BaseModel):
  name: str
  qty: int | None = None
  price: float | None = None


class Bill(BaseModel):
  items: list[Item] = Field(default_factory=list)
  subtotal: float | None = None
  tax: float | None = None
  service: float | None = None
  etc: float | None = None
  total: float | None = None