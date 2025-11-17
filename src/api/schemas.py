from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class PredictRequest(BaseModel):
    store_id: int
    periods: int
    promo: int = 0
    stateholiday: int = 0
    schoolholiday: int = 0
