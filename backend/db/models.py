from datetime import datetime
from typing import List, Optional, Tuple

from sqlmodel import JSON, TIMESTAMP, Column, Field, SQLModel, text


class PredResults(SQLModel, table=True):
    id: str | None = Field(default=None, primary_key=True)
    status: str = Field(default="processing")
    clip_name: str
    prediction: str
    confidences: List[Tuple[str, float]] = Field(sa_column=Column(JSON))
    frames: List[str] = Field(sa_column=Column(JSON))
    time_taken: float
    created_datetime: Optional[datetime] = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        )
    )
