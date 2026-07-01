"""Structured-parse schema for the cosmos-direction Runner. Imported LAZILY by
runner.py only when COSMOS_PARSE=structured, so ratchet's pure-Python venv never
needs pydantic/instructor for the default (regex) path.

The schema mirrors legacy _parse_direction's normalization: it strips + uppercases
before validating, so a lowercase model reply is not a spurious miss. Note the
strictness difference vs the regex, which matches the first keyword ANYWHERE in free
text: this schema requires the model to COMMIT a single field. That is the point
(no position-bias false positives), but it also means prose like "going downhill"
that never yields a clean field is a miss here where regex would have matched.
Compare the two on the SAME raw text before trusting an accuracy delta (see the
plan's parity task)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, field_validator

Direction = Literal["DOWNHILL", "UPHILL", "FLAT"]


class DirectionRead(BaseModel):
    direction: Direction

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v
