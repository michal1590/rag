from typing import Optional

from pydantic import BaseModel


class Passage(BaseModel):
    question: str
    answer: str
    index: int
    generated_answer: Optional[str] = None
    similarity: Optional[float] = None
    LMM_judgment: Optional[str] = None
