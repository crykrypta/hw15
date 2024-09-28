from pydantic import BaseModel


class LLMQuery(BaseModel):
    text: str
