from pydantic import BaseModel

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list = []