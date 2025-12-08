from fastapi import HTTPException, APIRouter

from backend.generator import generate_chat
from backend.schemas import Question

router = APIRouter()


@router.post("/ask")
async def ask_question(question: Question):
    """Process question and return answer"""
    try:
        answer = ""
        async for chunk in generate_chat(question.question):
            answer += chunk

        return {
            "answer": answer,
            "sources": []  # Add your actual sources here if available
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

