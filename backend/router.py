from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse

import json

from backend.generator import generate_chat
from backend.schemas import Question

router = APIRouter()


@router.post("/ask")
async def ask_question(question: Question):
    """Process question and return streaming answer"""
    try:
        async def stream_generator():
            answer = ""
            async for chunk in generate_chat(question.question):
                answer += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/")
async def serve_frontend():
    return FileResponse(path="frontend/index.html", media_type="text/html")
