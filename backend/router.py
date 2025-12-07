from fastapi import HTTPException, APIRouter

router = APIRouter()

# Initialize RAG system
rag_system = None

# @router.post("/ask", response_model=Answer)
# async def ask_question(question: Question):
#     """Process question and return answer"""
#     try:
#         if rag_system is None:
#             raise HTTPException(status_code=500, detail="RAG system not initialized")

#         result = rag_system.query(question.question)

#         return Answer(
#             answer=result['answer'],
#             sources=result['sources']
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Q&A API is running",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/docs": "GET - API documentation",
        },
    }
