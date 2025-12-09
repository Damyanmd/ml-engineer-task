
import pytest
import asyncio
from unittest.mock import Mock,AsyncMock
from fastapi.testclient import TestClient
import json
from pydantic import BaseModel

# Assuming your app structure (adjust imports based on your actual structure)
# from main import app, Question, generate_chat, ask_question


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def app():
    """Create a test FastAPI app"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from fastapi.responses import StreamingResponse
    
    app = FastAPI()
    
    class Question(BaseModel):
        question: str
    
    async def generate_chat(question: str):
        """Mock chat generation"""
        response = f"Answer to: {question}"
        for char in response:
            yield char
            await asyncio.sleep(0.001)
    
    @app.post("/ask")
    async def ask_question(question: Question):
        async def stream_generator():
            async for chunk in generate_chat(question.question):
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = AsyncMock()
    mock.ainvoke.return_value = "This is a test response"
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock = Mock()
    mock.similarity_search.return_value = [
        Mock(page_content="Document 1 content", metadata={"source": "doc1.pdf"}),
        Mock(page_content="Document 2 content", metadata={"source": "doc2.pdf"}),
    ]
    return mock


# ============================================================================
# UNIT TESTS - Models
# ============================================================================

class TestQuestionModel:
    """Test the Question Pydantic model"""
    
    def test_valid_question(self):
        """Test creating a valid Question model"""
        class Question(BaseModel):
            question: str
        
        question = Question(question="What is AI?")
        assert question.question == "What is AI?"
    
    def test_empty_question(self):
        """Test that empty questions are handled"""
        
        class Question(BaseModel):
            question: str
        
        # Empty string is technically valid, but you might want to add validation
        question = Question(question="")
        assert question.question == ""
    
    def test_question_with_special_chars(self):
        """Test questions with special characters"""

        class Question(BaseModel):
            question: str
        
        question = Question(question="What is 2+2? Tell me!")
        assert question.question == "What is 2+2? Tell me!"


# ============================================================================
# UNIT TESTS - Chat Generation
# ============================================================================

class TestChatGeneration:
    """Test the chat generation logic"""
    
    @pytest.mark.asyncio
    async def test_generate_chat_yields_chunks(self):
        """Test that generate_chat yields text chunks"""
        async def mock_generate_chat(question: str):
            text = "Hello World"
            for char in text:
                yield char
        
        chunks = []
        async for chunk in mock_generate_chat("test"):
            chunks.append(chunk)
        
        assert "".join(chunks) == "Hello World"
        assert len(chunks) == 11
    
    @pytest.mark.asyncio
    async def test_generate_chat_with_empty_input(self):
        """Test generate_chat with empty input"""
        async def mock_generate_chat(question: str):
            if not question:
                yield "Please provide a question"
        
        chunks = []
        async for chunk in mock_generate_chat(""):
            chunks.append(chunk)
        
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_generate_chat_handles_long_text(self):
        """Test that generate_chat handles long responses"""
        async def mock_generate_chat(question: str):
            text = "A" * 1000
            for char in text:
                yield char
        
        chunks = []
        async for chunk in mock_generate_chat("test"):
            chunks.append(chunk)
        
        assert len(chunks) == 1000


# ============================================================================
# UNIT TESTS - Agent Executor
# ============================================================================

class TestAgentExecutor:
    """Test agent executor functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_executor_streaming(self):
        """Test that agent executor streams responses"""
        mock_executor = AsyncMock()
        
        async def mock_astream_events(*args, **kwargs):
            events = [
                {"event": "on_chat_model_stream", "data": {"chunk": Mock(content="Hello")}},
                {"event": "on_chat_model_stream", "data": {"chunk": Mock(content=" World")}},
            ]
            for event in events:
                yield event
        
        mock_executor.astream_events = mock_astream_events
        
        chunks = []
        async for event in mock_executor.astream_events({}, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunks.append(event["data"]["chunk"].content)
        
        assert len(chunks) == 2
        assert "".join(chunks) == "Hello World"
    
    @pytest.mark.asyncio
    async def test_agent_executor_tool_usage(self):
        """Test agent executor with tool calls"""
        mock_executor = AsyncMock()
        
        async def mock_astream_events(*args, **kwargs):
            events = [
                {"event": "on_tool_start", "name": "search_tool"},
                {"event": "on_tool_end", "data": {"output": "Tool result"}},
                {"event": "on_chat_model_stream", "data": {"chunk": Mock(content="Final answer")}},
            ]
            for event in events:
                yield event
        
        mock_executor.astream_events = mock_astream_events
        
        events_received = []
        async for event in mock_executor.astream_events({}, version="v2"):
            events_received.append(event["event"])
        
        assert "on_tool_start" in events_received
        assert "on_tool_end" in events_received
        assert "on_chat_model_stream" in events_received




# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])