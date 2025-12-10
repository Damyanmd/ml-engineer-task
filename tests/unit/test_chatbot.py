import pytest
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import app

client = TestClient(app)


class TestAskEndpoint:
    
    def test_ask_question_success(self):
        """Test successful streaming response"""
        # Mock the generate_chat function to yield chunks
        async def mock_generate_chat(question):
            chunks = ["Hello", " world", "!"]
            for chunk in chunks:
                yield chunk
        
        # Patch generate_chat in the backend.router module where it's defined
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "What is AI?"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["connection"] == "keep-alive"
            
            # Parse streamed response
            content = response.text
            lines = [line for line in content.split('\n') if line.startswith('data:')]
            
            assert len(lines) == 3
            
            # Verify chunks
            for line in lines:
                data = json.loads(line[6:])  # Remove 'data: ' prefix
                assert 'chunk' in data
                assert 'done' in data
                assert data['done'] is False
    
    def test_ask_question_empty_stream(self):
        """Test with empty response from generate_chat"""
        async def mock_generate_chat(question):
            return
            yield  # Empty generator
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "test"}
            )
            
            assert response.status_code == 200
    
    def test_ask_question_invalid_payload(self):
        """Test with invalid request payload"""
        response = client.post(
            "/ask",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_question_missing_payload(self):
        """Test with missing request body"""
        response = client.post("/ask")
        
        assert response.status_code == 422
    
    def test_ask_question_missing_question_field(self):
        """Test with empty JSON body"""
        response = client.post(
            "/ask",
            json={}
        )
        
        assert response.status_code == 422
    
    def test_ask_question_exception_handling(self):
        """Test exception handling in generate_chat"""
        async def mock_generate_chat_error(question):
            raise ValueError("Something went wrong")
            yield
        
        with patch('backend.router.generate_chat', side_effect=lambda q: mock_generate_chat_error(q)):
            # The test client will raise the exception since it occurs during streaming
            with pytest.raises(ValueError, match="Something went wrong"):
                response = client.post(
                    "/ask",
                    json={"question": "test"}
                )
    
    def test_ask_question_large_stream(self):
        """Test with large number of chunks"""
        async def mock_generate_chat(question):
            for i in range(100):
                yield f"chunk{i} "
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "Long question"}
            )
            
            assert response.status_code == 200
            lines = [line for line in response.text.split('\n') if line.startswith('data:')]
            assert len(lines) == 100
    
    def test_ask_question_special_characters(self):
        """Test with special characters in response"""
        async def mock_generate_chat(question):
            yield '{"special": "chars"}'
            yield "\n\t\r"
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "test"}
            )
            
            assert response.status_code == 200
            # Verify JSON is properly escaped
            lines = [line for line in response.text.split('\n') if line.startswith('data:')]
            for line in lines:
                data = json.loads(line[6:])
                assert isinstance(data['chunk'], str)
    
    
    def test_ask_question_single_chunk(self):
        """Test with single chunk response"""
        async def mock_generate_chat(question):
            yield "Single response"
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "simple question"}
            )
            
            assert response.status_code == 200
            lines = [line for line in response.text.split('\n') if line.startswith('data:')]
            assert len(lines) == 1
            data = json.loads(lines[0][6:])
            assert data['chunk'] == "Single response"
            assert data['done'] is False
    
    def test_ask_question_with_long_question(self):
        """Test with a very long question"""
        async def mock_generate_chat(question):
            yield "Response to long question"
        
        long_question = "What is " + "AI " * 100 + "?"
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": long_question}
            )
            
            assert response.status_code == 200
    
    def test_ask_question_content_type(self):
        """Test that response has correct content type"""
        async def mock_generate_chat(question):
            yield "test"
        
        with patch('backend.router.generate_chat', return_value=mock_generate_chat("test")):
            response = client.post(
                "/ask",
                json={"question": "test"}
            )
            
            assert "text/event-stream" in response.headers["content-type"]