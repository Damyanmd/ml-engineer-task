from dotenv import load_dotenv
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def validate_key(key: str):
    if not os.environ.get(key):
        os.environ[key] = getpass.getpass(f"Enter API key for {key}: ")

def get_llm() -> str:
    validate_key("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", streaming=True)

def get_embedding_model() -> str:
    validate_key("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")