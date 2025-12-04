from langchain_postgres import PGVector
from embedding_model import embeddings
from dotenv import load_dotenv
import os

load_dotenv()

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=os.getenv("DATABASE_URL"),
)

test_embedding = embeddings.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")