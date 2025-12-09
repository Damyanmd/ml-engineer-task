from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.tools import tool

from backend.connect_db import index
from backend.embedding_model import embeddings


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant context passages from the vector database
    for a given user query.
    """
    bm25_encoder = BM25Encoder().load("backend/bm25_encoder.json")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )

    result = retriever.invoke(query)
    contents = "\n\n---\n\n".join([doc.page_content for doc in result])

    return contents

