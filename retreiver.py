from connect_db import index
from embedding_model import embeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.tools import tool


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant context passages from the vector database
    for a given user query.
    """
    # load to your BM25Encoder object
    bm25_encoder = BM25Encoder().load("bm25_encoder.json")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )

    result = retriever.invoke(query)
    contents = "\n\n---\n\n".join([doc.page_content for doc in result])

    return contents

