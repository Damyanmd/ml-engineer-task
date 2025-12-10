import os
from pinecone import Pinecone, ServerlessSpec

from backend.utils import validate_key

def get_index():
    index_name = "isi-data-test"

    validate_key("PINECONE_API_KEY")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,  
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)
