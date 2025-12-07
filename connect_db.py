import os

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

index_name = "isi-data-hybrid"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(index_name)
