-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for storing vector embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),  -- Adjust dimension based on your model (1536 for OpenAI Ada-002)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for faster similarity search using HNSW (Hierarchical Navigable Small World)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Create an index on metadata for filtering
CREATE INDEX idx_documents_metadata ON documents USING gin (metadata);

