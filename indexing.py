from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import pymupdf
import os
import json
from pathlib import Path
from langchain_community.retrievers import PineconeHybridSearchRetriever
from connect_db import index
from embedding_model import embeddings

load_dotenv()

# Configuration
documents_path = os.getenv("DOCUMENTS_PATH")
BATCH_SIZE = 100

# Load metadata from JSONL file
def load_metadata(metadata_path):
    """Load metadata from JSONL file and create a lookup dictionary by UUID"""
    metadata_lookup = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc_meta = json.loads(line)
            # Use UUID as key (you can also use title if that's how files are named)
            metadata_lookup[doc_meta['uuid']] = doc_meta
    return metadata_lookup

# Extract UUID from filename (adjust based on your naming convention)
def get_uuid_from_filename(filename):
    """Extract UUID from PDF filename. Adjust this based on your file naming."""
    # Example: if files are named like "uuid.pdf"
    return filename.stem  # Returns filename without extension
    # OR if UUID is part of filename: return filename.stem.split('_')[0]

# Initialize encoders
print("Loading embedding models...")
dense_model = embeddings
sparse_model = BM25Encoder().default()

# Load metadata
metadata_path = Path("data/metadata.jsonl")  # Adjust path as needed
print(f"Loading metadata from {metadata_path}...")
metadata_lookup = load_metadata(metadata_path)
print(f"Loaded metadata for {len(metadata_lookup)} documents")

# Load PDFs
folder_path = Path(documents_path)
pdf_files = list(folder_path.glob("*.pdf"))
all_documents = []
errors = []

print(f"\nProcessing {len(pdf_files)} PDF files...")
for pdf_file in pdf_files:
    try:
        if pdf_file.stat().st_size == 0:
            errors.append(f"{pdf_file.name}: Empty file")
            continue
        
        # Get UUID from filename
        file_uuid = get_uuid_from_filename(pdf_file)
        
        # Load PDF
        loader = PyMuPDFLoader(str(pdf_file))
        documents = loader.load()
        
        # Enrich each document with metadata from JSONL
        if file_uuid in metadata_lookup:
            doc_metadata = metadata_lookup[file_uuid]
            for doc in documents:
                # Merge existing metadata with JSONL metadata
                doc.metadata.update({
                    'uuid': doc_metadata['uuid'],
                    'title': doc_metadata['title'],
                    'industries': doc_metadata['industries'],
                    'date': doc_metadata['date'],
                    'country_codes': doc_metadata['country_codes']
                })
        else:
            print(f"⚠ No metadata found for {pdf_file.name}")
        
        all_documents.extend(documents)
        print(f"✓ {pdf_file.name}: {len(documents)} pages")
        
    except (pymupdf.EmptyFileError, pymupdf.FileDataError) as e:
        errors.append(f"{pdf_file.name}: Corrupted PDF")
    except Exception as e:
        errors.append(f"{pdf_file.name}: {type(e).__name__} - {str(e)}")

print(f"\nLoaded {len(all_documents)} pages from {len(pdf_files) - len(errors)} files")
if errors:
    print(f"Errors in {len(errors)} files:")
    for error in errors:
        print(f"  - {error}")

# Split documents
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(all_documents)
print(f"Created {len(all_splits)} text chunks")

# Prepare corpus for BM25
corpus_texts = [doc.page_content for doc in all_splits]

# Fit and save BM25 encoder
print("\nFitting BM25 encoder...")
sparse_model.fit(corpus_texts)
sparse_model.dump("bm25_encoder.json")

# Load BM25 encoder
bm25_encoder = BM25Encoder().load("bm25_encoder.json")


# Create retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, 
    sparse_encoder=bm25_encoder, 
    index=index
)

# Upload documents to Pinecone with metadata
print(f"\nUploading {len(all_splits)} chunks to Pinecone with metadata...")
try:
    retriever.add_texts(
        texts=[doc.page_content for doc in all_splits],
        metadatas=[doc.metadata for doc in all_splits]
    )
    print("✓ Upload complete!")
except Exception as e:
    print(f"✗ Upload failed: {str(e)}")
    raise

# Verify upload
try:
    stats = index.describe_index_stats()
    print(f"\n✓ Index now contains {stats['total_vector_count']} vectors")
except Exception as e:
    print(f"⚠ Could not verify index stats: {str(e)}")

# Test query with metadata filtering (optional)
print("\n--- Testing retrieval with metadata ---")
test_query = "GDP growth"
results = retriever.get_relevant_documents(test_query, top_k=3)
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Content: {doc.page_content[:100]}...")
    print(f"  Metadata: {doc.metadata}")