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
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            doc_meta = json.loads(line)
            metadata_lookup[doc_meta["uuid"]] = doc_meta
    return metadata_lookup


# Extract UUID from filename
def get_uuid_from_filename(filename):
    """Extract UUID from PDF filename. Adjust this based on your file naming."""
    return filename.stem


# Load metadata
metadata_path = Path("data/metadata.jsonl")
print(f"Loading metadata from {metadata_path}...")
metadata_lookup = load_metadata(metadata_path)
print(f"Loaded metadata for {len(metadata_lookup)} documents\n")


# Load PDFs and check which ones need to be embedded
folder_path = Path(documents_path)
pdf_files = list(folder_path.glob("*.pdf"))


all_documents = []
errors = []

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
                doc.metadata.update(
                    {
                        "uuid": doc_metadata["uuid"],
                        "title": doc_metadata["title"],
                        "industries": doc_metadata["industries"],
                        "date": doc_metadata["date"],
                        "country_codes": doc_metadata["country_codes"],
                    }
                )
        else:
            print(f"⚠ No metadata found for {pdf_file.name}")

        all_documents.extend(documents)
        print(f"✓ {pdf_file.name}: {len(documents)} pages - NEW (will be embedded)")

    except (pymupdf.EmptyFileError, pymupdf.FileDataError) as e:
        errors.append(f"{pdf_file.name}: Corrupted PDF")
    except Exception as e:
        errors.append(f"{pdf_file.name}: {type(e).__name__} - {str(e)}")

print(f"\n{'=' * 60}")
print(f"Summary:")
print(f"  Total PDF files found: {len(pdf_files)}")
print(f"  New files to embed: {len(pdf_files) - len(errors)}")
print(f"  Errors: {len(errors)}")
print(f"  Total pages loaded: {len(all_documents)}")
print(f"{'=' * 60}\n")

if errors:
    print(f"Errors encountered:")
    for error in errors:
        print(f"  - {error}")
    print()

# Split documents
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(all_documents)
print(f"Created {len(all_splits)} text chunks\n")

# Filter out empty or very short texts to prevent sparse vector errors
print("Filtering valid chunks...")
valid_splits = [
    doc for doc in all_splits 
    if doc.page_content.strip() and len(doc.page_content.strip()) > 10
]
print(f"Valid chunks after filtering: {len(valid_splits)}")

if len(valid_splits) == 0:
    raise ValueError("No valid text chunks found after filtering!")

# Prepare corpus for BM25
corpus_texts = [doc.page_content for doc in valid_splits]

# Initialize and fit BM25 encoder
print("Initializing and fitting BM25 encoder...")
bm25_encoder = BM25Encoder()
bm25_encoder.fit(corpus_texts)
bm25_encoder.dump("bm25_encoder.json")
print("✓ BM25 encoder fitted and saved\n")

# Create retriever with the fitted encoder
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, 
    sparse_encoder=bm25_encoder, 
    index=index
)

# Upload documents to Pinecone with metadata
print(f"Uploading {len(valid_splits)} chunks to Pinecone with metadata...")
try:
    retriever.add_texts(
        texts=[doc.page_content for doc in valid_splits],
        metadatas=[doc.metadata for doc in valid_splits],
    )
    print("✓ Upload complete!")
except Exception as e:
    print(f"✗ Upload failed: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# Verify upload
try:
    stats = index.describe_index_stats()
    print(f"\n✓ Index now contains {stats['total_vector_count']} vectors")
except Exception as e:
    print(f"⚠ Could not verify index stats: {str(e)}")