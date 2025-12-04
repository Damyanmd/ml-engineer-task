from langchain_community.document_loaders import PyMuPDFLoader
from connect_db import vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pymupdf
import os
from pathlib import Path

load_dotenv()
documents_path = os.getenv("DOCUMENTS_PATH")

# Convert to Path object
folder_path = Path(documents_path)
pdf_files = list(folder_path.glob("*.pdf"))

all_documents = []
errors = []

for pdf_file in pdf_files:
    try:
        # Skip empty files
        if pdf_file.stat().st_size == 0:
            errors.append(f"{pdf_file.name}: Empty file")
            continue
        
        loader = PyMuPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"✓ {pdf_file.name}: {len(documents)} pages")
    except (pymupdf.EmptyFileError, pymupdf.FileDataError) as e:
        errors.append(f"{pdf_file.name}: Corrupted PDF")
    except Exception as e:
        errors.append(f"{pdf_file.name}: {type(e).__name__}")

print(f"\nLoaded {len(all_documents)} pages from {len(pdf_files) - len(errors)} files")
if errors:
    print(f"Errors in {len(errors)} files:")
    for error in errors:
        print(f"  - {error}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(all_documents)
print(f"Split blog post into {len(all_splits)} sub-documents.")

# DEBUG: Check if splits have content
print(f"\nFirst split preview: {all_splits[0].page_content[:200] if all_splits else 'No splits'}")

# DEBUG: Add documents with error handling
try:
    print("\nAdding documents to vector store...")
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Successfully added {len(document_ids)} documents")
    print(f"Sample IDs: {document_ids[:3]}")
    
    # VERIFY: Check if documents are actually in the database
    print("\nVerifying embeddings in database...")
    test_results = vector_store.similarity_search("test", k=1)
    print(f"Found {len(test_results)} documents in vector store")
    
except Exception as e:
    print(f"Error adding documents: {e}")
    import traceback
    traceback.print_exc()