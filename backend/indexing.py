import logging
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

from backend.connect_db import get_index
from backend.utils import get_embedding_model

# Configuration
index = get_index()
embedding_model = get_embedding_model()

DOCUMENTS_PATH = Path("data")
METADATA_PATH = Path("data/metadata.jsonl")
BM25_ENCODER_PATH = "backend/bm25_encoder.json"
BATCH_SIZE = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 10

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """Load metadata from JSONL file and create a lookup dictionary by UUID."""
    logger.info(f"Loading metadata from {metadata_path}...")
    metadata_lookup = {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                doc_meta = json.loads(line)
                metadata_lookup[doc_meta["uuid"]] = doc_meta
        
        logger.info(f"Loaded metadata for {len(metadata_lookup)} documents")
        return metadata_lookup
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        raise


def load_pdf_with_metadata(
    pdf_file: Path,
    metadata_lookup: Dict[str, Dict]
) -> Tuple[List, str]:
    """
    Load a single PDF and enrich with metadata.
    
    Returns:
        Tuple of (documents list, error message or None)
    """
    try:
        if pdf_file.stat().st_size == 0:
            return [], f"Empty file"

        file_uuid = pdf_file.stem
        loader = PyMuPDFLoader(str(pdf_file))
        documents = loader.load()

        # Enrich with metadata
        if file_uuid in metadata_lookup:
            doc_metadata = metadata_lookup[file_uuid]
            for doc in documents:
                doc.metadata.update({
                    "uuid": doc_metadata["uuid"],
                    "title": doc_metadata["title"],
                    "industries": doc_metadata["industries"],
                    "date": doc_metadata["date"],
                    "country_codes": doc_metadata["country_codes"],
                })
        else:
            logger.warning(f"No metadata found for {pdf_file.name}")

        logger.info(f"✓ {pdf_file.name}: {len(documents)} pages loaded")
        return documents, None

    except (pymupdf.EmptyFileError, pymupdf.FileDataError):
        return [], "Corrupted PDF"
    except Exception as e:
        return [], f"{type(e).__name__} - {str(e)}"


def load_all_pdfs(
    folder_path: Path,
    metadata_lookup: Dict[str, Dict]
) -> Tuple[List, List[str]]:
    """
    Load all PDFs from folder with metadata.
    
    Returns:
        Tuple of (all documents, list of errors)
    """
    logger.info(f"Loading PDFs from {folder_path}...")
    pdf_files = list(folder_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    all_documents = []
    errors = []

    for pdf_file in pdf_files:
        documents, error = load_pdf_with_metadata(pdf_file, metadata_lookup)
        
        if error:
            error_msg = f"{pdf_file.name}: {error}"
            errors.append(error_msg)
            logger.error(error_msg)
        else:
            all_documents.extend(documents)

    return all_documents, errors


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """Split documents into chunks."""
    logger.info("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    
    splits = text_splitter.split_documents(documents)
    logger.info(f"Created {len(splits)} text chunks")
    
    return splits


def filter_valid_chunks(splits: List, min_length: int) -> List:
    """Filter out empty or very short text chunks."""
    logger.info("Filtering valid chunks...")
    
    valid_splits = [
        doc for doc in splits 
        if doc.page_content.strip() and len(doc.page_content.strip()) > min_length
    ]
    
    logger.info(f"Valid chunks after filtering: {len(valid_splits)}")
    
    if len(valid_splits) == 0:
        raise ValueError("No valid text chunks found after filtering!")
    
    return valid_splits


def create_and_save_bm25_encoder(corpus_texts: List[str], save_path: str) -> BM25Encoder:
    """Initialize, fit, and save BM25 encoder."""
    logger.info("Initializing and fitting BM25 encoder...")
    
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(corpus_texts)
    bm25_encoder.dump(save_path)
    
    logger.info(f"✓ BM25 encoder fitted and saved to {save_path}")
    return bm25_encoder


def upload_to_pinecone(retriever: PineconeHybridSearchRetriever, documents: List) -> None:
    """Upload documents to Pinecone with metadata."""
    logger.info(f"Uploading {len(documents)} chunks to Pinecone...")
    
    try:
        retriever.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        logger.info("✓ Upload complete!")
    except Exception as e:
        logger.error(f"✗ Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def verify_index_stats(index) -> None:
    """Verify and log index statistics."""
    try:
        stats = index.describe_index_stats()
        logger.info(f"✓ Index now contains {stats['total_vector_count']} vectors")
    except Exception as e:
        logger.warning(f"Could not verify index stats: {str(e)}")


def print_summary(total_files: int, errors: List[str], total_pages: int) -> None:
    """Print processing summary."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("Summary:")
    print(f"  Total PDF files found: {total_files}")
    print(f"  Successfully processed: {total_files - len(errors)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total pages loaded: {total_pages}")
    print(f"{separator}\n")

def main():
    """Main execution function."""
    try:
        # Load metadata
        metadata_lookup = load_metadata(METADATA_PATH)

        # Load PDFs
        all_documents, errors = load_all_pdfs(DOCUMENTS_PATH, metadata_lookup)
        
        # Print summary
        pdf_files = list(DOCUMENTS_PATH.glob("*.pdf"))
        print_summary(len(pdf_files), errors, len(all_documents))

        if not all_documents:
            logger.error("No documents were loaded successfully. Exiting.")
            return

        # Split documents
        all_splits = split_documents(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)

        # Filter valid chunks
        valid_splits = filter_valid_chunks(all_splits, MIN_CHUNK_LENGTH)

        # Prepare corpus for BM25
        corpus_texts = [doc.page_content for doc in valid_splits]

        # Create BM25 encoder
        bm25_encoder = create_and_save_bm25_encoder(corpus_texts, BM25_ENCODER_PATH)

        # Create retriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=embedding_model, 
            sparse_encoder=bm25_encoder, 
            index=index
        )

        # Upload to Pinecone
        upload_to_pinecone(retriever, valid_splits)

        # Verify upload
        verify_index_stats(index)

        logger.info("✓ All operations completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()