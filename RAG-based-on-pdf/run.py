import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
import pdf_ingest
import chunking

if __name__ == "__main__":
    print("=" * 80)
    print("RAG PDF Processing Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Download PDFs
    print("STEP 1: Downloading PDFs")
    print("-" * 80)
    pdf_ingest.download_all_pdfs()
    
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Text from PDFs")
    print("=" * 80)
    
    # Step 2: Extract text from PDFs
    pdf_ingest.extract_all_pdfs()
    
    print("\n" + "=" * 80)
    print("STEP 3: Chunking Text for RAG")
    print("=" * 80)
    
    # Step 3: Chunk text and tables
    chunking.run_semantic_chunking()
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    print("\nOutput:")
    print("  PDFs: data/*.pdf")
    print("  Extracted text: data/extracted_text/")
    print("  Chunks: data/chunks/")
    print("\nNext steps:")
    print("  1. Generate embeddings for chunks")
    print("  2. Store in vector database")
    print("  3. Build RAG retrieval pipeline")
    print("=" * 80)
