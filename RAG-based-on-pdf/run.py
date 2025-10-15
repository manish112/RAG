import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the pdf_ingest module
import pdf_ingest

if __name__ == "__main__":
    print("=" * 60)
    print("PDF Ingestion and Text Extraction Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Download PDFs
    print("STEP 1: Downloading PDFs")
    print("-" * 60)
    pdf_ingest.download_all_pdfs()
    
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Text from PDFs")
    print("=" * 60)
    
    # Step 2: Extract text from PDFs
    pdf_ingest.extract_all_pdfs()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
