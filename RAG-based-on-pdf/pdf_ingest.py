import os
import requests
from pathlib import Path
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
import pdfplumber
import json
import warnings
warnings.filterwarnings('ignore')

# Define PDF sources
PDF_SOURCES = [
    "https://arxiv.org/pdf/1706.03762.pdf",  # Attention is All You Need
    "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
    "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
    "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
    "https://arxiv.org/pdf/1910.10683.pdf",  # T5
]

def create_data_folder():
    #Create data folder if it doesn't exist
    data_folder = Path(__file__).parent / "data"
    data_folder.mkdir(exist_ok=True)
    return data_folder

def download_pdf(url, save_path):
    """Download a PDF from the given URL and save it to the specified path"""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Successfully downloaded: {save_path.name}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {url}: {str(e)}")
        return False

def download_all_pdfs():
    """Download all PDFs from the sources list"""
    data_folder = create_data_folder()
    print(f"Data folder: {data_folder}\n")
    
    success_count = 0
    total_count = len(PDF_SOURCES)
    
    for idx, url in enumerate(PDF_SOURCES, 1):
        # Extract filename from URL
        filename = url.split('/')[-1]
        save_path = data_folder / filename
        
        # Skip if file already exists
        if save_path.exists():
            print(f"[{idx}/{total_count}] {filename} already exists, skipping...")
            success_count += 1
            continue
        
        # Download the PDF
        print(f"[{idx}/{total_count}] Downloading {filename}...")
        if download_pdf(url, save_path):
            success_count += 1
        print()
    
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded/verified: {success_count}/{total_count} PDFs")
    print(f"PDFs stored in: {data_folder.absolute()}")

def extract_text_pypdf2(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        text_content = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
        
        return text_content, num_pages
    except Exception as e:
        print(f"Error extracting text with PyPDF2 from {pdf_path.name}: {str(e)}")
        return [], 0

def extract_text_pdfminer(pdf_path):
    """Extract text from PDF using pdfminer.six with layout analysis"""
    try:
        # LAParams helps preserve layout structure
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=False,
            all_texts=False
        )
        
        text = pdfminer_extract_text(str(pdf_path), laparams=laparams)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text with pdfminer from {pdf_path.name}: {str(e)}")
        return ""

def extract_text_pdfplumber(pdf_path):
    """
    Extract text from PDF using pdfplumber with excellent table support.
    Tables are embedded directly into the text output.
    """
    try:
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from page
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                
                # Extract tables from page and embed them in text
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables, start=1):
                        # Format table as text and add to text_parts
                        table_text = format_table_as_text(table, page_num, table_idx)
                        text_parts.append(table_text)
        
        full_text = "\n".join(text_parts)
        return full_text.strip(), num_pages
    except Exception as e:
        print(f"Error extracting text with pdfplumber from {pdf_path.name}: {str(e)}")
        return "", 0

def format_table_as_text(table, page_num, table_num):
    """Format extracted table as readable text"""
    if not table:
        return ""
    
    text_parts = [f"\n[TABLE {table_num} on Page {page_num}]"]
    
    for row_idx, row in enumerate(table):
        # Clean None values and convert to strings
        clean_row = [str(cell) if cell is not None else "" for cell in row]
        
        if row_idx == 0:
            # Header row
            text_parts.append(" | ".join(clean_row))
            text_parts.append("-" * 80)
        else:
            text_parts.append(" | ".join(clean_row))
    
    text_parts.append(f"[END TABLE {table_num}]\n")
    return "\n".join(text_parts)

def extract_and_save_text(pdf_path, output_folder):
    """
    Extract text from a PDF using multiple methods and store the best result.
    Methods: pdfplumber, pdfminer.six, PyPDF2
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save extracted text
    
    Returns:
        Dictionary with extraction results
    """
    pdf_name = pdf_path.stem
    output_file = output_folder / f"{pdf_name}_full_text.txt"
    
    # Check if text already extracted
    if output_file.exists():
        print(f"\nSkipping: {pdf_path.name} (text already extracted)")
        return {
            'pdf_name': pdf_name,
            'skipped': True,
            'output_file': str(output_file)
        }
    
    print(f"\nExtracting text from: {pdf_path.name}")
    
    results = {
        'pdf_name': pdf_name,
        'pdf_path': str(pdf_path),
        'pdfplumber': {},
        'pdfminer': {},
        'pypdf2': {},
        'extraction_method': None,
        'all_methods_comparison': {}
    }
    
    num_pages = 0
    extraction_results = {}
    
    # Method 1: pdfplumber (best for tables)
    print("  - Method 1: pdfplumber (with table extraction)...")
    try:
        plumber_text, plumber_pages = extract_text_pdfplumber(pdf_path)
        num_pages = plumber_pages
        
        results['pdfplumber'] = {
            'num_pages': plumber_pages,
            'text_length': len(plumber_text),
            'success': len(plumber_text) > 0
        }
        
        if len(plumber_text) > 0:
            extraction_results['pdfplumber'] = {
                'text': plumber_text,
                'length': len(plumber_text),
                'score': len(plumber_text)
            }
            print(f"    ✓ Extracted {len(plumber_text):,} chars")
        else:
            print(f"    ✗ No text extracted")
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        results['pdfplumber'] = {'success': False, 'error': str(e)}
    
    # Method 2: pdfminer.six
    print("  - Method 2: pdfminer.six...")
    try:
        pdfminer_text = extract_text_pdfminer(pdf_path)
        
        # Get page count if not already set
        if num_pages == 0:
            pypdf2_content, num_pages = extract_text_pypdf2(pdf_path)
        
        results['pdfminer'] = {
            'num_pages': num_pages,
            'text_length': len(pdfminer_text),
            'success': len(pdfminer_text) > 0
        }
        
        if len(pdfminer_text) > 0:
            extraction_results['pdfminer'] = {
                'text': pdfminer_text,
                'length': len(pdfminer_text),
                'score': len(pdfminer_text)
            }
            print(f"    ✓ Extracted {len(pdfminer_text):,} chars")
        else:
            print(f"    ✗ No text extracted")
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        results['pdfminer'] = {'success': False, 'error': str(e)}
    
    # Method 3: PyPDF2
    print("  - Method 3: PyPDF2...")
    try:
        pypdf2_content, pypdf2_pages = extract_text_pypdf2(pdf_path)
        pypdf2_text = "\n\n".join([page['text'] for page in pypdf2_content])
        
        if num_pages == 0:
            num_pages = pypdf2_pages
        
        results['pypdf2'] = {
            'num_pages': pypdf2_pages,
            'text_length': len(pypdf2_text),
            'success': len(pypdf2_text) > 0
        }
        
        if len(pypdf2_text) > 0:
            extraction_results['pypdf2'] = {
                'text': pypdf2_text,
                'length': len(pypdf2_text),
                'score': len(pypdf2_text)
            }
            print(f"    ✓ Extracted {len(pypdf2_text):,} chars")
        else:
            print(f"    ✗ No text extracted")
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        results['pypdf2'] = {'success': False, 'error': str(e)}
    
    # Compare all methods and select the best one
    if not extraction_results:
        print("  ✗ All extraction methods failed!")
        return results
    
    print("\n  Comparing extraction methods:")
    for method, data in extraction_results.items():
        print(f"    - {method}: {data['length']:,} chars (score: {data['score']:,})")
    
    # Select the best method (highest score)
    best_method = max(extraction_results.items(), key=lambda x: x[1]['score'])
    selected_method = best_method[0]
    best_result = best_method[1]
    
    results['extraction_method'] = selected_method
    results['all_methods_comparison'] = {
        method: {'length': data['length'], 'score': data['score']} 
        for method, data in extraction_results.items()
    }
    
    final_text = best_result['text']
    
    print(f"\n  ✓ Selected method: {selected_method} (best score: {best_result['score']:,})")
    
    # Save extracted text (with embedded tables if from pdfplumber)
    text_file = output_folder / f"{pdf_name}_full_text.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"  ✓ Extracted {len(final_text):,} characters using {selected_method}")
    print(f"  ✓ Saved to: {text_file.name}")
    
    return results

def extract_all_pdfs():
    """Extract text from all downloaded PDFs"""
    data_folder = Path(__file__).parent / "data"
    
    if not data_folder.exists():
        print("Error: Data folder not found. Please download PDFs first.")
        return
    
    # Create output folder for extracted text
    output_folder = data_folder / "extracted_text"
    output_folder.mkdir(exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(data_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the data folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Output folder: {output_folder}\n")
    print("=" * 60)
    
    all_results = []
    
    for pdf_path in pdf_files:
        results = extract_and_save_text(pdf_path, output_folder)
        all_results.append(results)
    
    print("\n" + "=" * 60)
    print("Extraction Summary:")
    print(f"  Total PDFs processed: {len(all_results)}")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting pythoinPDF Ingestion Script")
    print("=" * 60)
    print()
    
    # Step 1: Download PDFs
    print("STEP 1: Downloading PDFs")
    print("-" * 60)
    download_all_pdfs()
    
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Text from PDFs")
    print("=" * 60)
    
    # Step 2: Extract text from PDFs
    extract_all_pdfs()
    
    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)
