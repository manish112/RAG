"""
Semantic Chunking using LangChain
Creates semantically coherent chunks for RAG applications using recursive splitting
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class SemanticChunkingPipeline:
    """
    Semantic chunking pipeline using LangChain's RecursiveCharacterTextSplitter
    Creates chunks that respect semantic boundaries (paragraphs, sentences, words)
    Fixed chunk size: 1000 characters with 200 character overlap
    """
    
    def __init__(self):
        """
        Initialize the semantic chunking pipeline with fixed parameters
        """
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Default separators for semantic splitting (ordered by preference)
        self.separators = [
            "\n\n",      # Double newline (paragraph breaks)
            "\n",        # Single newline
            ". ",        # Sentence endings
            "! ",        # Exclamation sentences
            "? ",        # Question sentences
            "; ",        # Semicolon
            ", ",        # Comma
            " ",         # Space (word boundary)
            ""           # Character-level (last resort)
        ]
        
        print(f"Initializing RecursiveCharacterTextSplitter...")
        print(f"  Chunk size: 1000 characters (~250 tokens)")
        print(f"  Overlap: 200 characters (~50 tokens)")
        
        # Initialize recursive character text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        print("✓ Text splitter initialized")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Chunk text using recursive character splitting
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of LangChain Document objects with chunks
        """
        if metadata is None:
            metadata = {}
        
        # Create document
        doc = Document(page_content=text, metadata=metadata)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        return chunks
    
    def chunk_file(self, file_path: Path, pdf_name: str = None) -> List[Dict]:
        """
        Chunk a single text file and incorporate tables from corresponding JSON file
        
        Args:
            file_path: Path to text file
            pdf_name: Name of the PDF (extracted from filename if not provided)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if pdf_name is None:
            pdf_name = file_path.stem.replace('_full_text', '')
        
        print(f"\nProcessing: {pdf_name}")
        
        # Read text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"  Original text length: {len(text):,} characters")
        
        # Check for corresponding tables JSON file
        tables_file = file_path.parent / f"{pdf_name}_tables.json"
        tables_data = None
        if tables_file.exists():
            try:
                with open(tables_file, 'r', encoding='utf-8') as f:
                    tables_data = json.load(f)
                print(f"  Found {tables_data.get('tables_count', 0)} tables")
            except Exception as e:
                print(f"  Warning: Could not load tables file: {e}")
        
        # Chunk the text
        metadata = {
            'pdf_name': pdf_name,
            'source_file': str(file_path),
            'has_tables': tables_data is not None
        }
        
        print(f"  Creating semantic chunks...")
        langchain_docs = self.chunk_text(text, metadata)
        
        # Convert to dictionary format
        chunks = []
        for idx, doc in enumerate(langchain_docs, 1):
            chunk_dict = {
                'chunk_id': idx,
                'text': doc.page_content,
                'length': len(doc.page_content),
                'pdf_name': pdf_name,
                'method': 'semantic',
                'metadata': doc.metadata
            }
            chunks.append(chunk_dict)
        
        # Add tables as separate chunks if they exist
        if tables_data and tables_data.get('tables'):
            print(f"  Adding {len(tables_data['tables'])} tables as separate chunks...")
            for table_info in tables_data['tables']:
                table_text = self._format_table_for_rag(table_info, pdf_name)
                
                # Create a chunk for the table
                table_chunk = {
                    'chunk_id': len(chunks) + 1,
                    'text': table_text,
                    'length': len(table_text),
                    'pdf_name': pdf_name,
                    'method': 'table',
                    'is_table': True,
                    'table_page': table_info.get('page'),
                    'table_number': table_info.get('table_number'),
                    'metadata': {
                        'pdf_name': pdf_name,
                        'source_file': str(file_path),
                        'is_table': True,
                        'page': table_info.get('page'),
                        'table_number': table_info.get('table_number')
                    }
                }
                chunks.append(table_chunk)
        
        # Calculate statistics
        text_chunks = [c for c in chunks if not c.get('is_table', False)]
        table_chunks = [c for c in chunks if c.get('is_table', False)]
        
        avg_length = sum(c['length'] for c in text_chunks) / len(text_chunks) if text_chunks else 0
        min_length = min(c['length'] for c in text_chunks) if text_chunks else 0
        max_length = max(c['length'] for c in text_chunks) if text_chunks else 0
        
        print(f"  ✓ Created {len(text_chunks)} text chunks")
        if table_chunks:
            print(f"  ✓ Created {len(table_chunks)} table chunks")
        print(f"    - Average text chunk length: {int(avg_length)} chars (~{int(avg_length/4)} tokens)")
        print(f"    - Min/Max text chunk length: {min_length}/{max_length} chars")
        
        return chunks
    
    def _format_table_for_rag(self, table_info: Dict, pdf_name: str) -> str:
        """
        Format a table for RAG retrieval
        Converts table data into a readable text format
        
        Args:
            table_info: Dictionary containing table data
            pdf_name: Name of the PDF
            
        Returns:
            Formatted table as string
        """
        page = table_info.get('page', 'unknown')
        table_num = table_info.get('table_number', 'unknown')
        table_data = table_info.get('data', [])
        
        lines = [
            f"[TABLE {table_num} from Page {page} of {pdf_name}]",
            ""
        ]
        
        if not table_data:
            lines.append("(Empty table)")
            return "\n".join(lines)
        
        # Format table rows
        for row_idx, row in enumerate(table_data):
            # Clean and join cells
            cells = [str(cell).strip() if cell else "" for cell in row]
            
            if row_idx == 0:
                # Header row
                lines.append(" | ".join(cells))
                lines.append("-" * min(80, len(" | ".join(cells))))
            else:
                # Data rows
                lines.append(" | ".join(cells))
        
        lines.append("")
        lines.append(f"[END TABLE {table_num}]")
        
        return "\n".join(lines)
    
    def process_all_texts(self, output_folder_name: str = "chunks"):
        """
        Process all extracted texts from data/extracted_text folder
        
        Args:
            output_folder_name: Name of the output folder for chunks
        """
        data_folder = Path(__file__).parent / "data" / "extracted_text"
        
        if not data_folder.exists():
            print("Error: No extracted text found. Please run pdf_ingest.py first.")
            return
        
        # Create output folder
        chunks_folder = Path(__file__).parent / "data" / output_folder_name
        chunks_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all text files
        text_files = list(data_folder.glob("*_full_text.txt"))
        
        if not text_files:
            print("No text files found to chunk.")
            return
        
        print("=" * 80)
        print(f"Semantic Chunking Pipeline")
        print("=" * 80)
        print(f"Found {len(text_files)} text files to process")
        print(f"Chunk size: 1000 characters (~250 tokens)")
        print(f"Overlap: 200 characters (~50 tokens)")
        print(f"Output folder: {chunks_folder}")
        print("=" * 80)
        
        all_stats = []
        
        for text_file in text_files:
            # Process file
            chunks = self.chunk_file(text_file)
            
            # Save chunks
            pdf_name = text_file.stem.replace('_full_text', '')
            chunks_file = chunks_folder / f"{pdf_name}_chunks.json"
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'pdf_name': pdf_name,
                    'method': 'semantic_recursive',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'total_chunks': len(chunks),
                    'chunks': chunks
                }, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved to: {chunks_file.name}")
            
            # Calculate statistics
            text_chunks = [c for c in chunks if not c.get('is_table', False)]
            table_chunks = [c for c in chunks if c.get('is_table', False)]
            
            avg_length = sum(c['length'] for c in text_chunks) / len(text_chunks) if text_chunks else 0
            
            stats = {
                'pdf_name': pdf_name,
                'total_chunks': len(chunks),
                'text_chunks': len(text_chunks),
                'table_chunks': len(table_chunks),
                'avg_chunk_length': int(avg_length),
                'min_chunk_length': min(c['length'] for c in text_chunks) if text_chunks else 0,
                'max_chunk_length': max(c['length'] for c in text_chunks) if text_chunks else 0
            }
            all_stats.append(stats)
        
        # Save summary
        summary_file = chunks_folder / "chunking_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': 'semantic_recursive',
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'total_pdfs': len(text_files),
                'total_chunks': sum(s['total_chunks'] for s in all_stats),
                'total_text_chunks': sum(s['text_chunks'] for s in all_stats),
                'total_table_chunks': sum(s['table_chunks'] for s in all_stats),
                'statistics': all_stats
            }, f, indent=2)
        
        print("\n" + "=" * 80)
        print("Chunking Summary:")
        print(f"  Total PDFs processed: {len(text_files)}")
        print(f"  Total chunks created: {sum(s['total_chunks'] for s in all_stats)}")
        print(f"    - Text chunks: {sum(s['text_chunks'] for s in all_stats)}")
        print(f"    - Table chunks: {sum(s['table_chunks'] for s in all_stats)}")
        print(f"  Average chunks per PDF: {sum(s['total_chunks'] for s in all_stats) / len(all_stats):.1f}")
        print(f"  Summary saved to: {summary_file.name}")
        print("=" * 80)


def run_semantic_chunking():
    """
    Run semantic chunking on all extracted texts with fixed parameters
    """
    pipeline = SemanticChunkingPipeline()
    pipeline.process_all_texts()


if __name__ == "__main__":
    print("=" * 80)
    print("Semantic Chunking for RAG")
    print("Using LangChain's RecursiveCharacterTextSplitter")
    print("=" * 80)
    print()
    
    run_semantic_chunking()
    
    print("\n" + "=" * 80)
    print("✓ Semantic chunking complete!")
    print("=" * 80)
    print("\nNext steps for RAG:")
    print("  1. Generate embeddings for each chunk")
    print("  2. Store embeddings in vector database (ChromaDB, Pinecone, etc.)")
    print("  3. Build retrieval and generation pipeline")
    print("=" * 80)
