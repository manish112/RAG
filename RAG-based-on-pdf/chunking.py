# Agentic Chunking - Simple Version
# Flow: Prompt -> Model -> Text File -> Chunks Output

import os
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CHUNK_SIZE = 1500


# Initialize model once at module level
print(f"Loading model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("Model loaded successfully!\n")


# The prompt that tells the model how to chunk
CHUNKING_PROMPT = """You are a text chunking assistant. Split this text into semantic chunks of approximately {chunk_size} characters each.

Rules:
- Keep complete thoughts together
- Respect paragraph boundaries
- Don't split mid-sentence

Text to chunk:
{text}

Chunked output:"""


def chunk_text_with_llm(text: str) -> List[str]:
    # Send text to LLM and get back chunks
    
    prompt = CHUNKING_PROMPT.format(chunk_size=MAX_CHUNK_SIZE, text=text[:6000])
    
    try:
        result = llm(prompt)
        response = result[0]['generated_text'][len(prompt):]
        
        # Split by the separator
        chunks = [c.strip() for c in response.split('---CHUNK---') if c.strip()]
        
        # If model didn't follow format or returned nothing, split by paragraphs
        if not chunks or len(chunks) < 2:
            chunks = simple_paragraph_split(text)
        
        return chunks
    
    except Exception as e:
        print(f"  Model error: {e}")
        print("  Using simple paragraph splitting")
        return simple_paragraph_split(text)


def simple_paragraph_split(text: str) -> List[str]:
    # Fallback: just split by paragraphs into chunks
    
    chunks = []
    current_chunk = ""
    
    for paragraph in text.split('\n\n'):
        if len(current_chunk) + len(paragraph) < MAX_CHUNK_SIZE:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def process_file(text_file: Path, output_folder: Path):
    # Process one text file: read it, chunk it, save chunks
    
    pdf_name = text_file.stem.replace('_full_text', '')
    output_file = output_folder / f"{pdf_name}_chunks.txt"
    
    # Check if chunks already exist
    if output_file.exists():
        print(f"Skipping: {pdf_name} (chunks already exist)")
        return
    
    print(f"Processing: {pdf_name}")
    
    # Read the full text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"  Text length: {len(text):,} characters")
    
    # Chunk it
    chunks = chunk_text_with_llm(text)
    
    print(f"  Created {len(chunks)} chunks")
    
    # Save chunks to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(chunk)
            if i < len(chunks) - 1:  # Don't add separator after last chunk
                f.write("\n\n")
    
    print(f"  Saved to: {output_file.name}\n")


def run_chunking():
    # Main function: process all text files
    
    base_dir = Path(__file__).parent
    extracted_folder = base_dir / 'data' / 'extracted_text'
    chunks_folder = base_dir / 'data' / 'chunks'
    
    # Create output folder
    chunks_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    text_files = sorted(extracted_folder.glob('*_full_text.txt'))
    
    if not text_files:
        print("No text files found!")
        return
    
    print(f"Found {len(text_files)} files to process\n")
    print("=" * 80)
    
    # Process each file
    for text_file in text_files:
        process_file(text_file, chunks_folder)
    
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    run_chunking()
