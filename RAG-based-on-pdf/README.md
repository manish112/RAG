# RAG PDF Processing Pipeline

A simple pipeline for processing PDFs for RAG (Retrieval-Augmented Generation) applications using **agentic chunking** with HuggingFace Transformers.

## Features

- **PDF Download**: Automatically downloads academic papers from arxiv
- **Text Extraction**: Extracts text and tables from PDFs using multiple methods
- **Agentic Chunking**: Uses HuggingFace LLM to intelligently chunk text at semantic boundaries

## Prerequisites

**Python 3.13+** with virtual environment

## Setup

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

That's it! No external services needed. The model will download automatically on first run.

## Usage

### Run Complete Pipeline

```bash
python run.py
```

The pipeline will:
1. Download 5 academic papers
2. Extract text and tables
3. Download Qwen 1.5B model (first run only, ~3GB)
4. Chunk using AI (~1500 char chunks)

### Run Individual Steps

```bash
# 1. Download PDFs only
python -c "import pdf_ingest; pdf_ingest.download_all_pdfs()"

# 2. Extract text only
python -c "import pdf_ingest; pdf_ingest.extract_all_pdfs()"

# 3. Chunk text only
python chunking.py
```

## Configuration

Edit `chunking.py` to change:
- **MODEL_NAME**: Default is `"Qwen/Qwen2.5-1.5B-Instruct"` (line 16)
- **MAX_CHUNK_SIZE**: Default is `1500` characters (line 17)

## Output Structure

```
data/
├── *.pdf                          # Downloaded PDFs
├── extracted_text/
│   └── *_full_text.txt           # Extracted text with tables
└── chunks/
    └── *_chunks.txt              # Chunked text (plain text)
```

## How It Works

1. **LLM Analysis**: HuggingFace model analyzes text to identify natural semantic boundaries
2. **Smart Splitting**: Creates chunks at identified topic boundaries
3. **Variable Size**: Chunk size varies based on semantic coherence (target ~1500 chars)
4. **Context Preservation**: Keeps related content together (tables with descriptions, complete sections)
5. **Automatic Fallback**: If LLM fails, uses paragraph-based chunking

## First Run

On first run, the model (~3GB) will be downloaded automatically:
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Size: ~3GB
- Location: ~/.cache/huggingface/
- Time: 5-15 minutes (depending on internet speed)

Subsequent runs will use the cached model (no download needed).

## Performance

- **Download**: ~10 seconds for 5 papers
- **Extraction**: ~5 seconds for 5 papers  
- **First Run Chunking**: ~5-15 minutes (model download + inference)
- **Subsequent Runs**: ~2-5 minutes (inference only)

## Hardware Requirements

- **CPU**: Works on any modern CPU
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB free (3GB model + data)
- **GPU**: Optional (CUDA will be used if available for faster inference)

## Papers Included

1. Attention Is All You Need (Transformer)
2. BERT: Pre-training of Deep Bidirectional Transformers
3. Language Models are Few-Shot Learners (GPT-3)
4. RoBERTa: A Robustly Optimized BERT Pretraining Approach
5. Exploring the Limits of Transfer Learning (T5)

## Next Steps

After running the pipeline:

1. **Generate Embeddings**: Create vector embeddings from chunks
2. **Vector Database**: Store chunks in vector database (Pinecone, ChromaDB, etc.)
3. **RAG Pipeline**: Build retrieval and generation pipeline

## Troubleshooting

### Slow First Run
- Normal - model is downloading (~3GB)
- Check: ~/.cache/huggingface/ folder

### Out of Memory
- Close other applications
- Reduce MAX_CHUNK_SIZE in chunking.py
- Use CPU-only mode (model will automatically detect)

### Model Download Fails
- Check internet connection
- Clear cache: rm -rf ~/.cache/huggingface/
- Try again

## License

MIT
