# Medicare Coverage Intelligence Agent

A RAG pipeline that answers Medicare coverage policy questions by retrieving from CMS National Coverage Determinations (NCDs) and Local Coverage Determinations (LCDs).

## Overview

This agent lets users ask natural-language questions about Medicare coverage policies and receive grounded, cited answers drawn from official CMS documents.

## Tech Stack

- **Python 3.11+**
- **LangChain** — orchestration and RAG pipeline
- **ChromaDB** — local vector store for document embeddings
- **OpenAI** — `text-embedding-3-small` for embeddings, `GPT-4o` for generation
- **Streamlit** — web UI
- **RAGAS** — RAG evaluation (faithfulness, answer relevancy)

## Project Structure

```
src/ingestion/   CMS data fetching and parsing
src/rag/         Embedding, indexing, and retrieval pipeline
src/evaluation/  Faithfulness judge and golden dataset
src/ui/          Streamlit app
data/            Raw JSON fetched from CMS
logs/            JSONL interaction logs
tests/           Unit tests
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

3. Ingest CMS data:
   ```bash
   python -m src.ingestion.fetch
   ```

4. Launch the UI:
   ```bash
   streamlit run src/ui/app.py
   ```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and generation |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional, for evaluation judges) |
