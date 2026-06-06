# Medicare Coverage Intelligence Agent

A RAG pipeline that answers Medicare coverage policy questions by retrieving from CMS National Coverage Determinations (NCDs) and Local Coverage Determinations (LCDs).

## Tech Stack

- **Python 3.11+**
- **LangChain** — orchestration and RAG pipeline
- **ChromaDB** — local vector store
- **HuggingFace** — `BAAI/bge-small-en-v1.5` for embeddings
- **Groq** — `llama-3.1-8b-instant` for generation and evaluation
- **Streamlit** — web UI
- **RAGAS** — evaluation (faithfulness, answer relevancy)

## Project Structure

```
src/ingestion/   CMS data fetching and parsing
src/rag/         Embedding, indexing, and retrieval pipeline
src/evaluation/  Golden dataset generation and RAGAS evaluation
src/ui/          Streamlit app
data/            Raw JSON fetched from CMS (gitignored)
logs/            JSONL interaction logs (gitignored)
tests/           Unit tests
```

## Setup

1. Create and activate the virtual environment:
   ```bash
   python -m venv agent
   agent\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API key:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

4. Ingest CMS data:
   ```bash
   python -m src.ingestion.fetch
   ```

5. Generate the golden evaluation dataset (200 samples):
   ```bash
   python -m src.evaluation.generate_golden
   ```

6. Launch the UI:
   ```bash
   streamlit run src/ui/app.py
   ```

## Evaluation

Run RAGAS faithfulness and answer relevancy scoring against the golden dataset:

```bash
python -m src.evaluation.judge
```

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for generation and evaluation |
