\# Medicare Coverage Intelligence Agent



\## What this is

A RAG pipeline that answers Medicare coverage policy questions

by retrieving from CMS NCDs and LCDs.



\## Tech stack

\- Python 3.11+

\- LangChain for orchestration

\- ChromaDB for vector storage

\- OpenAI API (text-embedding-3-small + GPT-4o)

\- Streamlit for UI



\## Project structure

src/ingestion/   - CMS data fetching and parsing

src/rag/         - embedding, indexing, retrieval pipeline

src/evaluation/  - faithfulness judge, golden dataset

src/ui/          - Streamlit app

data/            - raw JSON from CMS

logs/            - JSONL interaction logs

tests/           - unit tests



\## Conventions

\- Type hints on all functions

\- Docstrings on all public functions

\- Environment variables via python-dotenv, never hardcoded keys

\- Errors logged, never silently swallowed



## Environment
Virtual environment name: agent (located at agent\ in project root)
Activate with: agent\Scripts\activate
Always use this venv for all pip installs and Python execution.
Never install packages globally or use pip install --user.