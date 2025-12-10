# ISI Markets ML Engineer Task

This is interview task given for ML Engineer position.

## Installation

```bash
pip install -r /path/to/requirements.txt
```

## Requirements

This project requires the following:
- Python 3.8+
- Pinecone account and API key
- Gemini API key

The api keys are set via environment variables. You can create a `.env` file in the root directory as per env-example.

## Steps
1. **Select PDF Documents**: Place your PDF documents in the `data/` directory that you wish to embedd into Pinecone.

2. **Indexing Documents**: Run the indexing script to process PDF documents and create a Pinecone index from indexing.py (you can use this command from the root directory: python -m backend.indexing).

3. **Ask Questions**: Start a server from the main.py file and use the root endpoint to ask questions into the chatbot.


