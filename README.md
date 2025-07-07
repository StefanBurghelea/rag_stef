# RAG Project

A Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, and ChromaDB.

## Features

- FastAPI web framework for building APIs
- LangChain for LLM integration and RAG workflows
- ChromaDB for vector storage and similarity search
- OpenAI integration for language models
- Uvicorn ASGI server for production deployment

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd rag_port
   ```

2. Activate your virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **LangChain**: Framework for developing applications with LLMs
- **OpenAI**: OpenAI API client
- **ChromaDB**: Vector database for embeddings and similarity search

## Development

To add new features:

1. Implement your RAG logic using LangChain and ChromaDB
2. Add new API endpoints in `main.py`
3. Test your changes using the interactive API docs at `/docs`

## Environment Variables

Consider setting up environment variables for:
- `OPENAI_API_KEY`: Your OpenAI API key
- `CHROMA_DB_PATH`: Path to your ChromaDB storage (optional)

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
``` 