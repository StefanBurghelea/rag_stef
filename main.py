from fastapi import FastAPI
import uvicorn
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

# Import the QA chain and cached embeddings from your qa.py file
from app.qa import get_qa_chain, get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL") or "http://localhost:3000"],  # Frontend URL from environment variable
    allow_credentials=True,
    allow_methods=["*"],  # Or specify methods like ["GET", "POST"]
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def ingest_docs():
    # Check if vectorstore already exists
    if Path("vectorstore").exists():        
        embeddings = get_embeddings()  # Use cached embeddings
        vectordb = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
        try:
            count = vectordb._collection.count()
            if count > 0:
                return
            else:
                logger.info("Vectorstore exists but is empty, recreating...")
        except Exception as e:
            logger.error(f"Error checking vectorstore: {e}")
    
    logger.info("Creating vectorstore")
    docs = []
    
    # Load all markdown files from docs directory
    docs_path = Path("docs")
    if not docs_path.exists():
        logger.error("docs/ directory not found!")
        return
        
    markdown_files = list(docs_path.glob("*.md"))
    logger.info(f"Found {len(markdown_files)} markdown files: {[f.name for f in markdown_files]}")
    
    for file in markdown_files:
        try:
            loader = TextLoader(str(file), encoding='utf-8')
            file_docs = loader.load()
            logger.info(f"Loaded {len(file_docs)} documents from {file.name}")
            docs.extend(file_docs)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not docs:
        logger.error("No documents loaded!")
        return

    logger.info(f"Total documents loaded: {len(docs)}")
    
    # Split documents with better chunk size for the content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=200,  # More overlap for better continuity
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    logger.info(f"Split into {len(split_docs)} chunks")

    # Create vectorstore
    embeddings = get_embeddings()  # Use cached embeddings
    vectordb = Chroma.from_documents(
        split_docs, 
        embeddings, 
        persist_directory="vectorstore"
    )
    vectordb.persist()
    logger.info(f"Vectorstore created and persisted with {len(split_docs)} document chunks")

@app.post("/ingest")
def ingest():
    try:
        ingest_docs()
        return {"message": "Docs ingested and embeddings saved."}
    except Exception as e:
        logger.error(f"Error in ingest: {e}")
        return {"error": str(e)}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Ask a question about Stefan's profile"""
    try:
        qa_chain = get_qa_chain()  # Now using cached chain
        result = qa_chain({"query": request.question})
        
        # Log for debugging
        logger.info(f"Question: {request.question}")
        logger.info(f"Answer: {result['result']}")
        
        return {
            "question": request.question,
            "answer": result["result"]
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to Stefan's RAG API - Ask questions about Stefan!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
if __name__ == "__main__":
    # Production-ready configuration
    port = int(os.getenv("PORT", 8000))  # Railway sets PORT automatically
    host = os.getenv("HOST", "0.0.0.0")
    
    # Initialize documents on startup
    ingest_docs()
    
    # Start server with production settings
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )