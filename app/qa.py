# app/qa.py

from functools import lru_cache
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def get_embeddings():
    """Get cached embeddings instance"""
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_vectorstore():
    """Get cached vectorstore instance"""
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
    # Check if vectorstore has documents
    try:
        count = vectorstore._collection.count()
        logger.info(f"Vectorstore loaded with {count} documents")
        if count == 0:
            logger.warning("Vectorstore is empty! Make sure to run /ingest first")
    except Exception as e:
        logger.error(f"Error checking vectorstore: {e}")
    return vectorstore

# Improved prompt template specifically for Stefan's profile
prompt_template = PromptTemplate.from_template(
"""You are Stefan Burghelea's AI assistant. Answer questions about Stefan's professional background, skills, experience, and projects based ONLY on the provided context.

Key guidelines:
- Provide specific, accurate information from the context
- If Stefan is open to opportunities, mention his availability for freelance/remote work
- Be professional and concise
- If the question cannot be answered from the context, say "I don't have that information in Stefan's profile"

Context about Stefan:
{context}

Question: {question}

Answer:"""
)

@lru_cache(maxsize=1)
def get_qa_chain():
    """Get cached QA chain"""
    try:
        vectorstore = get_vectorstore()
        llm = ChatOpenAI(
            temperature=0.1, 
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
        )
        
        logger.info("QA chain initialized successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error initializing QA chain: {e}")
        raise
