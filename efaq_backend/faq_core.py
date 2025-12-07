from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
import logging
import warnings

# Suppress any remaining deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)

load_dotenv()

def get_chain():
    """
    Initialize the chat chain with memory-efficient settings

    Key optimizations:
    1. ConversationBufferWindowMemory - Only keeps last N messages (bounded)      
    2. Lightweight embeddings model - all-MiniLM-L6-v2 (33MB)
    3. Single-worker uvicorn - Reduces total memory footprint
    """

    logger.info("Loading embeddings model...")
    try:
        # Lightweight embeddings model (~33MB, vs 400MB+ for larger models)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # Cache model to disk to avoid redownloading
            cache_folder="./models"
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise    logger.info("Loading FAISS vector store...")
    try:
        # Load FAISS vector database
        db = FAISS.load_local(
            "vectorstore/",
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever()
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise

    logger.info("Initializing LLM...")
    try:
        # Initialize LLM with OpenRouter (free tier available)
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="mistralai/mistral-7b-instruct:free",
            temperature=0,
            request_timeout=30,  # Prevent hanging requests
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    logger.info("Setting up memory management...")
    # IMPORTANT: Use ConversationBufferWindowMemory instead of ConversationBufferMemory
    # This keeps only the last K messages, preventing unbounded memory growth     
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,  # Keep only last 5 messages - adjust based on your needs
        human_prefix="User",
        ai_prefix="Assistant"
    )

    logger.info("Creating retrieval chain...")
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False  # Set to False in production to reduce logging overhead    
        )
        logger.info("Chat chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"Failed to create chain: {e}")
        raise
