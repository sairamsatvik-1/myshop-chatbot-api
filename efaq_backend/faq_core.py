from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)

load_dotenv()

def get_chain():
    """
    Initialize the chat chain with memory-efficient settings
    using OpenRouter embeddings + FAISS (rebuilt at startup).
    """

    # 1. Load documents (your FAQ / policy PDF)
    logger.info("Loading documents...")
    # 👉 change this path to your actual PDF
    pdf_path = "docs/myshop-faq.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks
    logger.info("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # 3. Embeddings (OpenRouter via OpenAIEmbeddings)
    logger.info("Loading embeddings model...")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="text-embedding-3-small",  # keep this consistent
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise

    # 4. Build FAISS index in memory (NO load_local)
    logger.info("Building FAISS vector store...")
    try:
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise

    # 5. LLM via OpenRouter
    logger.info("Initializing LLM...")
    try:
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="mistralai/mistral-7b-instruct:free",
            temperature=0,
            request_timeout=30,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    # 6. Bounded memory (last 5 messages)
    logger.info("Setting up memory management...")
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        human_prefix="User",
        ai_prefix="Assistant",
    )

    # 7. ConversationalRetrievalChain
    logger.info("Creating retrieval chain...")
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False,
        )
        logger.info("Chat chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"Failed to create chain: {e}")
        raise
