from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
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
    Load FAISS vectorstore (built offline) + OpenRouter LLM.
    No PDF loading on Render. Just reuse the saved index.
    """

    logger.info("Loading embeddings model...")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="text-embedding-3-small",
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise

    logger.info("Loading FAISS vector store...")
    try:
        db = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever()
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise

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

    logger.info("Setting up memory management...")
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        human_prefix="User",
        ai_prefix="Assistant",
    )

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
