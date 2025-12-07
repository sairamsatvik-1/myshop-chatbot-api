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
    logger.info("Loading embeddings model...")
    embeddings = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="text-embedding-3-small",
    )

    logger.info("Loading FAISS vector store...")
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever()

    logger.info("Initializing LLM...")
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        temperature=0,
        request_timeout=30,
    )

    logger.info("Setting up memory management...")
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        human_prefix="User",
        ai_prefix="Assistant",
    )

    logger.info("Creating retrieval chain...")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
    )
    logger.info("Chat chain created successfully")
    return chain
