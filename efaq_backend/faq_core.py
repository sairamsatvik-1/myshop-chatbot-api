# efaq_backend/faq_core.py

import os
import logging
import warnings

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)

load_dotenv()


class MyShopQA:
    """Very simple QA helper using retriever + LLM, no fancy chains."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def __call__(self, question: str) -> str:
        # 1) Retrieve relevant chunks
        docs = self.retriever.get_relevant_documents(question)
        if not docs:
            return "I couldn't find any relevant information in my FAQ for that."

        # 2) Build context from top documents
        context = "\n\n".join(d.page_content for d in docs[:4])

        # 3) Build prompt manually
        prompt = f"""You are MyShop's helpful FAQ assistant.

Use ONLY the information in the context below to answer the customer's question.
If the answer is not clearly in the context, say you don't know and suggest
contacting MyShop customer support.

Context:
{context}

Question: {question}

Give a concise answer in 2–5 sentences."""

        # 4) Call LLM
        response = self.llm.invoke(prompt)

        # 5) Extract text content
        try:
            text = response.content
        except AttributeError:
            text = str(response)

        # Clean up weird wrapper tokens if any
        for token in ["<s>", "</s>", "[OUT]", "[/OUT]"]:
            text = text.replace(token, "")

        return text.strip() or "I couldn't find a clear answer in my FAQ for that."


def get_chain():
    """
    Build a simple QA helper using:
      - OpenRouter embeddings (text-embedding-3-small)
      - Existing FAISS vectorstore/
      - Mistral-7B via OpenRouter
    """

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")

    logger.info("Loading embeddings model...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
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
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct:free",
        temperature=0,
        request_timeout=60,
    )

    logger.info("Creating MyShopQA helper...")
    qa = MyShopQA(retriever, llm)
    logger.info("MyShopQA ready.")
    return qa
