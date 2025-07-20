from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "vectorstore/"
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore/", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        temperature=0,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    return chain
