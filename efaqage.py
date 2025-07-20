from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

VECTOR_STORE_DIR = "vectorstore/"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()

def init_chat_model():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct:free",
        temperature=0,
    )

def run_qa():
    retriever = load_vectorstore()
    llm = init_chat_model()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    print("Ask your question (type 'exit' to quit):")
    while True:
        query = input("Q: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye, master!")
            break
        # Use invoke() with input dict and extract result key
        answer = qa_chain.invoke({"query": query})["result"]
        print("A:", answer)
        print("-" * 40)

if __name__ == "__main__":
    run_qa()
