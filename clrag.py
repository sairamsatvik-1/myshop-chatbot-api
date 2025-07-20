from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# === Load .env for OpenRouter Key ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# === Load Embeddings + FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# === Load LLM from OpenRouter ===
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model="gpt-3.5-turbo"
)

# === Setup Memory for Multi-turn Conversation ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# === Setup Conversational Retrieval Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# === Ask Questions in Loop ===
print("💬 Ask your questions (type 'exit' to quit)")
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print(f"🤖 Bot: {answer}")
