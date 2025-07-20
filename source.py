from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI  # ✅ updated import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # ✅ updated memory import
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✅ Updated Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Load FAISS vectorstore
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# ✅ Updated LLM from openrouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model="gpt-3.5-turbo"
)

# ✅ Create memory (set return key explicitly)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # 🔥 this solves the issue
)

# ✅ Build the RAG chain with source docs and memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
    verbose=True,
    output_key="answer"  # 🔥 must match what memory expects
)

# ✅ Chat loop
print("💬 Ask your questions (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    response = qa_chain.invoke({"question": query})  # ✅ no deprecation warning
    print(f"\n🤖 Bot: {response['answer']}")

    print("\n📄 Source Documents:")
    for i, doc in enumerate(response["source_documents"], 1):
        print(f"\n[{i}] {doc.page_content[:300]}...")
        if doc.metadata:
            print(f"    🔗 Metadata: {doc.metadata}")
