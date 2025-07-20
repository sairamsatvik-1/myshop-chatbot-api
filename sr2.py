import os
from dotenv import load_dotenv

# 🔐 Load environment variables
load_dotenv()

# 🧠 Imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper

from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# 🧠 Load embeddings & FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 🔍 Retrieval QA Chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="gpt-3.5-turbo"
    ),
    retriever=retriever
)

# 🔎 Web Search Tool (SerpAPI)
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

# 🧰 Smart Fallback Tool function
def smart_tool_fallback(query: str) -> str:
    try:
        vector_answer = qa_chain.run(query)
        # Basic heuristic: if answer is meaningful, return it
        if vector_answer and len(vector_answer.strip()) > 10 and "I don't know" not in vector_answer:
            return f"(Vector Search) {vector_answer}"
        else:
            web_answer = search.run(query)
            return f"(Web Search) {web_answer}"
    except Exception as e:
        return f"Error during search: {str(e)}"

tools = [
    Tool(
        name="Smart Search",
        func=smart_tool_fallback,
        description="Answers questions from PDF knowledge base or falls back to Web Search if PDF info unavailable."
    )
]

# 🧠 Memory for agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🤖 Agent Setup
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="gpt-3.5-turbo"
    ),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 🎯 Chat Loop
print("💬 Ask your questions (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent.run(query)
    print(f"\n🤖 Bot: {response}\n")
