from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load your API keys from .env
load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Setup embedding model (same as used for FAISS index creation)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the saved FAISS vectorstore index from disk
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Initialize the LLM via OpenRouter GPT-3.5 turbo
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    model="gpt-3.5-turbo"
)

# Setup SerpAPI tool for live web search
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Useful for answering questions about current events or factual information."
)

# Setup RAG tool that queries your FAISS vectorstore (retriever)
def rag_tool_func(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    # You can customize how to combine these docs; here, just join page contents
    return "\n\n".join([doc.page_content for doc in docs])

rag_tool = Tool(
    name="RAG",
    func=rag_tool_func,
    description="Useful for answering questions based on your custom documents."
)

# Setup memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with both tools and memory
agent = initialize_agent(
    tools=[search_tool, rag_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

print("💬 Ask your questions (type 'exit' to quit)")

while True:
    query = input("You: ")
    if query.strip().lower() == "exit":
        print("Bye master!")
        break
    response = agent.run(query)
    print("\n🤖 Bot:", response, "\n")
