from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load your API keys
load_dotenv()

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# LLM setup
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    model="gpt-3.5-turbo"
)

# Tool setup
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events or factual information."
    )
]

# 🧠 Memory setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent setup
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # ✅ Correct agent type with memory
    memory=memory,
    verbose=True
)

# Start chatting
print(agent.run("Hello, who won IPL 2025?"))
print(agent.run("What team did they defeat?"))
