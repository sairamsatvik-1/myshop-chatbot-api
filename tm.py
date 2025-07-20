from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

# Load your .env file
load_dotenv()
import os
# Get API key from .env
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize ChatOpenAI with OpenRouter endpoint
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="gpt-3.5-turbo"   # you can also use "gpt-4o" if your OpenRouter key supports it
)

# 🛠️ 2. Define tools
def calculator(input: str) -> str:
    try:
        return str(eval(input))
    except:
        return "Invalid math expression."

def dummy_search(input: str) -> str:
    return f"Sorry, I can't browse the web yet. You asked: '{input}'"

tools = [
    Tool(name="Calculator", func=calculator, description="Use for math expressions like '12 * 8'"),
    Tool(name="WebSearch", func=dummy_search, description="Use for questions about current events or unknown info")
]

# 🧠 3. Memory setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🤖 4. Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# 🧪 5. Run agent
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.run(query)
    print(f"Agent: {response}")
