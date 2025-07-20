# tools_agent.py

import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Get API key from .env
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize ChatOpenAI with OpenRouter endpoint
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="gpt-3.5-turbo"   # you can also use "gpt-4o" if your OpenRouter key supports it
)

# --- Define our tool ---

# Calculator Tool
def calculator_tool(input_text: str) -> str:
    try:
        result = eval(input_text)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Wrap in Tool
calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Use this tool to calculate math expressions. Input should be a math expression."
)

# Optional → Python REPL Tool → Can run Python code

# --- Initialize Agent ---

tools = [calculator]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# --- Run Agent ---

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.run(user_input)
    print(f"🤖 Agent: {response}")
