from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatOpenAI  # community version
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o"
)

# Simple tool (not structured)
def calculate_area(input_text: str) -> str:
    """Calculate area of a rectangle. Input format: 'width height'."""
    try:
        width, height = map(float, input_text.split())
        area = width * height
        return f"Area is {area} square units."
    except:
        return "Invalid input format. Provide two numbers like '10 20'."

tools = [
    Tool(
        name="RectangleAreaTool",
        func=calculate_area,
        description="Use to calculate rectangle area. Input two numbers: width and height."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# CLI loop
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        break
    response = agent.run(query)
    print(f"Agent: {response}")
