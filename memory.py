from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
import re
from dotenv import load_dotenv
import os

# Load environment variables (.env)
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Function to extract variable assignments from memory text
def extract_vars_from_memory(text):
    pattern = r"(\b[a-zA-Z_]\w*)\s*=\s*(-?\d+\.?\d*)"
    matches = re.findall(pattern, text)
    vars_dict = {}
    for var, val in matches:
        if '.' in val:
            vars_dict[var] = float(val)
        else:
            vars_dict[var] = int(val)
    return vars_dict

# Extract variables from entire memory
def extract_vars_from_memory_full(memory):
    full_text = " ".join([msg.content for msg in memory.chat_memory.messages])
    return extract_vars_from_memory(full_text)

# Calculator tool function evaluates expression with current variables
def calculator_eval(expr, variables):
    if '=' in expr and ',' in expr:
        return "Invalid expression (contains variable assignments). Please provide an expression."
    try:
        return str(eval(expr, {}, variables))
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="gpt-3.5-turbo"
)

# Calculator tool
calc_tool = Tool(
    name="Calculator",
    func=lambda expr: calculator_eval(expr, extract_vars_from_memory_full(memory)),
    description="Evaluate math expressions using existing variables. Provide expressions only, no variable assignments."
)

# Explanation prompt template
explanation_prompt = PromptTemplate(
    input_variables=["expression", "variables"],
    template="""
You are a helpful math tutor.

Given the expression:
{expression}

and the current variable values:
{variables}
you have use chat history.
Explain step-by-step how to compute the expression using these variables.
Provide a detailed explanation in simple language.
Response output as the explation all you observe.
final answer should be the explanation not only result.
"""
)

# LLMChain for explanation
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

# ExplainTool function calls the chain with expression and vars from memory
def explain_expression(expr):
    vars_dict = extract_vars_from_memory_full(memory)
    vars_str = ", ".join([f"{k}={v}" for k, v in vars_dict.items()]) or "No variables found."
    explanation = explanation_chain.run(expression=expr, variables=vars_str)
    return explanation

# ExplainTool
explain_tool = Tool(
    name="ExplainTool",
    func=explain_expression,
    description="Generate detailed step-by-step explanation for math expressions using current variable values."
)

# Initialize agent with tools and memory
agent = initialize_agent(
    tools=[calc_tool, explain_tool],
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
)

print("Agent ready! Type your input below. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.run(user_input)
    print("Agent:", response)
