from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Load API Key
api_key = os.getenv("OPENROUTER_API_KEY")

# Setup Chat LLM
llm = ChatOpenAI(
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="gpt-3.5-turbo",
        temperature=0.7,
    )
# Prompt Template
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="understand the sentence change as reqiured : {user_input}",
)

# Modern Chain style
chain = prompt | llm

# User input
user_text = "write a 100 lines paragraph about me."

# Run
response = chain.invoke({"user_input": user_text})

# Output
print("\n🔁 Rewritten Response:\n", response.content)
