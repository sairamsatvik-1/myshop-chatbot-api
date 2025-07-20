import os
from langchain_openai import ChatOpenAI
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

# Test LLM
response = llm.invoke("Hello, Master! This is a test from LangChain with OpenRouter. How are you?")

# Print response
print(response.content)
