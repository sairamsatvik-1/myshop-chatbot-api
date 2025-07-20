import os
from io import BytesIO
import requests
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForQuestionAnswering

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load BLIP models once to speed up ---
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

qa_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
qa_model = Blip2ForQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

def load_image(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def generate_caption(img_url: str) -> str:
    image = load_image(img_url)
    inputs = caption_processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

def answer_question(img_url: str, question: str) -> str:
    image = load_image(img_url)
    inputs = qa_processor(image, question, return_tensors="pt").to(device)
    out = qa_model.generate(**inputs)
    answer = qa_processor.decode(out[0], skip_special_tokens=True)
    return answer

# Store current image URL and caption in memory for context
class ImageMemory:
    def __init__(self):
        self.image_url = None
        self.caption = None

image_memory = ImageMemory()

# Tool for generating image caption
def caption_tool(input_text: str) -> str:
    # Expect input_text to be an image URL
    image_memory.image_url = input_text.strip()
    try:
        caption = generate_caption(image_memory.image_url)
        image_memory.caption = caption
        return f"Image caption: {caption}"
    except Exception as e:
        return f"Failed to caption image: {str(e)}"

# Tool for answering questions about the last image captioned
def qa_tool(question: str) -> str:
    if not image_memory.image_url:
        return "No image provided yet. Please provide an image URL first to generate caption."
    try:
        answer = answer_question(image_memory.image_url, question)
        return f"Answer: {answer}"
    except Exception as e:
        return f"Failed to answer question: {str(e)}"

tools = [
    Tool(
        name="Caption Image",
        func=caption_tool,
        description="Use this tool to generate a caption from an image URL. Input should be the image URL."
    ),
    Tool(
        name="Image Question Answering",
        func=qa_tool,
        description="Use this tool to answer questions about the last image captioned. Input should be a question."
    ),
]

# Memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Use any LLM you have access to, e.g. OpenRouter or OpenAI (adjust your keys accordingly)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

print("Agent is ready! You can now provide an image URL or ask questions about the image.")
print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Run the agent on user input
    response = agent.run(user_input)
    print(f"Agent: {response}\n")
