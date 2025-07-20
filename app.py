from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# ADD THIS IMPORT
from fastapi.middleware.cors import CORSMiddleware
from efaq_backend.faq_core import get_chain
import uvicorn

app = FastAPI()

# --- ADD THIS ENTIRE BLOCK ---
# This tells your Python server to allow requests from your website.
# Make sure the address of your e-commerce site is in this list.
origins = [
    "http://localhost",
    "http://127.0.0.1:5500", # A common address for VS Code Live Server
    "http://localhost:3000", # A common address for React dev servers
    # Add the actual origin of your frontend if it's different
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # This is important: it allows POST, GET, and OPTIONS
    allow_headers=["*"], # Allows all headers, like "Content-Type"
)
# --- END OF BLOCK TO ADD ---

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templetes")

qa_chain = get_chain()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

greetings = ["hi", "hello", "hey", "good morning", "good evening"]

def detect_intent(query: str):
    if query.lower() in greetings:
        return "greeting"
    return "faq"

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("question", "").strip()
    
    intent = detect_intent(query)
    if intent == "greeting":
        answer = "Hello! How can I assist you today?"
    else:
        # Assuming qa_chain.invoke returns a dictionary with an 'answer' key
        result = qa_chain.invoke({"question": query})
        answer = result.get("answer", "Sorry, I couldn't find an answer.")

    return JSONResponse({"answer": answer})
import os
if __name__ == "__main__":
    # Get the PORT from Render's environment variable, default to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    # Listen on 0.0.0.0 to accept connections from outside localhost (Render's internal network)
    uvicorn.run(app, host="0.0.0.0", port=port)