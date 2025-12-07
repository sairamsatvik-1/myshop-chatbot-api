from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import gc
import logging
from contextlib import asynccontextmanager
from efaq_backend.faq_core import get_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global chain instance (loaded once) ---
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global qa_chain
    logger.info("Loading chatbot model...")
    qa_chain = get_chain()
    logger.info("Chatbot model loaded successfully")
    yield
    logger.info("Shutting down chatbot...")
    qa_chain = None
    gc.collect()

app = FastAPI(lifespan=lifespan)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:8080",  # Common development port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

greetings = ["hi", "hello", "hey", "good morning", "good evening", "hey there"]

def detect_intent(query: str):
    """Detect greeting vs FAQ intent"""
    if query.lower().strip() in greetings:
        return "greeting"
    return "faq"

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint - optimized for memory efficiency"""
    try:
        data = await request.json()
        query = data.get("question", "").strip()

        if not query:
            return JSONResponse({"answer": "Please provide a question."}, status_code=400)

        # Limit query length to prevent memory issues
        if len(query) > 1000:
            query = query[:1000]

        intent = detect_intent(query)
        
        if intent == "greeting":
            answer = "Hello! How can I assist you today?"
        else:
            # Invoke chain with query
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")

        # Periodic garbage collection
        gc.collect()

        return JSONResponse({"answer": answer})

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        return JSONResponse(
            {"answer": "An error occurred. Please try again later."},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker to save memory
        loop="uvloop"  # Faster event loop
    )
