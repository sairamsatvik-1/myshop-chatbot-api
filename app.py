# app.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import logging
import gc

from efaq_backend.faq_core import get_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Initializing QA helper...")
qa_chain = get_chain()
logger.info("QA helper ready.")

ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://myshop-chatbot-ui.onrender.com",
    "*",  # keep during dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "MyShop Chatbot API is running 🚀"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = (data.get("question") or "").strip()

        if not question:
            return JSONResponse({"answer": "Ask me something!"}, status_code=400)

        # Greeting shortcut
        if question.lower() in ["hi", "hello", "hey"]:
            return {"answer": "Hello! How can I assist you today?"}

        # ✅ Call our simple QA helper like a function
        answer = qa_chain(question)
        logger.info(f"Answer: {answer!r}")

        gc.collect()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(
            {"answer": f"Server error: {type(e).__name__}: {e}"},
            status_code=500,
        )
