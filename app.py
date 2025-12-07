from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import gc
from efaq_backend.faq_core import get_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

qa_chain = get_chain()

ALLOWED_ORIGINS = [
    "https://myshop-chatbot-ui.onrender.com",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://ecommerce-project-wago.onrender.com/",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_answer(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for token in ["<s>", "</s>", "[OUT]", "[/OUT]"]:
        text = text.replace(token, "")
    return text.strip()


@app.get("/")
async def home():
    return {"message": "MyShop Chatbot API is running 🚀"}


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = (data.get("question") or "").strip()

        if not question:
            return JSONResponse({"answer": "Ask me something!"}, status_code=400)

        if question.lower() in ["hi", "hello", "hey"]:
            return {"answer": "Hello! How can I assist you today?"}

        raw_answer = qa_chain.invoke(question)
        answer = clean_answer(raw_answer)

        gc.collect()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse({"answer": "Sorry, something went wrong."}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}
