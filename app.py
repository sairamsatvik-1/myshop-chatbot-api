
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
    "https://ecommerce-project-wago.onrender.com/"
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

        # Simple greeting shortcut
        if question.lower() in ["hi", "hello", "hey"]:
            return {"answer": "Hello! How can I assist you today?"}

        # ✅ For langchain 0.3.x, prefer direct call instead of .invoke
        result = qa_chain({"question": question})

        # Log once on server to understand structure (only visible in Render logs)
        logger.info(f"Raw QA chain result: {result!r}")

        raw_answer = ""

        if isinstance(result, dict):
            # Try common keys in order
            for key in ["answer", "result", "output_text"]:
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    raw_answer = val
                    break

        # Fallback: if still empty, just dump the whole result
        if not raw_answer.strip():
            raw_answer = str(result)

        answer = clean_answer(raw_answer)

        # Final fallback if even after cleaning it's empty
        if not answer.strip():
            answer = "I couldn't find a clear answer in my FAQ for that. Please contact MyShop support."

        gc.collect()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(
            {"answer": "Sorry, something went wrong on the server."},
            status_code=500,
        )



@app.get("/health")
async def health():
    return {"status": "ok"}
