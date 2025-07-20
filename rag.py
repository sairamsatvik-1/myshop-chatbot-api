from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

texts = [
    "Python is a popular programming language created by Guido van Rossum.",
    "The capital of France is Paris.",
    "LangChain is a framework for building LLM-powered applications.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors."
]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class CustomHuggingFaceEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text)

    def __call__(self, text):
        return self.embed_query(text)

embedding_model = CustomHuggingFaceEmbeddings(embed_model)

vectorstore = FAISS.from_texts(texts, embedding_model)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    model="gpt-3.5-turbo"
)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What is Python?"
answer = qa_chain.invoke(query)  # Use invoke() instead of run()
print(answer)
