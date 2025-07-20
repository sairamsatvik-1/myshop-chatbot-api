import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ Step 1: Define Paths
DATA_FILE = r"C:\Users\kanum\OneDrive\Documents\EFAQ.txt"

VECTOR_STORE_DIR = "vectorstore/"

# ✅ Step 2: Load the Text File
def load_text_file(filepath):
    loader = TextLoader(filepath, encoding='utf-8')

    return loader.load()

# ✅ Step 3: Chunk the Content
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# ✅ Step 4: Embed & Save to FAISS
def create_and_save_vectorstore(chunks, save_dir):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_dir)
    print(f"✅ Vector store saved to: {save_dir}")

# ✅ Main Function
if __name__ == "__main__":
    docs = load_text_file(DATA_FILE)
    chunks = split_documents(docs)
    create_and_save_vectorstore(chunks, VECTOR_STORE_DIR)
