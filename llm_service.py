# llm_service.py
import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration ---
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY") # Set the API key for the google-genai library
MODEL_EMBEDDING = 'all-MiniLM-L6-v2'
MODEL_LLM = 'gemini-2.5-flash'

# Initialize Gemini Client and Sentence Transformer
try:
    client = genai.Client()
    embedder = SentenceTransformer(MODEL_EMBEDDING)
except Exception as e:
    print(f"Error initializing client/embedder: {e}")
    client = None
    embedder = None

# Global variables for the Vector Store
vector_store = None
doc_texts = []
FAISS_INDEX_PATH = "document_index.faiss"

def create_vector_store(file_path: str):
    """
    Loads a PDF, splits it into chunks, and creates a FAISS vector store.
    """
    global vector_store, doc_texts

    # 1. Load the PDF Document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split Document into Chunks (for RAG)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Extract the text content
    doc_texts = [chunk.page_content for chunk in chunks]

    # 3. Create Embeddings for all chunks
    # Note: SentenceTransformer is used here for local embeddings (faster/free)
    embeddings = embedder.encode(doc_texts)

    # 4. Create FAISS Index (Vector Database)
    dimension = embeddings.shape[1]
    # FAISS uses IndexFlatL2 for simple Euclidean distance search
    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(np.array(embeddings).astype('float32'))

    # Save the index to disk (for persistence)
    faiss.write_index(vector_store, FAISS_INDEX_PATH)
    
    return len(doc_texts)

def retrieve_context(query: str, k: int = 3) -> str:
    """
    Performs a vector search against the FAISS index to find relevant text chunks.
    """
    global vector_store, doc_texts

    if vector_store is None:
        return "Error: Document index not loaded. Please upload a PDF first."

    # 1. Create embedding for the user query
    query_embedding = embedder.encode([query])

    # 2. Perform similarity search (D=distances, I=indices)
    D, I = vector_store.search(np.array(query_embedding).astype('float32'), k)

    # 3. Retrieve the top K text chunks
    retrieved_texts = [doc_texts[i] for i in I[0]]
    
    # 4. Combine retrieved texts into a single context string
    context = "\n---\n".join(retrieved_texts)
    
    return context

def generate_rag_response(query: str) -> str:
    """
    Generates a response using the retrieved context and the Gemini API.
    """
    if client is None:
        return "Error: AI client not initialized. Check API key."

    # 1. Retrieve the relevant context chunks from the vector store
    context = retrieve_context(query)
    
    if context.startswith("Error"):
        return context

    # 2. Create the RAG prompt (Injecting context into the query)
    system_instruction = (
        "You are an expert document understanding assistant. Use ONLY the provided context to answer the user's question. "
        "If the answer is not found in the context, state 'I cannot find the answer in the document.' Do not use external knowledge."
    )
    
    full_prompt = (
        f"Context from the document:\n{context}\n\n"
        f"User Question: {query}"
    )

    # 3. Call the Gemini API
    try:
        response = client.models.generate_content(
            model=MODEL_LLM,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        return f"Gemini API Call Error: {e}"

# --- Initial Load Check ---
# Attempt to load the index and texts if they exist from a previous session
try:
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = faiss.read_index(FAISS_INDEX_PATH)
        # In a real app, you would load doc_texts from a separate file.
        # For simplicity here, assume doc_texts is stored/recreated with the index.
        # For this minimal version, the index is created fresh on every app start 
        # or when a new file is uploaded.
        pass
except Exception as e:
    print(f"Could not load FAISS index: {e}")