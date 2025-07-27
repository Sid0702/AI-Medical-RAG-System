import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import pickle
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv

import config

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Key and Model Initialization ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    logging.info("Successfully configured Gemini API key from .env file.")
else:
    logging.error("GOOGLE_API_KEY not found in .env file. Please create a .env file and add your key.")

try:
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    logging.info(f"Successfully loaded embedding model: {config.EMBEDDING_MODEL_NAME}")
except Exception as e:
    EMBEDDING_MODEL = None
    logging.error(f"Failed to load embedding model: {e}")

# --- Core Pipeline Functions ---

def load_and_chunk_pdf(file_path):
    doc_name = os.path.basename(file_path)
    full_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                full_text += page.get_text() + "\n"
    except Exception as e:
        logging.error(f"Failed to open or read PDF {file_path}: {e}")
        return []

    if not full_text.strip():
        logging.warning(f"No text extracted from PDF: {doc_name}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    text_chunks = text_splitter.split_text(full_text)
    logging.info(f"Split '{doc_name}' into {len(text_chunks)} chunks.")

    return [
        {"content": chunk, "source": f"{doc_name}, Chunk {i+1}"}
        for i, chunk in enumerate(text_chunks)
    ]

def create_faiss_index(chunks, index_path):
    if not EMBEDDING_MODEL or not chunks:
        return

    contents = [chunk['content'] for chunk in chunks]
    embeddings = EMBEDDING_MODEL.embed_documents(contents)
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, f"{index_path}.faiss")
    with open(f"{index_path}.meta", 'wb') as f:
        pickle.dump(chunks, f)

def search_faiss_index(query, index_path, top_k=3):
    if not EMBEDDING_MODEL:
        return []
        
    try:
        query_embedding = EMBEDDING_MODEL.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        index_file = f"{index_path}.faiss"
        if not os.path.exists(index_file):
            return []

        index = faiss.read_index(index_file)
        with open(f"{index_path}.meta", 'rb') as f:
            metadata = pickle.load(f)
            
        _, indices = index.search(query_vector, top_k)
        
        valid_indices = [i for i in indices[0] if i < len(metadata)]
        return [metadata[i] for i in valid_indices]
    except Exception as e:
        logging.error(f"An error occurred during FAISS search on {index_path}: {e}")
        return []

def get_llm_response(query, context):
    try:
        model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        context_str = "\n\n---\n\n".join([f"Source: {item['source']}\nContent: {item['content']}" for item in context])
        
        prompt = f"""
        You are an expert AI medical assistant designed for doctors and medical students. Your task is to provide a detailed, descriptive, and comprehensive answer to the following question based *exclusively* on the context provided from trusted medical documents.

        When formulating your answer, adhere to these guidelines:
        1.  **Be Detailed:** Synthesize information from all relevant context chunks to provide a thorough explanation. Do not just copy-paste a single chunk.
        2.  **Be Descriptive:** Use clear, professional medical terminology. Explain complex topics as you would to a medical colleague.
        3.  **Cite Everything:** After every key point or piece of information, you must cite the source using the format [Source: PDF_Name, Chunk X].
        4.  **Stay Grounded:** Only use the information from the context provided below. If the context does not contain the answer, state that clearly. Do not use outside knowledge.

        Context:
        ---
        {context_str}
        ---

        Question: {query}

        Detailed Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text, list(set([item['source'] for item in context]))
    except Exception as e:
        logging.error(f"An error occurred with the Gemini API call: {e}")
        return f"An error occurred with the AI model: {e}", []

def get_or_create_index(subject_name):
    """
    Checks if an index for a subject exists. If not, it creates one on-the-fly.
    """
    index_path = os.path.join(config.INDEX_FOLDER, subject_name.replace(" ", "_"))
    
    if os.path.exists(f"{index_path}.faiss"):
        logging.info(f"Index for '{subject_name}' already exists. Loading it.")
        return index_path
    
    logging.info(f"Index for '{subject_name}' not found. Creating it now...")
    pdf_path = os.path.join(config.PDF_FOLDER, f"{subject_name}.pdf")

    if not os.path.exists(pdf_path):
        logging.warning(f"PDF file not found for '{subject_name}' at {pdf_path}. Cannot create index.")
        return None
        
    chunks = load_and_chunk_pdf(pdf_path)
    if chunks:
        create_faiss_index(chunks, index_path)
        logging.info(f"✅ FAISS index created for: {subject_name}.pdf")
        return index_path
    
    return None

def get_rag_answer(query, subject=None, user_file_path=None):
    context = []
    
    # Always search the encyclopedia
    encyclopedia_index_path = get_or_create_index(config.ENCYCLOPEDIA_FILE.replace('.pdf', ''))
    if encyclopedia_index_path:
        context.extend(search_faiss_index(query, encyclopedia_index_path, top_k=3))

    # On-demand search for the selected subject
    if subject:
        subject_index_path = get_or_create_index(subject)
        if subject_index_path:
            context.extend(search_faiss_index(query, subject_index_path, top_k=3))

    # Search user-uploaded file
    if user_file_path:
        user_index_path = os.path.join(config.INDEX_FOLDER, "temp_user_file")
        user_chunks = load_and_chunk_pdf(user_file_path)
        if user_chunks:
            create_faiss_index(user_chunks, user_index_path)
            context.extend(search_faiss_index(query, user_index_path, top_k=2))
        
        if os.path.exists(f"{user_index_path}.faiss"): os.remove(f"{user_index_path}.faiss")
        if os.path.exists(f"{user_index_path}.meta"): os.remove(f"{user_index_path}.meta")

    if not context:
        return "Could not find any relevant information in the provided documents.", []

    unique_context = list({item['content']: item for item in context}.values())
    answer, sources = get_llm_response(query, unique_context)
    return answer, sorted(sources)

def initialize_rag_system():
    logging.info("--- Initializing RAG System for Local Development ---")
    os.makedirs(config.PDF_FOLDER, exist_ok=True)
    os.makedirs(config.INDEX_FOLDER, exist_ok=True)

    # On startup, only create the index for the encyclopedia
    logging.info("Checking for core Medical Encyclopedia index...")
    get_or_create_index(config.ENCYCLOPEDIA_FILE.replace('.pdf', ''))
    
    logging.info("--- RAG System Initialization Complete ---")
