import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import pickle
import streamlit as st
import requests
import zipfile
from io import BytesIO
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

# --- API Key Configuration ---
try:
    # This will work when deployed on Streamlit Community Cloud
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, AttributeError):
    # This is a fallback for local development if you use a .env file
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

# --- Model Initialization ---
try:
    # Use the updated HuggingFaceEmbeddings wrapper
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
except Exception as e:
    EMBEDDING_MODEL = None

# --- Text Splitting and PDF Loading ---
def load_and_chunk_pdf(file_path):
    """
    Loads a PDF, extracts its text, and splits it using RecursiveCharacterTextSplitter.
    """
    doc_name = os.path.basename(file_path)
    full_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                full_text += page.get_text() + "\n"
    except Exception as e:
        return []

    if not full_text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    text_chunks = text_splitter.split_text(full_text)

    chunks_with_metadata = [
        {"content": chunk, "source": f"{doc_name}, Chunk {i+1}"}
        for i, chunk in enumerate(text_chunks)
    ]
    
    return chunks_with_metadata

# --- FAISS Indexing ---
def create_faiss_index(chunks, index_path):
    if not EMBEDDING_MODEL or not chunks:
        return

    contents = [chunk['content'] for chunk in chunks]
    # Use the embed_documents method for batch processing
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
        # Use the embed_query method for single queries
        query_embedding = EMBEDDING_MODEL.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        index_file = f"{index_path}.faiss"
        meta_file = f"{index_path}.meta"

        if not os.path.exists(index_file):
            return []

        index = faiss.read_index(index_file)
        with open(meta_file, 'rb') as f:
            metadata = pickle.load(f)
            
        _, indices = index.search(query_vector, top_k)
        
        valid_indices = [i for i in indices[0] if i < len(metadata)]
        return [metadata[i] for i in valid_indices]
    except Exception:
        return []

# --- LLM and RAG Logic ---
def get_llm_response(query, context):
    try:
        model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        context_str = "\n\n---\n\n".join([f"Source: {item['source']}\nContent: {item['content']}" for item in context])
        
        prompt = f"""
        You are a highly intelligent AI medical assistant. Answer medical questions based *only* on the context provided below.
        
        Context:
        ---
        {context_str}
        ---
        
        Cite your sources clearly using the format [Source: PDF_Name, Chunk X]. If the context does not contain the answer, state that clearly. Do not use outside knowledge.

        Question: {query}
        Answer:
        """
        
        response = model.generate_content(prompt)
        sources = list(set([item['source'] for item in context]))
        return response.text, sources
    except Exception as e:
        return f"An error occurred with the AI model: {e}", []

def get_rag_answer(query, subject=None, user_file_path=None):
    context = []
    all_sources = set()

    encyclopedia_index_name = config.ENCYCLOPEDIA_FILE.replace('.pdf', '')
    encyclopedia_search_path = os.path.join(config.INDEX_FOLDER, encyclopedia_index_name)
    encyclopedia_results = search_faiss_index(query, encyclopedia_search_path, top_k=3)
    if encyclopedia_results:
        context.extend(encyclopedia_results)
        for item in encyclopedia_results: all_sources.add(item['source'])

    if subject:
        subject_index_name = subject.replace(" ", "_")
        subject_search_path = os.path.join(config.INDEX_FOLDER, subject_index_name)
        subject_results = search_faiss_index(query, subject_search_path, top_k=3)
        if subject_results:
            context.extend(subject_results)
            for item in subject_results: all_sources.add(item['source'])

    if user_file_path:
        user_index_path = os.path.join(config.INDEX_FOLDER, "temp_user_file")
        user_chunks = load_and_chunk_pdf(user_file_path)
        if user_chunks:
            create_faiss_index(user_chunks, user_index_path)
            user_results = search_faiss_index(query, user_index_path, top_k=2)
            if user_results:
                context.extend(user_results)
                for item in user_results: all_sources.add(item['source'])
        
        if os.path.exists(f"{user_index_path}.faiss"): os.remove(f"{user_index_path}.faiss")
        if os.path.exists(f"{user_index_path}.meta"): os.remove(f"{user_index_path}.meta")

    if not context:
        return "Could not find any relevant information in the provided documents.", []

    unique_context = list({item['content']: item for item in context}.values())
    answer, _ = get_llm_response(query, unique_context)
    return answer, sorted(list(all_sources))

# --- System Initialization ---
def initialize_rag_system():
    os.makedirs(config.PDF_FOLDER, exist_ok=True)
    pdf_files_exist = len(os.listdir(config.PDF_FOLDER)) >= len(config.MEDICAL_SUBJECTS)

    if not pdf_files_exist:
        PDF_ZIP_URL = "https://drive.google.com/uc?export=download&id=1ltGh9dDrCT5_cvIk_R9m7lhSRm3AuTZh" 
        if PDF_ZIP_URL != "https://drive.google.com/uc?export=download&id=1ltGh9dDrCT5_cvIk_R9m7lhSRm3AuTZh":
            try:
                response = requests.get(PDF_ZIP_URL, stream=True)
                response.raise_for_status()
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall(config.PDF_FOLDER)
            except Exception:
                return

    os.makedirs(config.INDEX_FOLDER, exist_ok=True)
    all_pdfs = config.MEDICAL_SUBJECTS + [config.ENCYCLOPEDIA_FILE.replace('.pdf', '')]
    
    print("--- Initializing RAG System: Checking for FAISS indexes ---")
    for name in all_pdfs:
        pdf_path = os.path.join(config.PDF_FOLDER, f"{name}.pdf")
        index_path = os.path.join(config.INDEX_FOLDER, name.replace(" ", "_").replace('.pdf', ''))
        
        if not os.path.exists(f"{index_path}.faiss"):
            print(f"Index for '{name}' not found. Creating now...")
            if os.path.exists(pdf_path):
                chunks = load_and_chunk_pdf(pdf_path)
                if chunks:
                    create_faiss_index(chunks, index_path)
                    print(f"✅ FAISS index created for: {name}.pdf")
            else:
                print(f"⚠️ PDF not found for '{name}'. Skipping index creation.")
    print("--- RAG System Initialization Complete ---")
