import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
INDEX_FOLDER = os.path.join(BASE_DIR, 'faiss_indexes')
PDF_FOLDER = os.path.join(BASE_DIR, 'medical_pdfs')

MEDICAL_SUBJECTS = [
    "Anatomy", 
    "Anesthesia", 
    "Biochemistry", 
    "Dermatology", 
    "Diseases of the Ear, Nose, and Throat Dhingra",
    "FORENSIC-MEDICINE-AND-TOXICOLOGY",
    "Gynecology", 
    "Manual-of-Medicine", 
    "Microbiology", 
    "Obstetrics",
    "Opthalmology",
    "Orthopaedics", 
    "Pathology", 
    "Pediatrics", 
    "Pharmacology",
    "Physiology", 
    "Radiology", 
    "Surgery"
]
ENCYCLOPEDIA_FILE = "Medical_Encyclopedia.pdf"

# --- Model & RAG Parameters ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = "gemini-1.5-flash" # Using Flash for speed and cost-effectiveness

# Industry-standard chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
