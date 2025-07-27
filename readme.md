# AI Medical RAG System

This project is an advanced Retrieval-Augmented Generation (RAG) system designed for doctors, medical students, and researchers. It provides a fast, subject-specific reference tool by leveraging a local library of medical PDFs and a powerful Large Language Model (LLM).

## 🚀 Key Features

- **Dynamic PDF Uploads**: Users can temporarily upload their own medical PDFs for context-specific Q&A
- **Pre-loaded Medical Library**: Comes with a fixed repository of categorized medical subject files and a persistent Medical Encyclopedia that is always included in searches
- **Intelligent Context Retrieval**: Combines relevant text chunks from user-uploaded files, a selected subject PDF, and the encyclopedia to provide comprehensive answers
- **Advanced Text Chunking**: Utilizes RecursiveCharacterTextSplitter to intelligently split text while preserving semantic context
- **High-Speed Search**: Employs a FAISS vector database for efficient and fast similarity searches
- **State-of-the-Art AI**: Powered by Google's Gemini 1.5 Flash for answer generation and Hugging Face sentence-transformers for creating text embeddings
- **Source Attribution**: Every answer includes citations pointing to the exact source document and chunk number, ensuring verifiability
- **User-Friendly Interface**: Built with Streamlit for a clean, responsive, and interactive user experience

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 via langchain-huggingface
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Parsing**: PyMuPDF
- **Text Splitting**: LangChain (RecursiveCharacterTextSplitter)
- **Backend**: Python

## 📂 Project Structure

```
ai_medical_rag/
│
├── 📂 faiss_indexes/      # Auto-generated storage for vector indexes
│
├── 📂 medical_pdfs/       # Store your core medical PDFs here
│
├── 📂 uploads/            # Temporary storage for user-uploaded files
│
├── 📜 config.py           # Main configuration for paths, models, and chunking
│
├── 📜 rag_pipeline.py     # Core RAG logic (PDF processing, search, LLM calls)
│
├── 📜 streamlit_app.py    # The main Streamlit frontend application
│
├── 📜 requirements.txt    # Python dependencies
│
└── 📜 README.md           # Project documentation (this file)
```

## ⚙️ Local Setup and Installation

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Sid0702/AI-Medical-RAG-System.git
cd AI-Medical-RAG-System
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

You need a Google API key to use the Gemini model.

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to create your API key
2. Create a file named `.env` in the root of your project directory
3. Add your API key to the `.env` file like this:

```env
GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. Add Your PDF Files

1. Create a folder named `medical_pdfs` in the project root
2. Place all your subject-specific PDFs and your `Medical_Encyclopedia.pdf` inside this folder
3. **Important**: Ensure the filenames exactly match the names listed in the `MEDICAL_SUBJECTS` list in `config.py`

### 6. Run the Application

The first time you run the app, it will process all your PDFs and create the FAISS indexes. This may take several minutes. Subsequent launches will be much faster.

```bash
streamlit run streamlit_app.py
```

Open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).

## ☁️ Deployment on Streamlit Community Cloud

Follow these steps to deploy the application for free.

### 1. Prepare Large Files

GitHub has a file size limit, so you cannot upload your `medical_pdfs` folder directly.

1. Create a `.zip` file of your `medical_pdfs` folder (e.g., `medical_pdfs.zip`)
2. Upload this `.zip` file to a cloud storage service like [Google Drive](https://drive.google.com/) or Dropbox
3. Get a direct download link for the file. For Google Drive, you can use a service like [gdown.pl](https://gdown.pl/) to convert the shareable link into a direct one

### 2. Update the Code for Deployment

1. Open `rag_pipeline.py`
2. Find the `initialize_rag_system` function
3. Replace the placeholder `YOUR_DIRECT_DOWNLOAD_LINK_HERE` with the direct download link you obtained in the previous step

### 3. Push to GitHub

1. Make sure your project is a public GitHub repository and that you have pushed all the latest code changes
2. Ensure your `.gitignore` file is correctly configured to exclude the `medical_pdfs/`, `faiss_indexes/`, and `venv/` folders

### 4. Deploy on Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account
2. Click "New app" and select your repository
3. In the "Advanced settings...", add your Google API key as a secret:

```toml
# secrets.toml
GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

4. Click "Deploy!". The app will build, download your PDFs, create the indexes, and then go live

## 📝 Usage

1. **Select a Medical Subject**: Choose from the dropdown menu to focus your search on a specific medical domain
2. **Upload Additional PDFs** (Optional): Upload your own medical documents for additional context
3. **Ask Your Question**: Enter your medical query in the text area
4. **Get Comprehensive Answers**: Receive detailed responses with source citations from the relevant medical literature

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. Always consult with qualified healthcare professionals for medical advice and decision-making.