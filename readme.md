Here's your content formatted as a proper `README.md` file:

```markdown
# 🧠 AI Medical RAG System

This project is an advanced **Retrieval-Augmented Generation (RAG)** system designed for **doctors**, **medical students**, and **researchers**. It provides a fast, subject-specific reference tool by leveraging a local library of medical PDFs and a powerful Large Language Model (LLM).

The system is optimized for performance with an on-demand indexing strategy. It only processes and indexes the main `Medical_Encyclopedia.pdf` on startup, creating indexes for other subjects the first time they are selected by the user.

---

## 🚀 Key Features

- **On-Demand Indexing**: Ensures fast startup times by only processing subject PDFs when they are first requested.
- **Dynamic PDF Uploads**: Users can temporarily upload their own medical PDFs for context-specific Q&A.
- **Pre-loaded Medical Library**: Comes with a repository of categorized medical subject files and a persistent Medical Encyclopedia.
- **Intelligent Context Retrieval**: Combines relevant text chunks from user-uploaded files, a selected subject PDF, and the encyclopedia.
- **Advanced Text Chunking**: Utilizes `RecursiveCharacterTextSplitter` to intelligently split text while preserving semantic context.
- **High-Speed Search**: Employs a **FAISS** vector database for efficient similarity searches.
- **State-of-the-Art AI**: Powered by **Google's Gemini 1.5 Flash** and **Hugging Face sentence-transformers**.
- **Source Attribution**: Every answer includes citations pointing to the exact source document and chunk number.
- **User-Friendly Interface**: Built with **Streamlit** for a clean and interactive user experience.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **LLM**: Google Gemini 1.5 Flash  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`  
- **Vector Database**: FAISS (Facebook AI Similarity Search)  
- **PDF Parsing**: PyMuPDF  
- **Text Splitting**: LangChain (`RecursiveCharacterTextSplitter`)  
- **Backend**: Python

---

## 📂 Project Structure

```

ai\_medical\_rag/
├── 📂 faiss\_indexes/      # Auto-generated storage for vector indexes
├── 📂 medical\_pdfs/       # Store your core medical PDFs here
├── 📂 uploads/            # Temporary storage for user-uploaded files
├── 📜 config.py           # Main configuration for paths, models, and chunking
├── 📜 rag\_pipeline.py     # Core RAG logic (PDF processing, search, LLM calls)
├── 📜 streamlit\_app.py    # The main Streamlit frontend application
├── 📜 requirements.txt    # Python dependencies
└── 📜 README.md           # Project documentation (this file)

````

---

## ⚙️ Local Setup and Installation

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Sid0702/AI-Medical-RAG-System.git
cd AI-Medical-RAG-System
````

### 2. Create a Virtual Environment

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

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

You need a Google API key to use the Gemini model.

* Go to **[Google AI Studio](https://makersuite.google.com/)** to create your API key.
* Create a file named `.env` in the root of your project directory.
* Add your API key like this:

```env
GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. Add Your PDF Files

* Create a folder named `medical_pdfs` in the project root.
* Place all your subject-specific PDFs and your `Medical_Encyclopedia.pdf` inside this folder.

> 🔥 **Important**: Ensure the filenames exactly match the names listed in the `MEDICAL_SUBJECTS` list in `config.py`.

### 6. Run the Application

The first time you run the app, it will only process the `Medical_Encyclopedia.pdf`. The first time you select a new subject from the dropdown, there will be a **one-time delay** as it creates the index for that file.

```bash
streamlit run streamlit_app.py
```

Then, open your browser and navigate to:

```
http://localhost:8501
```

---

## 📌 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Hugging Face](https://huggingface.co/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)
* [Google Gemini](https://ai.google.dev/)

```

Let me know if you want me to export this as a `.md` file.
```
