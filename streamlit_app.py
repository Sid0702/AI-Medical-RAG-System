import streamlit as st
import os
import time

import config
import rag_pipeline as rag

st.set_page_config(
    page_title="AI Medical RAG System",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Medical RAG System")
st.markdown("Your Subject-Specific Medical Study Reference, powered by Google Gemini.")

# Initialize the core system (encyclopedia) once and cache it
@st.cache_resource
def startup_initialization():
    rag.initialize_rag_system()
    return True

if startup_initialization():
    st.sidebar.success("System Initialized")
else:
    st.sidebar.error("System failed to initialize. Check logs.")

dropdown_options = ["Default (Medical Encyclopedia)"] + config.MEDICAL_SUBJECTS

with st.form("query_form"):
    st.header("Ask a Medical Question")

    st.subheader("1. Upload Your PDF (Optional)")
    st.markdown("Upload a temporary medical file for context-specific Q&A.")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="file_uploader"
    )

    st.subheader("2. Select a Medical Subject")
    st.markdown("Choose a core subject to add to the context search. Defaults to the Medical Encyclopedia.")
    selected_option = st.selectbox(
        "Select Subject:",
        options=dropdown_options,
        index=0,
        key="subject_selector"
    )

    st.subheader("3. Ask Your Question")
    query_text = st.text_area(
        "Enter your question here:",
        placeholder="e.g., 'What are the primary symptoms of myocardial infarction?'",
        height=150,
        key="query_input"
    )

    submit_button = st.form_submit_button(label="Get Answer")

if submit_button:
    if not query_text:
        st.error("Please enter a question.")
    else:
        # Determine the subject to search
        subject_to_search = None
        if selected_option != "Default (Medical Encyclopedia)":
            subject_to_search = selected_option

        # Set the spinner message based on whether a new subject is being processed
        spinner_message = "Searching documents and generating answer..."
        if subject_to_search:
            index_path = os.path.join(config.INDEX_FOLDER, subject_to_search.replace(" ", "_") + ".faiss")
            if not os.path.exists(index_path):
                spinner_message = f"Processing '{subject_to_search}' for the first time and generating answer..."

        with st.spinner(spinner_message):
            user_file_path = None
            if uploaded_file is not None:
                try:
                    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
                    user_file_path = os.path.join(config.UPLOAD_FOLDER, uploaded_file.name)
                    with open(user_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except Exception as e:
                    st.error(f"Error saving uploaded file: {e}")
            
            answer, sources = rag.get_rag_answer(
                query=query_text, 
                subject=subject_to_search, 
                user_file_path=user_file_path
            )

            if user_file_path and os.path.exists(user_file_path):
                os.remove(user_file_path)

            st.subheader("📝 Answer")
            st.markdown(answer)

            if sources:
                st.subheader("📚 Sources Consulted")
                for source in sources:
                    st.markdown(f"- `{source}`")
