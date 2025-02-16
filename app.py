#!/usr/bin/env python
import streamlit as st
import tempfile

from chapter_extractor import extract_chapters 
from summarizer import summarize_text
from rag_qa import answer_question
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self.model.encode(texts, convert_to_numpy=True)]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

def encode_pdf(path, chunk_size=2000):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=int(chunk_size*0.2), length_function=len
    )

    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformersEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def main():
    st.title("Book Chapter Summarization & Q&A")
    st.markdown(
        """
        **Workflow:**
        1. **Upload a Book (PDF):** The app extracts chapters from your book.
        2. **Chapter Summarization:** Select a chapter to generate its summary.
        3. **Chapter Q&A:** Ask specific questions about the chapter. 
           The answer uses the vectorstore built from the PDF to reduce hallucinations.
        """
    )

    uploaded_file = st.file_uploader("Upload your book (PDF)", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.info("Extracting chapters from the book...")
        chapters = extract_chapters(tmp_file_path)
        num_chapters = len(chapters)
        st.success(f"Extracted {num_chapters} chapters from the book.")

        st.info("Creating vectorstore from the uploaded PDF...")
        vectorstore = encode_pdf(tmp_file_path, chunk_size=1000)
        st.session_state["vectorstore"] = vectorstore
        st.success("Vectorstore created and stored for Q&A.")

        if st.button("Generate Global Book Summary"):
            global_summary = ""
            progress_bar = st.progress(0)
            for i, chapter in enumerate(chapters):
                st.write(f"Summarizing Chapter {i+1}...")
                chapter_summary = summarize_text(chapter)
                global_summary += f"Chapter {i+1} Summary:\n{chapter_summary}\n\n"
                progress_bar.progress((i + 1) / num_chapters)
            st.session_state["global_summary"] = global_summary
            st.success("Global book summary generated.")
            st.text_area("Global Book Summary", value=global_summary, height=300)

        chapter_number = st.selectbox(
            "Select a Chapter",
            list(range(1, num_chapters + 1)),
            format_func=lambda x: f"Chapter {x}"
        )
        chapter_text = chapters[chapter_number - 1]
        if st.button("Summarize Selected Chapter"):
            with st.spinner("Generating chapter summary..."):
                chapter_summary = summarize_text(chapter_text)
                st.session_state["chapter_summary"] = chapter_summary

        if "chapter_summary" in st.session_state:
            st.text_area("Chapter Summary", value=st.session_state["chapter_summary"], height=300)

        st.markdown("### Ask a Question about the Book")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                if "vectorstore" not in st.session_state:
                    st.error("Vectorstore not found. Please upload a PDF first.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = answer_question(st.session_state["vectorstore"], question)
                        st.markdown("**Answer:**")
                        st.write(answer)

if __name__ == "__main__":
    main()
