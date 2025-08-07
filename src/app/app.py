import os
import streamlit as st
import tempfile
import hashlib
from src.rag.rag_system import PDFRAGSystem
from dotenv import load_dotenv
from typing import Optional

class PDFQuestionAnsweringApp:
    """Streamlit application for PDF question answering using RAG."""

    def __init__(self):
        """Initialize the Streamlit application with RAG system."""
        try:
            load_dotenv()
            self._initialize_rag_system()
            self.initialize_session_state()
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            raise

    def _initialize_rag_system(self) -> None:
        """Initialize the RAG system with Groq API key."""
        if 'rag_system' not in st.session_state:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            st.session_state.rag_system = PDFRAGSystem(api_key=api_key)
        self.rag_system = st.session_state.rag_system

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state with default values."""
        defaults = {
            'messages': [],
            'pdf_processed': False,
            'pdf_name': "",
            'pdf_hash': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render(self) -> None:
        """Render the Streamlit application."""
        try:
            st.title("📚 RAG-Based PDF Question Answering Tool")
            st.markdown("Upload a PDF document and ask questions about its content!")
            self.render_sidebar()
            if not st.session_state.get('pdf_processed', False):
                self.render_welcome_screen()
            else:
                self.render_chat_interface()
        except Exception as e:
            st.error(f"Error rendering application: {str(e)}")

    def render_sidebar(self) -> None:
        """Render the sidebar with PDF upload controls."""
        with st.sidebar:
            st.header("📄 Document Upload")
            
            if st.session_state.get('pdf_processed', False):
                st.success(f"✅ PDF Loaded: {st.session_state.get('pdf_name', 'Unknown')}")
            else:
                st.info("No PDF loaded yet")
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type="pdf",
                help="Upload a PDF document to analyze (max 10MB)",
                key="pdf_uploader"
            )
            
            if uploaded_file is not None:
                file_content = uploaded_file.getvalue()
                if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
                    st.error("❌ File size exceeds 10MB limit")
                    return
                
                file_hash = hashlib.md5(file_content).hexdigest()
                
                if file_hash != st.session_state.get('pdf_hash'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file_content)
                        temp_path = tmp_file.name
                    
                    with st.spinner("Processing PDF..."):
                        result = self.rag_system.process_pdf(temp_path)
                        if "Successfully processed" in result:
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.pdf_hash = file_hash
                            st.success(f"✅ {result}")
                            try:
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to refresh page: {str(e)}")
                        else:
                            st.error(f"❌ {result}")
                    
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        print(f"Warning: Failed to delete temporary file: {str(e)}")
            
            if st.session_state.get('pdf_processed', False):
                st.markdown("---")
                if st.button("Clear Chat History", type="secondary"):
                    self.rag_system.clear_history()
                    st.session_state.messages = []
                    st.session_state.pdf_processed = False
                    st.session_state.pdf_name = ""
                    st.session_state.pdf_hash = None
                    st.success("Chat history and PDF cleared!")
                    try:
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to refresh page: {str(e)}")

    def render_welcome_screen(self) -> None:
        """Render welcome screen when no PDF is loaded."""
        st.markdown("""
        ## Welcome to the PDF QA Tool! 👋
        
        To get started:
        1. **Upload a PDF** using the file uploader in the sidebar (max 10MB)
        2. **Wait for processing** - this usually takes a few seconds
        3. **Start asking questions** about your document content
        
        ### Features:
        - 🔍 **Smart Search**: Uses advanced AI to find relevant information
        - 💬 **Chat Interface**: Natural conversation with your documents
        - 📝 **Context Aware**: Remembers previous questions in the conversation
        - ⚡ **Fast Responses**: Optimized for quick and accurate answers
        
        ### Supported Features:
        - Text extraction from PDF documents
        - Question answering based on document content
        - Chat history for context-aware responses
        - Concise, focused answers (max 3 sentences)
        """)

    def render_chat_interface(self) -> None:
        """Render chat interface when PDF is loaded."""
        st.header(f"💬 Chat with: {st.session_state.get('pdf_name', 'Unknown')}")
        
        for message in st.session_state.get('messages', []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask a question about your PDF..."):
            if not prompt.strip():
                st.error("Please enter a valid question")
                return
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = self.rag_system.query(prompt)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})