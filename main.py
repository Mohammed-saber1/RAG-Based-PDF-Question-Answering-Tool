import streamlit as st
from src.app.app import PDFQuestionAnsweringApp

class PDFQuestionAnsweringAppRunner:
    """Runner class for the PDF Question Answering Streamlit application."""

    def __init__(self):
        """Initialize the application runner."""
        self.app = PDFQuestionAnsweringApp()

    def run(self) -> None:
        """Run the Streamlit application."""
        try:
            st.set_page_config(
                page_title="RAG-Based PDF QA Tool",
                page_icon="📚",
                layout="wide"
            )
            self.app.render()
        except Exception as e:
            st.error(f"Failed to run application: {str(e)}")

if __name__ == "__main__":
    runner = PDFQuestionAnsweringAppRunner()
    runner.run()