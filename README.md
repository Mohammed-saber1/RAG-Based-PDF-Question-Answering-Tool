# RAG-Based PDF Question Answering Tool

A Streamlit application that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded PDF documents.

## Features
- Upload and process PDF documents
- Ask questions about the document content
- Context-aware responses with chat history
- Concise answers (max 3 sentences)
- Supports text extraction and similarity-based search

## Prerequisites
- Python 3.8+
- A Groq API key (sign up at https://console.groq.com)

## Setup Instructions
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
1. Open the application in your browser (usually at `http://localhost:8501`).
2. Upload a PDF file using the sidebar (max 10MB).
3. Wait for the PDF to be processed.
4. Ask questions about the document content in the chat interface.

## Project Structure
```
├── src/
│   ├── app/
│   │   └── app.py          # Streamlit application logic
│   └── rag/
│       ├── embeddings.py   # HuggingFace embeddings wrapper
│       └── rag_system.py   # RAG system implementation
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Notes
- The application uses a temporary directory for vector store persistence.
- Ensure your PDF files are text-based (not scanned images) for optimal results.
- The `.env` file should never be committed to version control.

## Troubleshooting
- If you encounter dependency issues, ensure you're using the specified versions in `requirements.txt`.
- For API-related errors, verify your Groq API key in the `.env` file.
- For large PDFs, processing may take longer; consider splitting large documents.

## License
MIT License
