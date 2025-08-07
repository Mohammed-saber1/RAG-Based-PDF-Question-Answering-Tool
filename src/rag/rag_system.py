import tempfile
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.rag.embeddings import HuggingFaceEmbeddingsWrapper
from typing import Optional
from dotenv import load_dotenv
import os

class PDFRAGSystem:
    """A Retrieval-Augmented Generation system for processing PDFs and answering questions."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the PDF RAG system.

        Args:
            api_key (str): API key for Groq model.
            model (str): Name of the Groq model to use. Defaults to 'llama-3.3-70b-versatile'.
        
        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            load_dotenv()  # Load environment variables from .env file
            self.embeddings = HuggingFaceEmbeddingsWrapper()
            self.llm = ChatGroq(model=model, api_key=api_key)
            self.vector_store = None
            self.rag_chain = None
            self.chat_history = []
            print("✅ PDF RAG System initialized!")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PDFRAGSystem: {str(e)}")

    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file and create a vector store.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Status message indicating success or failure.
        """
        try:
            if not pdf_path.endswith('.pdf'):
                return "❌ Invalid file format: Please provide a PDF file."

            print(f"📄 Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"📖 Loaded {len(documents)} pages")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")

            temp_dir = tempfile.mkdtemp()
            print("🔍 Creating vector store...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=temp_dir
            )

            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            self._setup_rag_chain(retriever)
            
            result = f"Successfully processed PDF with {len(chunks)} chunks."
            print(result)
            return result
        except Exception as e:
            error_msg = f"❌ Error processing PDF: {str(e)}"
            print(error_msg)
            return error_msg

    def _setup_rag_chain(self, retriever) -> None:
        """
        Setup the RAG chain with history awareness.

        Args:
            retriever: The retriever object for document search.
        """
        try:
            print("🔗 Setting up RAG chain...")
            
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, just "
                "reformulate it if needed and otherwise return it as is."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )
            
            qa_system_prompt = (
            "You are a helpful AI assistant answering questions only based on the provided document. "
            "IMPORTANT: Keep all responses short and concise – maximum 3 sentences. "
            "Provide only the most essential information as a clear overview. "
            "Give direct, brief answers that summarize the key points from the document. "
            "If the user asks about anything unrelated to this PDF, respond ONLY with: 'Out of context' "
            "\nContext: {context}"
        )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            print("✅ RAG chain setup complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to setup RAG chain: {str(e)}")

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.

        Args:
            question (str): The question to ask about the PDF content.

        Returns:
            str: The answer or an error message.
        """
        if not self.rag_chain:
            return "Please upload and process a PDF first."
        
        try:
            if not question or not isinstance(question, str):
                return "Invalid question: Please provide a non-empty string."
                
            print(f"❓ Processing query: {question}")
            result = self.rag_chain.invoke({
                "input": question, 
                "chat_history": self.chat_history
            })
            
            answer = result['answer']
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            print(f"💭 Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            print(error_msg)
            return error_msg

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        print("🗑️ Chat history cleared!")