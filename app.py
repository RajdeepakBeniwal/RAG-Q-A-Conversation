## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import os

from dotenv import load_dotenv
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .response-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Header
st.markdown("""
    <div class="main-header">
        <h1>📄 PDF Q&A Assistant</h1>
        <p>Chat with your PDF documents using AI-powered retrieval and conversation</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("API Settings", expanded=True):
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Enter your Groq API key to enable the chatbot"
        )
    
    with st.expander("Session Settings"):
        session_id = st.text_input(
            "📌 Session ID",
            value="default_session",
            help="Unique identifier for maintaining conversation history"
        )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info(
        "This application allows you to:\n"
        "1. Upload multiple PDF files\n"
        "2. Ask questions about their content\n"
        "3. Maintain conversation history\n"
        "4. Get AI-powered answers based on document context"
    )

# Main content
if not api_key:
    st.markdown("""
        <div class="warning-box">
            <h3>🔐 API Key Required</h3>
            <p>Please enter your Groq API key in the sidebar to get started.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")
    
    # Initialize session state
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    # File Upload Section
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files to analyze",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )
    
    if uploaded_files:
        # Show upload status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Uploaded", len(uploaded_files))
        
        # Process PDFs
        with st.spinner("Processing documents..."):
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = f"./temp.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
            
            if documents:
                with col2:
                    st.metric("Pages Processed", len(documents))
                
                # Split and create embeddings
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=500
                )
                splits = text_splitter.split_documents(documents)
                
                with col3:
                    st.metric("Chunks Created", len(splits))
                
                # Create vector store
                client = chromadb.EphemeralClient()
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    client=client,
                    collection_name="pdf_documents"
                )
                retriever = vectorstore.as_retriever()
                
                # Setup RAG chain
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm,
                    retriever,
                    contextualize_q_prompt
                )
                
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise.\n\n{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                def get_session_history(session: str) -> BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id] = ChatMessageHistory()
                    return st.session_state.store[session_id]
                
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                # Chat Interface
                st.header("💬 Ask Questions")
                st.markdown("---")
                
                # Get session history once
                session_history = get_session_history(session_id)
                
                # Question input
                user_input = st.text_input(
                    "Your question:",
                    placeholder="What would you like to know about your documents?",
                    help="Type your question and press Enter"
                )
                
                if user_input:
                    with st.spinner("Searching documents and generating response..."):
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )
                        
                        # Display response
                        st.markdown("""
                            <div class="response-box">
                                <h4>✅ Answer</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(response['answer'])
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Chat History Section
                if session_history.messages:
                    with st.expander("📋 Chat History", expanded=False):
                        st.markdown("**Conversation**")
                        for i, message in enumerate(session_history.messages):
                            if message.type == "human":
                                st.markdown(f"**You:** {message.content}")
                            else:
                                st.markdown(f"**Assistant:** {message.content}")
                            if i < len(session_history.messages) - 1:
                                st.markdown("---")
    
    else:
        st.markdown("""
            <div class="info-box">
                <h4>📚 Getting Started</h4>
                <p>Upload one or more PDF files to begin your conversation. The AI will analyze the documents and be ready to answer your questions!</p>
            </div>
        """, unsafe_allow_html=True)










