import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="RAG Chat", layout="wide")

import os
import tempfile
from werkzeug.utils import secure_filename
import time
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz  # PyMuPDF for PDF handling
import easyocr
import ssl
from io import BytesIO
import requests
from transformers import pipeline

# Import RAG system components
from rag_system.rag_system import RAGSystem

# SSL configuration for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

# Add professional styling
st.markdown("""
    <style>
    /* Base text colors */
    .stApp {
        color: #2C3E50;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #2C3E50 !important;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 5px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E8F4F9;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E3D59;
        color: white !important;
    }

    /* Text colors for better visibility */
    .stMarkdown {
        color: #2C3E50 !important;
    }
    
    div[data-testid="stText"] {
        color: #2C3E50 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1E3D59 !important;
        font-weight: 600 !important;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #E9ECEF !important;
    }

    .stChatMessage [data-testid="StyledLinkIconContainer"] {
        color: #2C3E50 !important;
    }

    .stChatMessage p {
        color: #2C3E50 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #E8F4F9 !important;
        color: #1E3D59 !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        border: 1px solid #E9ECEF !important;
        background-color: #ffffff !important;
    }

    /* Button text color */
    .stButton button {
        color: white !important;
        background-color: #1E3D59 !important;
    }

    /* Info message styling */
    .stInfo {
        background-color: #E8F4F9 !important;
        color: #1E3D59 !important;
    }

    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }

    /* Error message styling */
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }

    /* Custom summary card */
    .summary-card {
        background-color: #ffffff;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .summary-header {
        color: #1E3D59;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1em;
    }

    .summary-content {
        color: #2C3E50;
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        line-height: 1.5;
    }

    /* Chat container styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 200px);
        position: relative;
    }

    .chat-history {
        flex-grow: 1;
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 80px;  /* Space for input box */
    }

    .chat-input-fixed {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 1px solid #E9ECEF;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }

    /* Ensure chat messages are visible */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #E9ECEF !important;
        margin: 8px 0 !important;
        padding: 15px !important;
    }

    .user-message {
        background-color: #E8F4F9 !important;
        margin-left: 20px !important;
    }

    .assistant-message {
        background-color: #F9F9F9 !important;
        margin-right: 20px !important;
    }

    /* ChatGPT-like interface styling */
    .chat-interface {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 150px);
        padding: 20px;
    }

    .chat-history {
        flex-grow: 1;
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 60px;
    }

    .chat-message {
        padding: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }

    .user-message {
        background-color: #f7f7f8;
        border-radius: 8px;
    }

    .assistant-message {
        background-color: #f7f7f8;
        border-radius: 8px;
    }

    .message-content {
        flex-grow: 1;
        color: #2C3E50;
        line-height: 1.6;
    }

    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem 20px;
        background-color: #f7f7f8;
    }

    .chat-input-box {
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 8px;
        background-color: #f7f7f8;
    }

    /* Avatar styling */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 4px;
        margin-right: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background-color: #1E3D59;
        color: white;
    }

    /* Source styling */
    .source-container {
        background-color: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }

    .source-title {
        color: #1E3D59;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .role-label {
        font-weight: 600;
        margin-bottom: 8px;
        color: #1E3D59;
    }
    </style>
""", unsafe_allow_html=True)

# Add title
st.title("📘 RAG-SWAT")

# Create temp folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize the RAG system
@st.cache_resource
def initialize_rag_system():
    rag = RAGSystem(
        api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk",
        model_name="gemini-2.0-flash",
        embedding_model="all-MiniLM-L6-v2"
    )
    return rag

# Initialize Gemini for text processing and summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"
)

# Initialize EasyOCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

# Initialize LLaVA pipeline for image description
@st.cache_resource
def get_vlm_pipeline():
    try:
        # Using a smaller model for faster inference
        vlm = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return vlm
    except Exception as e:
        st.error(f"Failed to load VLM: {e}")
        return None

# Initialize session state for documents and chat history
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'document_names' not in st.session_state:
    st.session_state.document_names = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_bytes):
    """Extract text from image using EasyOCR"""
    try:
        reader = get_ocr_reader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        
        result = reader.readtext(temp_path)
        text = "\n".join([entry[1] for entry in result])
        
        os.unlink(temp_path)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def describe_image(image_bytes):
    """Generate description of image using VLM"""
    try:
        vlm = get_vlm_pipeline()
        if not vlm:
            return "Unable to load image description model"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        
        # Open the image
        image = Image.open(temp_path)
        
        # Generate description
        description = vlm(image)[0]['generated_text']
        
        os.unlink(temp_path)
        return description
    except Exception as e:
        st.error(f"Image description error: {e}")
        return ""

def generate_short_summary(text, title, is_image=False):
    """Generate a very short summary (max 100 words) using Gemini"""
    if not text or len(text.strip()) < 10:
        return "Not enough content to generate summary"
    
    try:
        if is_image:
            prompt = f"""
            Please provide a very short summary (maximum 100 words) of this image titled '{title}' 
            based on its visual description. Focus on the main subject and key visual elements:
            
            {text[:5000]}
            """
        else:
            prompt = f"""
            Please provide a very short summary (maximum 100 words) of the following content from '{title}'. 
            Focus on the main topic and key points:
            
            {text[:5000]}
            """
        
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        
        # Ensure summary is within word limit
        words = summary.split()
        if len(words) > 100:
            summary = ' '.join(words[:100]) + "..."
        
        return summary
    except Exception as e:
        st.error(f"Short summary error: {e}")
        return "Summary unavailable"

def summarize_text(text, title):
    """Generate a summary of the extracted text using Gemini via LangChain"""
    if not text or len(text.strip()) < 100:
        return text

    try:
        prompt = f"""
        Please summarize the following text extracted from the document titled '{title}'. 
        Maintain key facts, figures, and important information:
        
        {text[:10000]}
        """
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        
        return f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"
    except Exception as e:
        st.error(f"Summarization Error: {e}")
        return text

def process_files(uploaded_files):
    saved_paths = []
    processed_files = []
    summaries = []  # To store summaries for display
    
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.name):
            filename = secure_filename(uploaded_file.name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            summary = None
            is_image = file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            
            if is_image:
                # Process image files
                extracted_text = extract_text_from_image(uploaded_file.getvalue())
                
                if not extracted_text or len(extracted_text.strip()) < 10:
                    # If no text found, use VLM to describe the image
                    description = describe_image(uploaded_file.getvalue())
                    
                    if description:
                        text_filename = f"{filename}_description.txt"
                        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        
                        with open(text_path, 'w', encoding='utf-8') as text_file:
                            text_file.write(f"IMAGE DESCRIPTION: {description}")
                        
                        saved_paths.append(text_path)
                        st.session_state.document_names.append(f"{filename} (Image Description)")
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": text_filename,
                            "type": "image_description"
                        })
                        summary = generate_short_summary(description, filename, is_image=True)
                else:
                    # We have OCR text
                    text_filename = f"{filename}_ocr.txt"
                    text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                    
                    processed_text = summarize_text(extracted_text, filename)
                    
                    with open(text_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(processed_text)
                    
                    saved_paths.append(text_path)
                    st.session_state.document_names.append(f"{filename} (OCR)")
                    processed_files.append({
                        "original_name": filename,
                        "processed_name": text_filename,
                        "type": "image_ocr"
                    })
                    summary = generate_short_summary(extracted_text, filename, is_image=True)
            
            elif file_ext == 'pdf':
                # Process PDF files
                try:
                    doc = fitz.open(file_path)
                    pdf_text = ""
                    for page in doc:
                        pdf_text += page.get_text()
                    doc.close()
                    
                    if pdf_text:
                        summary = generate_short_summary(pdf_text, filename)
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                
                saved_paths.append(file_path)
                st.session_state.document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
            
            else:
                # Process other document types
                saved_paths.append(file_path)
                st.session_state.document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
                
                # Try to read text files for summary
                if file_ext == 'txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read(5000)  # Read first 5000 chars
                            summary = generate_short_summary(text, filename)
                    except:
                        pass
            
            if summary:
                summaries.append({
                    "filename": filename,
                    "summary": summary,
                    "type": "image" if is_image else "document"
                })
    
    return saved_paths, processed_files, summaries

def display_chat_message(role, content, sources=None):
    """Display a chat message with appropriate styling"""
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**{content}**")
            
            if sources:
                with st.expander("View Sources", expanded=False):
                    for i, source in enumerate(sources, 1):
                        doc_name = source['metadata'].get('document_name', 'Unknown')
                        
                        if "(OCR)" in doc_name:
                            icon = "🖼️ (Text from Image)"
                        elif "(Image Description)" in doc_name:
                            icon = "🖼️ (Image Description)"
                        else:
                            icon = "📄"
                        
                        st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
                        content = source['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content)

# Get RAG instance
rag = initialize_rag_system()

# Initialize session state for summaries if not exists
if 'summaries' not in st.session_state:
    st.session_state.summaries = []

# Sidebar for document upload and management
with st.sidebar:
    st.subheader("📂 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process button
    if st.button("Process Documents", key="process_btn"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
                    new_documents = rag.document_processor.process_documents(saved_paths)
                    st.session_state.all_documents.extend(new_documents)
                    rag.create_vector_store(st.session_state.all_documents)
                    rag.create_rag_chain(chain_type="stuff", k=4)
                    st.session_state.documents_processed = True
                    st.session_state.processed_files.extend(processed_files)
                    
                    # Store summaries in session state
                    if summaries:
                        if 'summaries' not in st.session_state:
                            st.session_state.summaries = []
                        st.session_state.summaries.extend(summaries)
                    
                    st.success(f"✅ Processed {len(uploaded_files)} files!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        else:
            st.warning("Please upload files first")
    
    # Display processed documents count
    if st.session_state.get("documents_processed", False):
        total_docs = len(st.session_state.get("processed_files", []))
        st.write(f"📚 **Total documents processed:** {total_docs}")
    
    # Clear all button
    if st.button("Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.summaries = []
        st.session_state.documents_processed = False
        st.session_state.processed_files = []
        st.session_state.all_documents = []
        st.rerun()

# Create tabs for Summary and Chat sections
tab1, tab2 = st.tabs(["📝 Document Summaries", "💬 Chat Interface"])

# Summary Tab
with tab1:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 20px;'>
            <h2 style='color: #1E3D59; margin-bottom: 10px;'>📝 Document Summaries</h2>
            <p style='color: #2C3E50; font-size: 1.1em;'>View summaries of all processed documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get("summaries"):
        # Create a grid layout for summaries
        cols = st.columns(2)
        for idx, item in enumerate(st.session_state.summaries):
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class='summary-card'>
                        <div class='summary-header'>
                            {'🖼️' if item['type'] == 'image' else '📄'} {item['filename']}
                        </div>
                        <div class='summary-content'>
                            {item['summary']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📥 Upload and process documents to see their summaries here")

# Chat Tab
with tab2:
    st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
    
    # Chat input (placed before history to stay at bottom)
    st.markdown("""
        <div class="chat-input-container">
            <div class="chat-input-box">
    """, unsafe_allow_html=True)
    
    user_question = st.chat_input("Ask a question about your documents...")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Chat history
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    
    # Display chat history in reverse order (excluding the last message if it's new)
    history_to_display = st.session_state.chat_history[:-2] if user_question else st.session_state.chat_history
    
    for chat in reversed(history_to_display):
        role = chat["role"]
        content = chat["content"]
        sources = chat.get("sources", None)
        
        st.markdown(f"""
            <div class="chat-message {'user-message' if role == 'user' else 'assistant-message'}">
                <div class="avatar">{'👤' if role == 'user' else '🤖'}</div>
                <div class="message-content">
                    <div class="role-label">{role.capitalize()}</div>
                    <div>{content}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available with toggle
        if sources:
            with st.expander("View Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    doc_name = source['metadata'].get('document_name', 'Unknown')
                    
                    if "(OCR)" in doc_name:
                        icon = "🖼️ (Text from Image)"
                    elif "(Image Description)" in doc_name:
                        icon = "🖼️ (Image Description)"
                    else:
                        icon = "📄"
                    
                    st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                    
                    content = source['content']
                    if len(content) > 500:
                        content = content[:500] + "..."
                    st.code(content)
    
    st.markdown("</div>", unsafe_allow_html=True)

    if user_question:
        # Show the new user question
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="avatar">👤</div>
                <div class="message-content">
                    <div class="role-label">User</div>
                    <div>{user_question}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Create a placeholder for the assistant's response
        assistant_placeholder = st.empty()
        
        # Get assistant response
        with st.spinner("Thinking..."):
            try:
                result = rag.query(user_question)
                response_text = result['answer']
                
                # Display response with typing effect
                displayed_response = ""
                for i in range(len(response_text)):
                    displayed_response += response_text[i]
                    assistant_placeholder.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div class="avatar">🤖</div>
                            <div class="message-content">
                                <div class="role-label">Assistant</div>
                                <div>{displayed_response}▌</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    if i % 2 == 0:  # Adjust speed by changing this value
                        time.sleep(0.01)  # Adjust typing speed
                
                # Show final response without cursor
                assistant_placeholder.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="avatar">🤖</div>
                        <div class="message-content">
                            <div class="role-label">Assistant</div>
                            <div>{response_text}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                if result['sources']:
                    with st.expander("View Sources", expanded=False):
                        for i, source in enumerate(result['sources'], 1):
                            doc_name = source['metadata'].get('document_name', 'Unknown')
                            
                            if "(OCR)" in doc_name:
                                icon = "🖼️ (Text from Image)"
                            elif "(Image Description)" in doc_name:
                                icon = "🖼️ (Image Description)"
                            else:
                                icon = "📄"
                            
                            st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                            
                            content = source['content']
                            if len(content) > 500:
                                content = content[:500] + "..."
                            st.code(content)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": result['sources']
                })
                
                # Rerun to update the chat history
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add some spacing at the bottom for the fixed chat input
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

