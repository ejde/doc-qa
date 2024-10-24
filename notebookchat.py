import streamlit as st
import chromadb
from PyPDF2 import PdfReader
import os

# Import the necessary langchain modules with pydantic v2 compatibility
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Streamlit App Setup
st.title("Document Q&A with RAG Setup")
st.write("Upload PDF or TXT files to create embeddings and ask questions.")

# Ask user to choose between Google Generative AI and OpenAI
model_choice = st.selectbox("Choose an AI model:", ["Google Generative AI", "OpenAI"])

# API Key inputs based on model choice
if model_choice == "Google Generative AI":
    api_key = st.text_input("Enter your Google Generative AI API key:", type="password")
    if not api_key:
        st.warning("Please enter your Google Generative AI API key to proceed.")
        st.stop()
    
    google_genai_ef = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', google_api_key=api_key, temperature=0.8)

elif model_choice == "OpenAI":
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    
    openai_ef = OpenAIEmbeddings(openai_api_key=api_key)
    chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0.8)

# Set up ChromaDB client
chroma_client = chromadb.Client()
try:
    chroma_collection = chroma_client.create_collection(name="document_embeddings")
except Exception as e:
    if 'already exists' in str(e).lower():
        chroma_collection = chroma_client.get_collection(name="document_embeddings")
    else:
        st.error("An error occurred while creating or accessing the collection.")
        st.stop()

uploaded_files = st.file_uploader("Upload your PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Helper Functions
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def create_embeddings(content):
    embedding_id = str(hash(content))  # Generate a unique ID for each content
    if model_choice == "Google Generative AI":
        embeddings = google_genai_ef.embed_query(content)
    elif model_choice == "OpenAI":
        embeddings = openai_ef.embed_query(content)
    chroma_collection.add(ids=[embedding_id], documents=[content], embeddings=[embeddings])

def retrieve_relevant_context(query):
    if model_choice == "Google Generative AI":
        query_embedding = google_genai_ef.embed_query(query)
    elif model_choice == "OpenAI":
        query_embedding = openai_ef.embed_query(query)
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)
    return [doc[0] if isinstance(doc, list) else doc for doc in results['documents']]

def answer_query_with_context(query, context):
    messages = [
        SystemMessage(content="You are an assistant that answers questions based on the provided context."),
        HumanMessage(content=f"Context: {context}\nQuestion: {query}\nAnswer:")
    ]
    response = chat_model(messages)
    return response.content.strip()

# Process Uploaded Files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text_content = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file format.")
            continue
        
        create_embeddings(text_content)
        st.success(f"Embeddings created for {uploaded_file.name}")

# Ask Questions
user_query = st.text_input("Ask a question based on the uploaded documents:")
if user_query:
    relevant_contexts = retrieve_relevant_context(user_query)
    combined_context = "\n".join(map(str, relevant_contexts))
    answer = answer_query_with_context(user_query, combined_context)
    st.write(answer)
