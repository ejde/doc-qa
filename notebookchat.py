import streamlit as st
import chromadb
from PyPDF2 import PdfReader
import os

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama

# Streamlit App Setup
st.title("Document Q&A with RAG Setup")
st.write("Upload PDF or TXT files to create embeddings and ask questions.")

# Ask user to choose between Google Generative AI, OpenAI, or Local LLM
model_choice = st.selectbox("Choose an AI model:", ["Google Generative AI", "OpenAI", "Local LLM (Ollama)"])

# API Key inputs based on model choice
chat_model = None
embeddings_model = None
if model_choice == "Google Generative AI":
    api_key = st.text_input("Enter your Google Generative AI API key:", type="password")
    if not api_key:
        st.warning("Please enter your Google Generative AI API key to proceed.")
        st.stop()
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', google_api_key=api_key, temperature=0.8)

elif model_choice == "OpenAI":
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    embeddings_model = OpenAIEmbeddings(api_key=api_key, model='text-embedding-ada-002')
    chat_model = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=None, timeout=None, max_retries=2)

elif model_choice == "Local LLM (Ollama)":
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    chat_model = ChatOllama(model='mistral')

# Set up ChromaDB client if not already initialized
if 'chroma_client' not in st.session_state:
    try:
        st.session_state['chroma_client'] = chromadb.Client(settings=chromadb.config.Settings(persist_directory="./chroma_db", anonymized_telemetry=False))
    except ValueError as e:
        st.error(f"An error occurred while setting up ChromaDB client: {e}")
        st.stop()

chroma_client = st.session_state['chroma_client']

# Set up ChromaDB collection
if 'chroma_collection' not in st.session_state:
    try:
        st.session_state['chroma_collection'] = chroma_client.create_collection(name="document_embeddings")
    except Exception as e:
        if 'already exists' in str(e).lower():
            st.session_state['chroma_collection'] = chroma_client.get_collection(name="document_embeddings")
        else:
            st.error("An error occurred while creating or accessing the collection.")
            st.stop()

chroma_collection = st.session_state['chroma_collection']

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
    embedding_id = str(hash(content))
    embeddings = embeddings_model.embed_query(content)
    chroma_collection.add(ids=[embedding_id], documents=[content], embeddings=[embeddings])

def retrieve_relevant_context(query):
    query_embedding = embeddings_model.embed_query(query)
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)
    return [doc[0] if isinstance(doc, list) else doc for doc in results['documents']]

def answer_query_with_context(query, context):
    if model_choice in ["Google Generative AI", "OpenAI"]:
        messages = [
            SystemMessage(content="You are an assistant that answers questions based on the provided context."),
            HumanMessage(content=f"Context: {context}\nQuestion: {query}\nAnswer:")
        ]
        try:
            response = chat_model(messages)
        except Exception as e:
            st.error(f"An error occurred while querying the model: {e}")
            return "Error: Unable to get response."
        return response.content.strip()
    elif model_choice == "Local LLM (Ollama)":
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        try:
            response = chat_model.invoke(prompt)
        except Exception as e:
            st.error(f"An error occurred while querying the local LLM: {e}")
            return "Error: Unable to get response."
        return response.content

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

# Ask Questions - with Chat History
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_query = st.text_input("Ask a question based on the uploaded documents:")
if user_query:
    relevant_contexts = retrieve_relevant_context(user_query)
    combined_context = "\n".join(map(str, relevant_contexts))
    answer = answer_query_with_context(user_query, combined_context)
    st.session_state['chat_history'].append((user_query, answer))

# Display Chat History
for question, response in st.session_state['chat_history']:
    st.write(f"**You:** {question}")
    st.write(f"**Assistant:** {response}")
