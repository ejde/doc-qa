# Document Q&A RAG App - Setup Guide

This guide provides instructions for setting up and running the Document Q&A Retrieval-Augmented Generation (RAG) Streamlit app using Ollama for local LLM support.

## Prerequisites

- **Operating System**: macOS or Linux (Windows users need WSL).
- **Python**: Version 3.8 or higher.
- **Streamlit**: Make sure Streamlit is installed (`pip install streamlit`).

## Installation Steps

### Step 1: Install Ollama

1. **macOS**: Install via Homebrew:
   ```bash
   brew install ollama
   ```

2. **Linux**: Download from the [Ollama GitHub Releases page](https://github.com/ollama/ollama/releases), extract, and move it to your PATH.
   ```bash
   tar -xzf ollama-linux-x86_64.tar.gz
   sudo mv ollama /usr/local/bin/
   ```

3. **Windows (via WSL)**: Install WSL and follow the Linux instructions above.

### Step 2: Verify Installation

Run to check installation:
```bash
ollama --version
```

### Step 3: Download Models

Download the `mistral` model (or any other model you need):
```bash
ollama pull mistral
```

## Step 4: Install Python Dependencies

Navigate to the app's directory and install the required Python packages:
```bash
pip install -r requirements.txt
```
Ensure that you have `streamlit`, `langchain`, and `chromadb` installed.

## Running the Streamlit App

After setting up Ollama and downloading the required models, run the app using Streamlit:
```bash
streamlit run app.py
```
Replace `app.py` with the name of your main Python file.

## Using Ollama in Your Streamlit App

- Ensure Ollama is installed and models are downloaded.
- This app uses the `ChatOllama` integration from LangChain to provide local LLM functionality.
- Ollama is used for querying local language models to answer questions based on uploaded documents.

## Additional Resources

- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Documentation](https://ollama.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

