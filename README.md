# Document Q&A RAG App - Setup Guide

This guide provides instructions for setting up and running the Document Q&A Retrieval-Augmented Generation (RAG) Streamlit app using Ollama for local LLM support - the typical RAG 101 project.

## Prerequisites

- **Operating System**: macOS or Linux (Windows users need WSL).
- **Python**: Version 3.8 or higher.
- **Streamlit**: Make sure Streamlit is installed (`pip install streamlit`).

## Installation Steps

### Step 1: Install Ollama

1. **macOS**: Install via Homebrew:
   ```sh
   brew install ollama
   ```

2. **Linux**: Download from the [Ollama GitHub Releases page](https://github.com/ollama/ollama/releases), extract, and move it to your PATH.
   ```sh
   tar -xzf ollama-linux-x86_64.tar.gz
   sudo mv ollama /usr/local/bin/
   ```

3. **Windows (via WSL)**: Install WSL and follow the Linux instructions above.

### Step 2: Download Models

Download the `mistral` and the `text-embedding-ada-002` models:
```sh
ollama pull mistral
ollama pull text-embedding-ada-002
```

## Step 3: Install Python Dependencies

Navigate to the app's directory and install the required Python packages:
```sh
pip install -r requirements.txt
```
Ensure that you have `streamlit`, `langchain`, and `chromadb` installed.

## Running the Streamlit App

After setting up Ollama and downloading the required models, run the app using Streamlit:
```sh
streamlit run doc-qa.py
```
## Additional Resources

- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Documentation](https://ollama.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

