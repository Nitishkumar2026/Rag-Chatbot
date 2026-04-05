# 🚀 End-to-End Setup Guide: DataSphere RAG Chatbot

Follow these steps to set up and run the RAG Chatbot for the Amlgo Labs assignment.

---

## 📋 Prerequisites
- **Python 3.9+**: [Download here](https://www.python.org/downloads/)
- **Ollama**: Required for local LLM execution. [Download here](https://ollama.com/)
- **Git**: To clone the repository.

---

## ⚙️ Step 1: Clone & Install Dependencies
Open your terminal and run:
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

# 2. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

---

## 🧠 Step 2: Set Up Local LLM (Ollama)
We use **Mistral-7B** for fast, high-quality local generation.
1. Start the Ollama application.
2. In your terminal, download the Mistral model:
   ```bash
   ollama pull mistral
   ```
3. Ensure the Ollama server is running (usually automatic after opening the app).

---

## 🔐 Step 3: Environment Configuration
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and verify the settings:
   - `LLM_BACKEND=ollama`
   - `OLLAMA_MODEL=mistral`
   - *(Optional)* Add a `GROQ_API_KEY` for ultra-fast cloud inference.

---

## 📦 Step 4: Build the Knowledge Base
You must process the document and build the vector database before launching the app.
```bash
# 1. Clean and Chunk the document
# This processes data/document.txt and saves to chunks/chunks.json
python src/document_processor.py

# 2. Generate Embeddings and Build FAISS Index
# This uses all-MiniLM-L6-v2 to index the chunks in vectordb/
python src/embeddings.py
```

---

## 🚀 Step 5: Launch the Chatbot
Start the Streamlit application:
```bash
streamlit run app.py
```

---

## 🛠️ Troubleshooting
- **Ollama Connection Error**: Make sure the Ollama app is open and `ollama serve` is not blocked by a firewall.
- **Missing index.faiss**: Ensure you ran `python src/embeddings.py` successfully.
- **Torch/CUDA errors**: The current setup uses `faiss-cpu`. If you have a GPU, you can install `faiss-gpu` for faster indexing, but it is not required for this document size.

---
*Developed for Amlgo Labs · Junior AI Engineer Assignment*
