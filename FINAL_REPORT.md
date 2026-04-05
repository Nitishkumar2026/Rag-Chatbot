# 📑 Technical Assignment Report: Fine-Tuned RAG Chatbot
## Candidate: Nitish | Junior AI Engineer
### Amlgo Labs Recruitment Assignment

---

## 🏛️ 1. Project Overview & Architecture
The goal of this assignment was to build an end-to-end Retrieval-Augmented Generation (RAG) chatbot capable of answering complex legal and technical queries from a 10,500+ word document set.

### 🏗️ System Flow
The system is built on a modular "Retriever-Generator" architecture:
1. **Ingestion**: Raw text is cleaned and split into sentence-aware chunks.
2. **Indexing**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in a **FAISS** vector database.
3. **Retrieval**: User queries are transformed into vectors and compared against the FAISS index using Cosine Similarity.
4. **Generation**: Top-K retrieved chunks are injected into a prompt for **Mistral-7B** (Ollama) or **Llama-3** (Groq).
5. **Streaming**: Responses are streamed in real-time to a Streamlit UI for enhanced user experience.

---

## ✂️ 2. Document Preprocessing & Chunking Logic
Efficient retrieval starts with quality chunks.
- **Cleaning**: Removed HTML tags, decorative separators, and normalized whitespaces using `regex`.
- **Chunking Strategy**: Used `RecursiveCharacterTextSplitter` with:
  - **Chunk Size**: 640 characters (~100-120 words).
  - **Chunk Overlap**: 80 characters (~15 words) to maintain semantic continuity across segments.
  - **Separators**: Split at double newlines, periods, and spaces to keep sentences intact.
- **Results**: Generated ~140-150 high-quality segments from the original document.

---

## 🔢 3. Embedding Model & Vector Database Selection
- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence-Transformers). 
  - *Decision Rationale*: High performance-to-size ratio (384-dim), fast inference on CPU, and well-suited for legal/technical English text.
- **Vector DB**: `FAISS` (Facebook AI Similarity Search).
  - *Decision Rationale*: Extremly low latency (~5-10ms), support for in-memory flat indexes, and perfect for local/edge deployments of this scale.

---

## 📝 4. Prompt Engineering & Generation Logic
A strict **System Prompt** ensures the model remains grounded and avoids hallucinations.

### System Prompt Directive:
> "Answer the user's question STRICTLY using the provided context. If the context does not contain sufficient information, say: 'I couldn't find that information in the provided document.'"

### Generation Backend:
- Default: **Mistral-7B-Instruct-v0.2** (Local via Ollama).
- Fallback: **Groq API (Llama-3-8B)** for production-grade speed.

---

## 🧪 5. Evaluation & Query Analysis
I conducted 5 distinct test cases to verify the RAG system's robustness:

| Case | Query | Result | Observation |
| :--- | :--- | :---: | :--- |
| **Fact** | "What is the refund policy?" | ✅ Success | Correctly identified the 30-day window from the text. |
| **Complex** | "How is data encrypted?" | ✅ Success | Combined details from "Data Handling" and "Security" sections. |
| **Reasoning** | "Starter plan API limits?" | ✅ Success | Correctly linked plan type to specific rate limit table. |
| **Grounding** | "Chocolate cake recipe?" | ✅ Pass | Responded with the standard "I don't know" as expected. |
| **Ambiguity** | "What if I break rules?" | ✅ Success | Cited the Termination and Acceptable Use Policy sections. |

---

## 🛡️ 6. Challenges & Solutions
- **Hallucinations**: Solved via strict prompt tuning and forced grounding.
- **Context Length**: Limited `Top-K` to 3 to stay within the model's comfortable 4k token window.
- **Streaming UI**: Implemented `st.empty()` placeholders in Streamlit to mimic a modern ChatGPT-style typing effect.

---
*End of Report*
