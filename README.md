# LinkedIn AI Copilot

A Streamlit-based AI assistant for LinkedIn that generates **reply suggestions** and **comment suggestions** for posts. The app provides context-aware, sentiment-aligned suggestions using advanced language models and embeddings.

---

## Features

- Analyze sentiment of posts/comments using **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`)  
- Generate semantic embeddings with **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`)  
- Suggest AI-generated replies and comments using **Mistral (Ollama)**  
- Store embeddings in **FAISS** vector database for fast retrieval  
- Interactive **Streamlit UI** for reviewing and editing AI suggestions  

---

## Architecture

```mermaid
flowchart TD
    A[LinkedIn UI<br>(User inputs post/comment)] --> B[services/sentiment.py<br>Analyze sentiment using DistilBERT]
    B --> C[services/embeddings.py<br>Generate semantic embeddings with MiniLM<br>Store in FAISS database]
    C --> D[agents/copilot_agent.py<br>Generate replies/comments using Mistral (Ollama) based on context & sentiment]
    D --> E[Streamlit UI<br>- Display AI suggestions<br>- Show sentiment<br>- User edits & posts]
