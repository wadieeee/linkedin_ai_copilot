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

