import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load HuggingFace sentence embeddings"""
    return HuggingFaceEmbeddings(model_name=model_name)

def init_vectorstore(docs=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     persist_dir="data/faiss_index"):
    """
    Initialize FAISS vectorstore.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embeddings(embedding_model)
    index_path = os.path.join(persist_dir, "index.faiss")

    if docs:  # Build new index
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        vectordb = FAISS.from_texts(texts, embeddings, metadatas=metas)
        vectordb.save_local(persist_dir)
        return vectordb
    elif os.path.exists(index_path):  # Load existing index
        return FAISS.load_local(persist_dir, embeddings,
                                allow_dangerous_deserialization=True)
    else:
        return None

def add_to_vectorstore(vectordb, docs, persist_dir="data/faiss_index"):
    """Add new docs to FAISS index and save."""
    if not docs:
        return vectordb

    texts = [d["text"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]

    if vectordb is None:
        vectordb = FAISS.from_texts(texts, get_embeddings(), metadatas=metas)
    else:
        new_db = FAISS.from_texts(texts, vectordb.embedding_function, metadatas=metas)
        vectordb.merge_from(new_db)

    vectordb.save_local(persist_dir)
    return vectordb

def search_similar(vectordb, query: str, k: int = 3):
    """Search for similar documents in FAISS"""
    if vectordb is None or not query.strip():
        return []
    return vectordb.similarity_search(query, k=k)
