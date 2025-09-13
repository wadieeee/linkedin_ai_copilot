import streamlit as st
import yaml
from agents.copilot_agent import load_copilot_agent, generate_replies, load_comment_agent, generate_comments
from services.embeddings import init_vectorstore, add_to_vectorstore, search_similar

st.set_page_config(page_title="LinkedIn AI Copilot", layout="wide")

# --------- Load config ----------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

CHROMA_DIR = CFG["chroma_dir"]
DEFAULT_TONE = CFG.get("tone", "professional")
SIMILAR_K = int(CFG.get("similar_k", 3))

st.title("🤖 LinkedIn AI Copilot")

# --------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    model_tag = st.text_input("Ollama Model", value=CFG["model"])
    tone = st.selectbox("Default tone",
                        ["professional", "casual", "inspiring", "friendly"],
                        index=["professional","casual","inspiring","friendly"].index(DEFAULT_TONE))
    st.caption("Change model in config.yaml or here.")

    if st.button("🔄 Reset Index"):
        st.session_state.vectordb = None
        st.success("Vector DB reset. Re-index to continue.")

# --------- Vector store ----------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = init_vectorstore(persist_dir=CHROMA_DIR,
                                                 embedding_model=CFG["embedding_model"])

# --------- Manual Playground ---------
st.subheader("✍️ Manual Playground")
manual_post = st.text_area("LinkedIn Post (context)", height=150,
                           placeholder="Paste a post you're responding to...")
manual_comment = st.text_area("Comment to reply to", height=150,
                              placeholder="Paste a comment...")
play_tone = st.selectbox("Tone",
                         ["professional", "casual", "inspiring", "friendly"],
                         index=["professional","casual","inspiring","friendly"].index(DEFAULT_TONE),
                         key="play_tone")

# Automatically index manual post
if manual_post.strip():
    doc = [{"text": manual_post.strip(), "metadata": {"type": "post"}}]
    st.session_state.vectordb = add_to_vectorstore(st.session_state.vectordb, doc,
                                                   persist_dir=CHROMA_DIR)

# Generate replies
if st.button("💡 Generate Replies"):
    if not manual_post or not manual_comment:
        st.error("Please enter both a post and a comment.")
    else:
        similar_docs = search_similar(st.session_state.vectordb, manual_comment, k=SIMILAR_K)
        similar_context = "\n\n".join([d.page_content for d in similar_docs]) if similar_docs else ""

        agent = load_copilot_agent(model=model_tag,
                                   max_new_tokens=CFG["max_new_tokens"],
                                   temperature=CFG["temperature"],
                                   top_p=CFG["top_p"])
        replies = generate_replies(agent, post=manual_post, comment=manual_comment,
                                   sentiment=None, tone=play_tone,
                                   context=similar_context)

        st.subheader("✨ Suggested Replies")
        for i, r in enumerate(replies, 1):
            st.text_area(f"Option {i}", value=r.strip(), height=80,
                         key=f"manual_suggestion_{i}")

# --------- Suggest comments for another post ---------
st.subheader("💡 Suggest Comment for Another Post")
other_post = st.text_area("Another LinkedIn Post", height=150,
                          placeholder="Paste a different post here...")

if st.button("💬 Generate Suggestions for Other Post"):
    if other_post.strip():
        # Index the other post
        doc = [{"text": other_post.strip(), "metadata": {"type": "post"}}]
        st.session_state.vectordb = add_to_vectorstore(st.session_state.vectordb, doc,
                                                       persist_dir=CHROMA_DIR)

        similar_docs = search_similar(st.session_state.vectordb, other_post, k=SIMILAR_K)
        similar_context = "\n\n".join([d.page_content for d in similar_docs]) if similar_docs else ""

        comment_agent = load_comment_agent(model=model_tag,
                                          max_new_tokens=CFG["max_new_tokens"],
                                          temperature=CFG["temperature"],
                                          top_p=CFG["top_p"])
        suggested_comments = generate_comments(comment_agent, post=other_post,
                                               tone=play_tone, context=similar_context)

        st.subheader("✨ Suggested Comments")
        for i, c in enumerate(suggested_comments, 1):
            st.text_area(f"Option {i}", value=c.strip(), height=80,
                         key=f"other_post_suggestion_{i}")
    else:
        st.error("Please enter a post to generate suggestions.")
