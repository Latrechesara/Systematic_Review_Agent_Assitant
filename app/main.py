import time
import streamlit as st
from src.rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="Systematic Review Assistant",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

pipeline = load_rag_pipeline()

st.title("🫁 Systematic Review Assistant")
st.subheader("Multimodal AI for Lung Cancer Literature Search")

with st.sidebar:
    st.header("⚙️ Configuration")
    rag_mode = st.radio(
        "Execution Mode",
        options=["Standard RAG", "Agentic RAG (Tool-Calling)"],
        help="Standard uses direct retrieval; Agentic lets the agent iteratively search and reflect."
    )
    top_k = st.slider("Top K Documents", min_value=1, max_value=10, value=5)
    
    if rag_mode == "Standard RAG":
        search_strategy = st.selectbox("Search Index Type", ["hybrid", "vector", "keyword"])
    else:
        search_strategy = "hybrid"

    st.divider()
    st.markdown("**Dataset:** Medical Abstracts (`rag_db`)")
    st.markdown("**Model:** `gpt-4o-mini`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a medical literature question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing evidence..."):
            start_time = time.time()

            if rag_mode == "Agentic RAG (Tool-Calling)":
                result = pipeline.answer_query_agentic(user_query, top_n=top_k)
            else:
                result = pipeline.answer_query_basic(user_query, top_n=top_k, search_mode=search_strategy)

            elapsed_time = round(time.time() - start_time, 2)

            answer = result.get("answer", "No response generated.")
            citations = result.get("citations", [])
            is_sufficient = result.get("is_sufficient", True)
            retrieved_docs = result.get("context", [])

            st.markdown(answer)
            st.caption(f"⚡ Mode: **{rag_mode}** | Latency: **{elapsed_time}s** | Evidence Sufficient: **{is_sufficient}**")

            if citations:
                st.markdown("### 🏷️ Citations & Takeaways")
                for cite in citations:
                    st.write(f"- **`{cite.get('doc_id')}`**: {cite.get('key_takeaway')}")

            if retrieved_docs:
                with st.expander("📚 Retrieved Paper Context"):
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        doc_id = doc.get("doc_id", "unknown")
                        title = doc.get("title", "Untitled")
                        abstract = doc.get("abstract", doc.get("text", ""))
                        st.markdown(f"**[{idx}] Document ID: `doc_{doc_id}` — {title}**")
                        st.write(abstract)
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})