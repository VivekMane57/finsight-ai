import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="FinSight AI",
    layout="wide"
)

st.title("FinSight AI")
st.caption("Financial Intelligence Copilot using Hybrid RAG + Azure OpenAI")

tab1, tab2 = st.tabs(["Upload Document", "Chat with Document"])

with tab1:
    st.header("Upload Financial Document")

    uploaded_file = st.file_uploader(
        "Upload annual report PDF",
        type=["pdf"]
    )

    if st.button("Process Document"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }

            with st.spinner("Processing document..."):
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files=files
                )

            if response.status_code == 200:
                data = response.json()
                st.success("Document indexed successfully.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Characters", data["characters"])
                col2.metric("Chunks", data["chunks"])
                col3.metric("Embedding Dim", data["embedding_dimension"])

                st.json(data)
            else:
                st.error(response.text)

with tab2:
    st.header("Ask Financial Questions")

    query = st.text_area(
        "Ask a question",
        value="What are the major business risks?"
    )

    if st.button("Ask FinSight AI"):
        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{API_URL}/chat/query",
                json={"query": query}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Retrieval Details")
            st.write("Method:", data["retrieval_method"])
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(data["retrieved_chunks"], 1):
                    st.markdown(f"### Source Chunk {i}")
                    st.write(chunk)
        else:
            st.error(response.text)