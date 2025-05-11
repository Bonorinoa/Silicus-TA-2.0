import streamlit as st

st.set_page_config(page_title="Silicus TA", page_icon="📚")
st.title("Silicus TA – Econ 167")
st.markdown(
    """
**Welcome!**  
Ask questions about *Econ 167* lecture PDFs and get instant, cited answers.

**How it works**  
1. 📄 We pre‑OCR all lecture PDFs and store page‑level embeddings.  
2. 🔍 When you ask a question, we pull the 10 most relevant pages.  
3. 🤖 Mistral’s chat model answers using those excerpts (RAG).

*No personal data is stored.* You can inspect the source on GitHub and fork it for your own courses!
"""
)
st.page_link("pages/1_Silicus_TA.py", label="👉 Go to the Chat page")
