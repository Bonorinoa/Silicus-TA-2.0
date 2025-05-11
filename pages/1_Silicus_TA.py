# pages/1_Silicus_TA.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from src.mistral_rag_pipeline import MistralRAGPipeline

import base64, streamlit.components.v1 as components

# --------------------------------------------------------------------- #
# ─── 0. Configuration ──────────────────────────────────────────────── #
st.set_page_config(page_title="Silicus TA", page_icon="💬")

DATA_ROOT = Path(__file__).parents[1] / "data"      # data/<course>/
DEFAULT_COURSE = "econ167"                          # fallback

def discover_courses(root: Path) -> dict[str, Path]:
    """Return {course_name: parquet_path} for every existing store."""
    courses = {}
    for p in root.glob("*/*_pages.parquet"):
        course = p.parent.name          # folder name == course
        courses[course] = p
    return courses

COURSES = discover_courses(DATA_ROOT)
if DEFAULT_COURSE not in COURSES and COURSES:
    DEFAULT_COURSE = sorted(COURSES)[0]

# --------------------------------------------------------------------- #
# ─── 1. Sidebar – course selector ───────────────────────────────────── #
with st.sidebar.expander("👤 User settings", expanded=True):
    st.markdown("**Select course**")
    chosen_course = st.selectbox(
        "Course",
        options=sorted(COURSES),
        index=sorted(COURSES).index(DEFAULT_COURSE),
    )

    st.markdown("**Prompt Template**")
    st.markdown(
        "You can use the following template to ask questions:\n"
        "```\n"
        "<goal>\nDescribe the task for the LLM\n</goal>\n"
        "<guidelines>\nDescsribe any formatting request or any sort of guidelines the LLM must be aware of\n</guidelines>\n"
        "<context>\nDecribe addition context for the LLM to reference, like what have you tried or read already.</context>\n"
        "```"
    )


# Reset chat if the user switched courses
if "active_course" not in st.session_state or st.session_state.active_course != chosen_course:
    st.session_state.active_course = chosen_course
    st.session_state.messages = []        # wipe chat history

# --------------------------------------------------------------------- #
# ─── 2. Load embeddings for the chosen course (cached) ─────────────── #
@st.cache_resource(show_spinner="Loading embeddings …")
def load_pipeline_and_df(course: str):
    parquet_path = COURSES[course]
    df = pd.read_parquet(parquet_path)
    api_key = st.secrets["MISTRAL_API_KEY"]
    pipeline = MistralRAGPipeline(api_key)
    return pipeline, df

if not COURSES:
    st.error("No course stores found in data/. Ask admin to upload PDFs.")
    st.stop()

pipeline, df_pages = load_pipeline_and_df(chosen_course)

# --------------------------------------------------------------------- #
# ─── 3. Chat UI  ────────────────────────────────────────────────────── #
st.title(f"Silicus TA • {chosen_course.upper()}")

# Replay previous turns
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question …")
if prompt:
    # ---- a) store + echo user ----
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # ---- b) retrieval -------------
    q_vec = pipeline._embed_batch([prompt])[0]
    sims = 1 - np.array([
        (np.dot(q_vec, emb) /
         (np.linalg.norm(q_vec) * np.linalg.norm(emb)))
        for emb in df_pages["embedding"]
    ])
    top_idx = sims.argsort()[:10]
    top_pages = df_pages.iloc[top_idx].copy()
    top_pages["similarity"] = 1 - sims[top_idx]

    # ---- c) build chat‑history string (last 6 turns) ----
    history = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    # ----- d) generation -----
    answer, numbered_sources = pipeline.generate_answer_with_links(
        prompt, top_pages, course=chosen_course, chat_history=history, temperature=0.2
    )

    # Render assistant answer with live links
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer, unsafe_allow_html=True)

    # Modal handler (one per citation click)
    clicked = st.query_params.get("slide")
    if clicked:
        path, page_num = clicked.split("|")
        pdf_path = Path(path)
        if pdf_path.is_file():
            b64 = base64.b64encode(pdf_path.read_bytes()).decode()
            with st.modal(f"{pdf_path.name} – page {page_num}"):
                components.html(
                    f'<iframe src="data:application/pdf;base64,{b64}#page={page_num}" '
                    'width="100%" height="90%"></iframe>', height=700
                )
            # clear query param so refresh doesn’t reopen
            st.query_params.clear()


