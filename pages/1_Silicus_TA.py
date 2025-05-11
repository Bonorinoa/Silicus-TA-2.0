# pages/1_Silicus_TA.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from src.mistral_rag_pipeline import MistralRAGPipeline

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
st.sidebar.title("Settings")
chosen_course = st.sidebar.selectbox(
    "Select course",
    options=sorted(COURSES),
    index=list(COURSES).index(DEFAULT_COURSE) if COURSES else 0,
    key="course_select",
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

    # ---- d) generation ------------
    answer = pipeline.generate_answer(
        prompt,
        top_pages,
        course=chosen_course,
        chat_history=history,
        temperature=0.2,
    )
    
    with st.expander("📄 Sources"):
        st.markdown(top_pages[['page_number','page_content']].to_markdown())

    # assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)

