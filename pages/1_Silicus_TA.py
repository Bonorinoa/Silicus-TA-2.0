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
                # Create a temporary link to the PDF
                st.session_state.pdf_view = {
                    "path": pdf_path,
                    "page": page_num
                }
                
                # Open PDF in new tab
                b64 = base64.b64encode(pdf_path.read_bytes()).decode()
                pdf_url = f"data:application/pdf;base64,{b64}#page={page_num}"
                
                with st.popover(f"PDF Source: {pdf_path.name}, Page {page_num}", use_container_width=True):
                    st.link_button("📄 Open PDF in new tab", pdf_url, use_container_width=True)
                    
                    # Alternative inline view
                    with st.expander("Or view inline"):
                        components.html(
                            f'<embed src="data:application/pdf;base64,{b64}#page={page_num}" '
                            'width="100%" height="500" type="application/pdf">',
                            height=520
                        )
                
                # Clear query param so refresh doesn't reopen
                st.query_params.clear()


        # Add sources expander inside the assistant's message
        with st.expander("📚 Sources", expanded=False):
            st.markdown("### Referenced Excerpts")
            
            for i, (_, row) in enumerate(top_pages.iterrows(), 1):
                filename = row['filename']
                page_num = row['page_number']
                similarity = f"{row['similarity']:.2f}"
                excerpt = row['page_content']
                
                st.markdown(f"**[{i}] {filename} (Page {page_num}) · Similarity: {similarity}**")
                st.markdown(f"```\n{excerpt}\n```")
                
                # Add unique key to each download button
                pdf_path = DATA_ROOT / chosen_course / "pdfs" / filename
                if pdf_path.exists():
                    with open(pdf_path, "rb") as file:
                        st.download_button(
                            label=f"Download {filename}",
                            data=file,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"download_{i}_{filename}"  # Add unique key
                        )
                st.markdown("---")
        
        
        # Add at the bottom of the file, after generating the answer
        # Make sure we're displaying the sources expander
        #with st.expander("📚 Sources", expanded=False):
        #    from tabulate import tabulate
        #    
        #    # Format sources table with clickable links
        #    table_data = []
        #    for i, (_, row) in enumerate(top_pages.iterrows(), 1):
        #        filename = row['filename']
        #        page_num = row['page_number']
        #        similarity = f"{row['similarity']:.2f}"
        #        excerpt = row['page_content'][:100] + "..." 
        #        
        #        # Create a link for each source
        #        link = f"[View PDF](/PDF_Viewer?course={chosen_course}&filename=#{filename}&page={page_num})"
        #        
        #        table_data.append([i, filename, page_num, similarity, excerpt, #link])
        #    
        #    headers = ["#", "Document", "Page", "Similarity", "Excerpt", "Link"]
        #    st.markdown(tabulate(table_data, headers, tablefmt="pipe"))