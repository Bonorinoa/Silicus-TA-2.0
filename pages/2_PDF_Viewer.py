# pages/2_PDF_Viewer.py
import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="PDF Viewer", page_icon="📄", layout="wide")

# Get parameters using a simpler approach
course = st.query_params.get("course")
filename = st.query_params.get("filename") 
page_num = st.query_params.get("page", "1")

if not course or not filename:
    st.error("Missing required parameters. Use this page from citation links in chat.")
    st.stop()

# Reconstruct the path from course and filename 
DATA_ROOT = Path(__file__).parents[1] / "data"
pdf_path = DATA_ROOT / course / "pdfs" / filename

if not pdf_path.is_file():
    st.error(f"PDF not found: {filename} in course {course}")
    st.stop()

st.title(f"📄 {filename}")
st.caption(f"Page {page_num} • Course: {course.upper()}")

# Display the PDF
b64 = base64.b64encode(pdf_path.read_bytes()).decode()
components.html(
    f'<iframe src="data:application/pdf;base64,{b64}#page={page_num}" '
    'width="100%" height="800px"></iframe>',
    height=800,
    scrolling=True
)