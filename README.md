Silicus TA 2.0
==============

**A Streamlit‑powered RAG assistant that lets students chat with their course
PDFs, while professors manage those PDFs—and their embeddings—without ever
leaving the browser.**

---

## ✨ Key features
| Role | What you get |
|------|--------------|
| **Student** | • Sidebar course picker<br>• Chat UI with memory; “Hey” works just fine<br>• Answers cite page #s and show an expander with the exact excerpt |
| **Professor** | • Password‑protected **Admin** page<br>• Drag‑and‑drop PDF upload<br>• Duplicate detection & folder‑size guard (300 MB)<br>• Live progress bar during OCR + embedding<br>• One‑click commit to GitHub (no local Git needed)<br>• PDF preview iframe and delete button |

---

## 🖇️ Project structure

silicus-ta-2.0/
├─ .streamlit/ # Cloud config & secrets (not committed)
├─ app.py # Intro / landing page
├─ pages/
│ ├─ 1_Silicus_TA.py # Chat interface
│ └─ 9_Admin.py # Professor console
├─ src/
│ ├─ init.py
│ ├─ mistral_rag_pipeline.py
│ └─ precompute_embeddings.py
└─ data/
└─ <course>/
├─ pdfs/
├─ <course>_pages.parquet
└─ meta.json


---

## 🏁 Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Add your keys to .streamlit/secrets.toml (see example below).

MISTRAL_API_KEY = "sk-..."
ADMIN_PASSWORD  = "super-secret"
GH_TOKEN        = "github_pat_..."
GH_REPO         = "yourname/Silicus-TA-2.0"

🚀 Deploy to Streamlit Cloud
Push to GitHub.

Create a new Streamlit app → point to app.py.

Add the four secrets above.

Click Deploy.

📚 How it works (high level)
Extraction – every page of every PDF is OCR‑ed via mistral‑ocr‑latest.

Embedding – pages are chunked into ≤ 8 K tokens and embedded with
mistral-embed in ≤ 15 K‑token batches.

Retrieval – chat query is embedded on the fly and cosine‑matched against
cached vectors (Parquet → memory), so runtime cost ≈ one embedding per user
message.

Generation – top 10 pages are packed into a prompt with recent chat
history; mistral-medium-latest generates the answer.

🧑‍🎓 Student best practices
Ask one clear question per message for best citations.

Chain questions—the bot remembers the last ~6 turns.

If the bot says “I don’t know,” rephrase or pick a more specific topic.

🧑‍🏫 Professor best practices
Do	Why
Upload slide PDFs before class and click Build embeddings.	Students get instant answers during lecture.
Keep each course under 300 MB.	Prevents hitting GitHub’s 1 GB soft limit.
Use the preview 👁️ to confirm you uploaded the correct slide deck.	Saves token cost of re‑embedding.

✨ Planned improvements
Inline numbered citations ↔ page excerpts

Optional branch‑based PR workflow for course commits

Usage analytics (question heat‑map)

Contributions welcome — open an issue or PR!