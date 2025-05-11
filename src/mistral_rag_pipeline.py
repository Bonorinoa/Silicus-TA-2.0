# -------------------------------------------------------------
#  Mistral PDF Retrieval‑Augmented Generation (RAG) Pipeline
#  Streamlit‑ready — token‑aware batching, chat‑history aware
# -------------------------------------------------------------
#  requirements.txt excerpt
#  ------------------------
#  streamlit>=1.45
#  mistralai>=1.7
#  mistral-common>=1.5      # tokenizer for accurate token counts
#  pandas numpy scipy
# -------------------------------------------------------------
"""Quick start
-------------
from src.mistral_rag_pipeline import answer_with_rag

ans, pages = answer_with_rag(
    directory_path="data/econ167/pdfs",
    query="What is the analogy used to explain MLE in this lecture?",
    course="econ167",
    chat_history="",
    top_k=10,
    api_key=st.secrets["MISTRAL_API_KEY"],
)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from mistralai import Mistral
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from dotenv import load_dotenv
load_dotenv()

__all__ = ["MistralRAGPipeline", "answer_with_rag"]

# ---------------------------------------------------------------------------
# Helper for accurate token counting
# ---------------------------------------------------------------------------

def _build_request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content=text)])


# ---------------------------------------------------------------------------
class MistralRAGPipeline:
    """End‑to‑end Retrieval‑Augmented Generation pipeline."""

    _MAX_TEXT_TOKENS = 8_000      # single‑text limit (API ~8 192)
    _MAX_BATCH_TOKENS = 15_000    # batch limit (<16 k)

    def __init__(
        self,
        api_key: str,
        chat_model: str = "mistral-medium-latest",
        ocr_model: str = "mistral-ocr-latest",
        embed_model: str = "mistral-embed",
    ) -> None:
        if not api_key:
            raise ValueError("Provide an API key (arg or $MISTRAL_API_KEY).")

        self.client = Mistral(api_key=api_key)
        self.chat_model = chat_model
        self.ocr_model = ocr_model
        self.embed_model = embed_model

        self._tok = MistralTokenizer.v3()
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tok_encode(self, text: str) -> List[int]:
        return self._tok.encode_chat_completion(_build_request(text)).tokens

    def _embed_call(self, texts: List[str]) -> List[np.ndarray]:
        resp = self.client.embeddings.create(model=self.embed_model, inputs=texts)
        return [e.embedding for e in sorted(resp.data, key=lambda e: e.index)]

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        bucket: List[str] = []
        bucket_tok = 0
        vecs: List[np.ndarray] = []

        for t in texts:
            ids = self._tok_encode(t)
            if len(ids) > self._MAX_TEXT_TOKENS:
                ids = ids[: self._MAX_TEXT_TOKENS]
                t = self._tok.decode(ids)
            t_len = len(ids)

            if bucket_tok + t_len > self._MAX_BATCH_TOKENS:
                vecs.extend(self._embed_call(bucket))
                bucket, bucket_tok = [], 0

            bucket.append(t)
            bucket_tok += t_len

        if bucket:
            vecs.extend(self._embed_call(bucket))
        return np.array(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------
    def _upload_pdf(self, pdf_path: str):
        with open(pdf_path, "rb") as f:
            return self.client.files.upload(
                file={"file_name": Path(pdf_path).name, "content": f},
                purpose="ocr",
            )

    def _get_signed_url(self, file_obj):
        return self.client.files.get_signed_url(file_id=file_obj.id).url

    def _ocr_pdf(self, file_obj):
        url = self._get_signed_url(file_obj)
        return self.client.ocr.process(
            model=self.ocr_model,
            document={"type": "document_url", "document_url": url},
        )

    def extract_pages(self, pdf_dir: str | Path) -> List[Dict]:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.is_dir():
            raise FileNotFoundError(pdf_dir)

        pages: List[Dict] = []
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            try:
                f_obj = self._upload_pdf(str(pdf))
                ocr = self._ocr_pdf(f_obj)
                for idx, pg in enumerate(ocr.pages, 1):
                    pages.append(
                        {
                            "filename": pdf.name,
                            "page_number": getattr(pg, "page_number", idx),
                            "page_content": pg.markdown.strip(),
                        }
                    )
            except Exception as exc:
                logging.error("%s failed — %s", pdf.name, exc)
        return pages

    def rank_pages(self, pages: List[Dict], query: str, top_k: int = 10) -> pd.DataFrame:
        if not pages:
            raise ValueError("No pages to rank.")

        page_vecs = self._embed_batch([p["page_content"] for p in pages])
        query_vec = self._embed_batch([query])[0]
        sims = 1 - np.array([cosine(query_vec, v) for v in page_vecs])

        df = pd.DataFrame(pages)
        df["similarity"] = sims
        df = (
            df.sort_values("similarity", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        df.index += 1
        df.insert(0, "rank", df.index)
        return df

    # ------------------------------------------------------------------
    # RAG prompt + generation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(query: str, ctx: pd.DataFrame, history: str, course: str) -> str:
        if not ctx.empty:
            blocks = [f"[{r.filename} – p.{r.page_number}]\n{r.page_content}" for _, r in ctx.iterrows()]
            excerpts = "\n\n---\n".join(blocks)
            excerpt_section = f"\n\nCourse excerpts:\n{excerpts}"
        else:
            excerpt_section = ""

        return (
            "You are **Silicus TA**, a friendly teaching assistant for the course "
            f"*{course.upper()}*. Maintain a helpful, conversational tone.\n\n"
            "**Guidelines**:\n"
            "• If the user's question is clearly about course content, first check the *Course excerpts* section.\n"
            "  – If an answer is present, answer factually and cite pages in brackets.\n"
            "  – If not, say you are unsure and optionally offer general guidance.\n"
            "• If the user is simply chatting (greetings, thanks, etc.), respond naturally without excerpts.\n"
            "• Keep answers concise (≤ 3 short paragraphs), unless requested otherwise.\n"
            "• Use markdown for mathematical notation.\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"User: {query}\n\nAssistant:{excerpt_section}"
        )

    def generate_answer(
        self,
        query: str,
        top_pages: pd.DataFrame,
        course: str,
        chat_history: str = "",
        temperature: float = 0.3,
    ) -> str:
        prompt = self._build_prompt(query, top_pages, chat_history, course)
        resp = self.client.chat.complete(
            model=self.chat_model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    
    def generate_answer_with_links(self, query, top_pages: pd.DataFrame,
                               *, course, chat_history, temperature=0.3):
        # build RAG prompt exactly as before
        answer = self.generate_answer(query, top_pages,
                                    course=course,
                                    chat_history=chat_history,
                                    temperature=temperature)

        # inject HTML links for each page reference
        numbered = {}
        for i, row in enumerate(top_pages.itertuples(), 1):
            if hasattr(row, "file_path"):
                link = (
                    f'<a href="?slide={row.file_path}|{row.page_number}">[{i}]</a>'
                )
            else:                     # fallback → plain citation label
                link = f"[{i}]"
            numbered[f"[{i}]"] = link

        for k, v in numbered.items():
            answer = answer.replace(k, v, 1)
        return answer, numbered



# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def answer_with_rag(
    directory_path: str | Path,
    query: str,
    course: str,
    chat_history: str = "",
    top_k: int = 10,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
) -> Tuple[str, pd.DataFrame]:
    api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
    pipe = MistralRAGPipeline(api_key)
    pages = pipe.extract_pages(directory_path)
    ranked = pipe.rank_pages(pages, query, top_k)
    answer = pipe.generate_answer(query, ranked, course, chat_history, temperature)
    return answer, ranked


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG over PDFs with Mistral")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("course", type=str, help="Course name (folder)")
    parser.add
