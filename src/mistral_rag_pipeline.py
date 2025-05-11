# -------------------------------------------------------------
#  Mistral PDF Retrieval‑Augmented Generation (RAG) Pipeline
#  • Upload & OCR PDFs in a directory                     (Retrieval)
#  • Embed and rank pages by similarity to a query        (Retrieval)
#  • Build a context‑rich prompt and call a chat model    (Generation)
#  • Return a DataFrame of top‑k pages *and* the answer   (RAG)
# -------------------------------------------------------------
#  Tested with Python ≥ 3.10 on Google Colab
# -------------------------------------------------------------

"""
Quick Colab Demo
----------------
!pip -q install mistralai pandas numpy scipy

from mistral_rag_pipeline import answer_with_rag

answer, pages = answer_with_rag(
    directory_path="/content/econ167",
    query="what is the analogy used to explain MLE in this lecture?",
    top_k=10,
    api_key="YOUR_MISTRAL_API_KEY",
    temperature=0.2,
)
print(answer)
pages.head()
"""

# --- Dependencies -----------------------------------------------------------
# %pip -q install mistralai pandas numpy scipy

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from mistralai import Mistral
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------

__all__ = [
    "MistralRAGPipeline",
    "answer_with_rag",
]

# ---------------------------------------------------------------------------
# Helper: accurate token counting -------------------------------------------

def _build_request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content=text)])

# Helper Class ----------------------------------------------------------------

class MistralRAGPipeline:
    """End‑to‑end pipeline: directory → answer + relevant pages."""
    
    # -------- configuration caps -----------------------------------------
    _MAX_TEXT_TOKENS = 8_000     # single input hard limit (API ~8 192)
    _MAX_BATCH_TOKENS = 15_000   # keep <16 k for safety

    def __init__(
        self,
        api_key: str = os.getenv("MISTRAL_API_KEY", ""),
        chat_model: str = "mistral-medium-latest",
        ocr_model: str = "mistral-ocr-latest",
        embed_model: str = "mistral-embed"
    ) -> None:
        if not api_key:
            raise ValueError("Provide an API key (arg or $MISTRAL_API_KEY).")

        self.client = Mistral(api_key=api_key)
        self.chat_model = chat_model
        self.ocr_model = ocr_model
        self.embed_model = embed_model
        self._tok = MistralTokenizer.v3()  # Mistral’s own BPE tokenizer

        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    # --------------------------- Retrieval helpers -----------------------
    def _upload_pdf(self, pdf_path: str):
        with open(pdf_path, "rb") as f:
            return self.client.files.upload(
                file={"file_name": os.path.basename(pdf_path), "content": f},
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
        
    # --------------------------- Embedding helpers -----------------------

    def _tok_encode(self, text: str) -> List[int]:
        return self._tok.encode_chat_completion(_build_request(text)).tokens

    def _embed_call(self, texts: List[str]) -> List[np.ndarray]:
        resp = self.client.embeddings.create(model=self.embed_model, inputs=texts)
        return [e.embedding for e in sorted(resp.data, key=lambda e: e.index)]

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch inputs so that *no* request exceeds token quotas."""
        bucket: List[str] = []
        bucket_tokens = 0
        vectors: List[np.ndarray] = []

        for t in texts:
            ids = self._tok_encode(t)
            if len(ids) > self._MAX_TEXT_TOKENS:  # truncate huge pages
                ids = ids[: self._MAX_TEXT_TOKENS]
                t = self._tok.decode(ids)
            t_len = len(ids)

            if bucket_tokens + t_len > self._MAX_BATCH_TOKENS:
                vectors.extend(self._embed_call(bucket))
                bucket, bucket_tokens = [], 0

            bucket.append(t)
            bucket_tokens += t_len

        if bucket:
            vectors.extend(self._embed_call(bucket))

        return np.array(vectors, dtype=np.float32)

    # --------------------------- Public Retrieval API -------------------
    def extract_pages(self, directory_path: str | Path) -> List[Dict]:
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise FileNotFoundError(directory_path)

        pdf_paths = sorted(p for p in directory_path.glob("*.pdf"))
        if not pdf_paths:
            logging.warning("No PDFs found in %s", directory_path)
            return []

        pages: List[Dict] = []
        for path in pdf_paths:
            try:
                file_obj = self._upload_pdf(str(path))
                ocr = self._ocr_pdf(file_obj)
                for idx, pg in enumerate(ocr.pages, 1):
                    pages.append(
                        {
                            "filename": path.name,
                            "page_number": getattr(pg, "page_number", idx),
                            "page_content": pg.markdown.strip(),
                        }
                    )
            except Exception as exc:
                logging.error("%s failed – %s", path.name, exc)
        return pages

    def rank_pages(self, pages: List[Dict], query: str, top_k: int = 10) -> pd.DataFrame:
        if not pages:
            raise ValueError("No pages to rank.")

        contents = [p["page_content"] for p in pages]
        page_vecs = self._embed_batch(contents)
        query_vec = self._embed_batch([query])[0]
        sims = 1 - np.array([cosine(query_vec, v) for v in page_vecs])

        df = pd.DataFrame(pages)
        df["similarity"] = sims
        df = df.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
        df.index += 1
        df.insert(0, "rank", df.index)
        return df

    # --------------------------- Generation -----------------------------
    @staticmethod
    def _build_prompt(query: str, ctx: pd.DataFrame) -> str:
        blocks = [f"[{r.filename} – p.{r.page_number}]\n{r.page_content}" for _, r in ctx.iterrows()]
        excerpts = "\n\n---\n".join(blocks)
        return (
            "You are a helpful TA. Answer using ONLY the excerpts below. "
            "Cite pages in brackets. If unsure say 'I don't know'.\n\n"  # instructions
            f"Excerpts:\n{excerpts}\n\nUser question: {query}\n\nAssistant:"
        )

    def generate_answer(self, query: str, top_pages: pd.DataFrame, temperature=0.2) -> str:
        prompt = self._build_prompt(query, top_pages)
        resp = self.client.chat.complete(
            model=self.chat_model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Convenience wrapper --------------------------------------------------------

def answer_with_rag(
    directory_path: str | Path,
    query: str,
    top_k: int = 10,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Tuple[str, pd.DataFrame]:
    api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
    pipe = MistralRAGPipeline(api_key)
    pages = pipe.extract_pages(directory_path)
    ranked = pipe.rank_pages(pages, query, top_k)
    answer = pipe.generate_answer(query, ranked, temperature)
    return answer, ranked

# ---------------------------------------------------------------------------
# CLI for local testing ------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG over PDFs with Mistral")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("question", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", type=str, default=os.getenv("MISTRAL_API_KEY", ""))
    args = parser.parse_args()

    ans, df = answer_with_rag(args.pdf_dir, args.question, args.top_k, args.api_key, args.temperature)
    print("Answer:\n", ans)
    print("\nTop pages:\n", df.head())