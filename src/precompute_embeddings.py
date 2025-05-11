from __future__ import annotations
"""Pre‑compute embeddings for a single course or all courses.
Usage:
    python src/precompute_embeddings.py econ167 --api_key $MISTRAL_API_KEY
    python src/precompute_embeddings.py               # processes every course
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.mistral_rag_pipeline import MistralRAGPipeline  # ← fixed import

# ---------------------------------------------------------------------------

def process_course(course_dir: Path, api_key: str) -> None:
    """OCR & embed every PDF in <course_dir>/pdfs, save parquet."""
    slug = course_dir.name
    pdf_dir = course_dir / "pdfs"
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"{pdf_dir} does not exist")

    pipeline = MistralRAGPipeline(api_key)
    pages = pipeline.extract_pages(pdf_dir)
    if not pages:
        print(f"[{slug}] – no PDFs")
        return
    
    for page in pages:
        page["file_path"] = str(pdf)   # add this line

    embeds = pipeline._embed_batch([p["page_content"] for p in pages])
    df = pd.DataFrame(pages)
    df["embedding"] = list(embeds)
    out = course_dir / f"{slug}_pages.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"✓ {slug}: {len(df)} pages saved to {out}")

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("course", nargs="?", help="Course slug (folder)")
    parser.add_argument("--data_root", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--api_key", default=os.getenv("MISTRAL_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("MISTRAL_API_KEY missing")

    if args.course:
        dirs = [args.data_root / args.course]
    else:
        dirs = [d for d in args.data_root.iterdir() if d.is_dir()]

    for d in dirs:
        try:
            process_course(d, args.api_key)
        except Exception as e:
            print(f"✗ {d.name}: {e}")

if __name__ == "__main__":
    main()
