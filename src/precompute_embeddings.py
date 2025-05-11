from pathlib import Path
import pandas as pd
from mistral_rag_pipeline import MistralRAGPipeline
from dotenv import load_dotenv
import os 

load_dotenv()

COURSE = "econ101"
DATA_DIR = Path(__file__).parent.parent / "data" / "econ101"
PDF_DIR  = DATA_DIR / "pdfs"

pipeline = MistralRAGPipeline(os.getenv("MISTRAL_API_KEY"))
pages    = pipeline.extract_pages(PDF_DIR)

# No ranking yet – we just need embeddings
embeds   = pipeline._embed_batch([p["page_content"] for p in pages])

df = pd.DataFrame(pages)
df["embedding"] = list(embeds)      # keep as list-of-floats column
(DATA_DIR / f"{COURSE}_pages.parquet").parent.mkdir(exist_ok=True)
df.to_parquet(DATA_DIR / f"{COURSE}_pages.parquet")
