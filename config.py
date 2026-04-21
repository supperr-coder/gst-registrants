"""
Central configuration for the GST Entity Matcher.

Non-secret values are defined here as constants.
Secrets (API keys, credentials) are loaded from environment variables — set them in .env.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
S3_BUCKET = "sst-s3-gvt-agml-prodizna-d-andnwekll2vd-bucket"
S3_GST_PREFIX = "gst-registrants/"
S3_GST_FILE = f"{S3_GST_PREFIX}registered_names.csv"  # update to actual filename
S3_EMBEDDINGS_PREFIX = "gst-matching/embeddings/"
S3_CHECKPOINT_PREFIX = "gst-matching/checkpoints/"
S3_FAISS_KEY = f"{S3_EMBEDDINGS_PREFIX}gst_faiss.index"
S3_METADATA_KEY = f"{S3_EMBEDDINGS_PREFIX}gst_metadata.parquet"
S3_CONFIG_KEY = f"{S3_EMBEDDINGS_PREFIX}config.json"

# ---------------------------------------------------------------------------
# Embedding API (secrets loaded from .env)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024  # reduced from 3072 default for efficiency
EMBEDDING_BATCH_SIZE = 100
RATE_LIMIT_RPM = 50  # requests per minute

# ---------------------------------------------------------------------------
# FAISS / Matching
# ---------------------------------------------------------------------------
FAISS_TOP_K = 20       # candidates retrieved from FAISS per query
MATCH_TOP_N = 5        # final results returned per query
SIMILARITY_THRESHOLD = 0.70  # minimum cosine similarity to include a match

# ---------------------------------------------------------------------------
# Local temp paths (ephemeral storage on SageMaker)
# ---------------------------------------------------------------------------
TMP_FAISS_PATH = "/tmp/gst_faiss.index"
TMP_METADATA_PATH = "/tmp/gst_metadata.parquet"
