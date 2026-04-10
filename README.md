# GST Entity Matcher — Project Structure

```
gst-registrants/
│
├── .claude                     # Claude Code project instructions
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # All configurable values (S3 paths, model, thresholds)
│
├── indexing/                   # One-time job: embed + build FAISS index
│   ├── __init__.py
│   ├── embed.py                # Embed entity names via text-embedding-3-large API
│   ├── build_index.py          # Build FAISS index from embeddings, save to S3
│   └── run_indexing.py         # Entrypoint: orchestrates full indexing pipeline
│
├── matching/                   # Query-time logic: search + re-rank
│   ├── __init__.py
│   ├── search.py               # Load FAISS index, embed query, retrieve candidates
│   ├── rerank.py               # Re-rank candidates with RapidFuzz
│   └── pipeline.py             # End-to-end: query names in → matched results out
│
├── app/                        # Streamlit frontend
│   ├── __init__.py
│   ├── streamlit_app.py        # Main Streamlit app (upload CSV, show results, download)
│   └── utils.py                # Helper functions (CSV parsing, results formatting)
│
├── sagemaker/                  # SageMaker endpoint (for production deployment)
│   ├── inference.py            # model_fn, input_fn, predict_fn, output_fn
│   └── package_model.py        # Script to create model.tar.gz and upload to S3
│
├── deployment/                 # Deployment configs
│   ├── Dockerfile              # Docker image for Airbase deployment
│   ├── airbase.json            # Airbase project config
│   └── .gitlab-ci.yml          # SGTS GitLab CI/CD pipeline for Airbase auto-deploy
│
├── notebooks/                  # Jupyter notebooks for MAESTRO development
│   ├── 01_explore_data.ipynb   # Explore GST entity data from S3
│   ├── 02_run_indexing.ipynb   # Run the one-time embedding + indexing job
│   ├── 03_test_matching.ipynb  # Test queries against the built index
│   └── 04_deploy_endpoint.ipynb# Deploy SageMaker endpoint on MAESTRO
│
└── tests/                      # Basic tests
    ├── test_embed.py           # Test embedding API calls
    ├── test_search.py          # Test FAISS search returns valid results
    └── test_rerank.py          # Test RapidFuzz re-ranking logic
```

## Module Responsibilities

### `config.py`
Single source of truth for all configuration. Other modules import from here.
- S3 bucket name, prefixes
- Embedding model name, dimensions, batch size
- Rate limit settings
- FAISS top-k, final top-n, similarity threshold
- API base URL (from env var)

### `indexing/`
Run once (or whenever the GST reference list updates).
- `embed.py` — handles batched API calls to text-embedding-3-large with rate limiting + retries
- `build_index.py` — takes embeddings array, builds FAISS IndexFlatIP, saves index + metadata to S3
- `run_indexing.py` — orchestrates: load CSV → embed → build index → save to S3

### `matching/`
Used at query time (by both the Streamlit app and the SageMaker endpoint).
- `search.py` — loads FAISS index from S3 (cached in memory), embeds query names, returns top-k candidates with cosine similarity scores
- `rerank.py` — takes FAISS candidates, applies RapidFuzz token_set_ratio + jaro_winkler, sorts by combined score
- `pipeline.py` — single function: `match_entities(query_names) → DataFrame of results`

### `app/`
Streamlit frontend for prototyping and Airbase deployment.
- Single entity text input OR CSV file upload
- Results displayed as interactive table
- Download results as CSV
- Shows match confidence (fuzzy score) with color coding

### `sagemaker/`
For production: wraps matching logic as a SageMaker real-time endpoint.
- `inference.py` — the four SageMaker handler functions
- `package_model.py` — bundles FAISS index + code + requirements into model.tar.gz for SageMaker

### `deployment/`
Deployment artifacts for Airbase + GitLab CI/CD.
- `Dockerfile` — containerizes the Streamlit app
- `airbase.json` — Airbase project configuration
- `.gitlab-ci.yml` — auto-deploy to Airbase on push to main

### `notebooks/`
Step-by-step Jupyter notebooks for running on MAESTRO SageMaker JupyterLab.
These are the "do it interactively" versions of the pipeline — useful for initial setup, debugging, and testing.

## Development Workflow

1. Start in `notebooks/01_explore_data.ipynb` — verify S3 access, inspect the GST data
2. Run `notebooks/02_run_indexing.ipynb` — embed all entities, build + save FAISS index (~20 min)
3. Test in `notebooks/03_test_matching.ipynb` — query the index with sample names, tune thresholds
4. Run Streamlit locally: `streamlit run app/streamlit_app.py` (on MAESTRO or locally)
5. When ready for production: deploy SageMaker endpoint via `notebooks/04_deploy_endpoint.ipynb`
6. Deploy Streamlit to Airbase via GitLab CI/CD or Airbase CLI
