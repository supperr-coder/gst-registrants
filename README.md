# GST Entity Matcher — Project Structure

```
gst-registrants/
│
├── .claude                     # Claude Code project instructions
├── .gitignore                  # Git ignore rules (.env, __pycache__, etc.)
├── .env.example                # Template for environment variables (copy to .env)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Non-secret config (S3 paths, model, thresholds)
│
├── indexing/                   # One-time job: embed + build FAISS index
│   ├── __init__.py
│   ├── embed.py                # Embed entity names via text-embedding-3-large API
│   ├── build_index.py          # Build FAISS index from embeddings, save to S3
│   └── run_indexing.py         # Entrypoint: orchestrates full indexing pipeline
│
├── matching/                   # Query-time logic: embed + search
│   ├── __init__.py
│   ├── search.py               # Load FAISS index, embed query, retrieve candidates
│   └── pipeline.py             # End-to-end: query names in → matched results out
│
├── app/                        # Streamlit frontend (deployed on Airbase)
│   ├── __init__.py
│   ├── api_client.py           # HTTP client that calls the SageMaker endpoint
│   ├── streamlit_app.py        # Main Streamlit app (upload CSV, show results, download)
│   └── utils.py                # Helper functions (CSV parsing, results formatting)
│
├── sagemaker/                  # SageMaker endpoint (for production deployment)
│   ├── inference.py            # model_fn, input_fn, predict_fn, output_fn
│   └── package_model.py        # Script to create model.tar.gz and upload to S3
│
├── deployment/                 # Deployment configs (Airbase)
│   ├── Dockerfile              # Docker image for Airbase (only copies app/)
│   ├── requirements.txt        # Slim deps for Airbase (no faiss/openai/boto3)
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
    └── test_search.py          # Test FAISS search returns valid results
```

## Module Responsibilities

### `config.py`
Single source of truth for non-secret configuration. Other modules import from here.
- S3 bucket name, prefixes
- Embedding model name, dimensions, batch size
- Rate limit settings
- FAISS top-k, final top-n, similarity threshold
- API base URL (read from env var — the actual value lives in `.env`)

### `indexing/`
Run once (or whenever the GST reference list updates).
- `embed.py` — handles batched API calls to text-embedding-3-large with rate limiting + retries
- `build_index.py` — takes embeddings array, builds FAISS IndexFlatIP, saves index + metadata to S3
- `run_indexing.py` — orchestrates: load CSV → embed → build index → save to S3

### `matching/`
Used at query time by the SageMaker endpoint (not by the Streamlit app on Airbase).
- `search.py` — loads FAISS index from S3 (cached in memory), embeds query names, returns top-k candidates ranked by cosine similarity
- `pipeline.py` — single function: `match_entities(query_names) → DataFrame of results`

### `app/`
Streamlit frontend deployed on Airbase. Calls the SageMaker endpoint — no local FAISS, embedding API, or S3 access.
- `api_client.py` — HTTP client that POSTs entity names to the SageMaker endpoint URL (set via `SAGEMAKER_ENDPOINT_URL` env var)
- Single entity text input OR CSV file upload
- Results displayed as interactive table
- Download results as CSV

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
