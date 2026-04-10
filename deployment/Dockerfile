FROM python:3.10-slim

WORKDIR /app

# Use the slim Airbase requirements (no faiss/openai/boto3)
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what the Streamlit app needs
COPY app/ ./app/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
