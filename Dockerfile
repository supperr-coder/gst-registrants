FROM gdssingapore/airbase:python-3.13

ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app

COPY --chown=app:app app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app/ ./

USER app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
