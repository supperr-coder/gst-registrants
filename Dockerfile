FROM registry.sgts.gitlab-dedicated.com/innersource/sgts/runtime/airbase/images/gdssingapore/airbase:python-3.13

ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app

COPY --chown=app:app app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app/ ./

USER app

EXPOSE 3000

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-3000} --server.address=0.0.0.0 --server.headless=true"]
