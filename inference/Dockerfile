FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 COOKIE_SECURE=1 PORT=10000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_runtime.txt requirements_torch.txt requirements_web.txt ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check --no-deps -r requirements_runtime.txt \
    && python -m pip install --no-cache-dir --disable-pip-version-check --no-deps --index-url https://download.pytorch.org/whl/cpu -r requirements_torch.txt \
    && python -m pip install --no-cache-dir --disable-pip-version-check -r requirements_web.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser assets ./assets
COPY --chown=appuser:appuser source_app ./source_app
COPY --chown=appuser:appuser samples ./samples
COPY --chown=appuser:appuser demo_engine.py runtime_config.json self_test.py web_app.py index.html input_template.csv example_new_session.csv ./
RUN python self_test.py --output /tmp/build-self-test.json
USER appuser
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --no-access-log --limit-concurrency 16"]
