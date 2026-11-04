# Livestock LLM API (Qwen wrapper). NOT BUILT OR RUN in the development environment (no Docker there):
# treat as a starting point. No GPU or torch needed: the model is reached over HTTPS.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install fastapi "uvicorn[standard]" httpx sqlalchemy psycopg2-binary pydantic pillow python-multipart
# it reads the Vision API tables and verifies its JWTs, and imports livestock_vision.main (torch/ultralytics are only loaded lazily there, so not installed)
COPY src/livestock_llm /app/livestock_llm
COPY src/livestock_vision /app/livestock_vision
COPY shared /app/shared
ENV PYTHONPATH=/app ADVICE_PACK=/app/shared/advice_pack.json VISION_DATA_DIR=/data
# set at run time, never baked into the image: LLM_BASE_URL LLM_MODEL LLM_API_KEY AGRICODERS_JWT_SECRET DATABASE_URL
EXPOSE 8002
CMD ["uvicorn", "livestock_llm.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
