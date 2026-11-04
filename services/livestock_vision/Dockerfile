# Livestock Vision API (Service A). NOT BUILT OR RUN in the development environment (no Docker/GPU there):
# treat as a starting point and test the image on the target L4 server.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_AUTOINSTALL=False YOLO_CONFIG_DIR=/tmp/yolo
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CUDA build of torch; pick the index matching the server's driver (cu121 shown)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install "ultralytics==8.4.191" polars polars-runtime-32 fastapi "uvicorn[standard]" python-multipart \
    sqlalchemy psycopg2-binary pillow pyyaml
COPY src/livestock_vision /app/livestock_vision
COPY shared /app/shared
ENV PYTHONPATH=/app VISION_WEIGHTS=/models/best.pt VISION_DATA_DIR=/data
# the labels path default is repo-relative; inside the image the shared files live at /app/shared
ENV LABELS_PATH=/app/shared/labels.json
EXPOSE 8000
# ONE worker on purpose: the model lives in this process and is called from a single dedicated thread
CMD ["uvicorn", "livestock_vision.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
