FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN git clone --depth 1 \
    https://github.com/thu-sail-lab/Time-RCD.git /tmp/time-rcd && \
    pip install --no-cache-dir --no-deps /tmp/time-rcd && \
    rm -rf /tmp/time-rcd

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    huggingface-hub \
    jaxtyping \
    einops \
    fastapi \
    uvicorn \
    requests \
    prometheus-client
COPY app.py .

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
