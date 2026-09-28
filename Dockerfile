FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

RUN mkdir -p /models \
    && curl -fsSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx -o /models/silero_vad.onnx

COPY main.py ./main.py
COPY reazonspeech_server ./reazonspeech_server

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9090/health', timeout=3)"

CMD ["python", "main.py"]
