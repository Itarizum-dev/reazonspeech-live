FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY main.py ./main.py
COPY reazonspeech_server ./reazonspeech_server

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9090/health', timeout=3)"

CMD ["python", "main.py"]
