FROM python:3.10-slim

WORKDIR /app

# Dependencies first, so a code change does not reinstall them on every build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py main.py model.joblib ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
