FROM python:3.12-slim

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/ml/requirements.txt -r /app/backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend

ENV PYTHONPATH=/app/ml
EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
