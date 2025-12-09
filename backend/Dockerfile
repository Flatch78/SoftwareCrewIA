FROM python:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# IMPORTANT : écouter sur toutes les interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4242", "--reload"]

