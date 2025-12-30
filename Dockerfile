FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV PORT=8080
EXPOSE 8080

# Fly will choose the right process via fly.toml (web vs worker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
