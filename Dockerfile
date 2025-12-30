FROM python:3.11-slim

WORKDIR /app

# Copy everything first (prevents weird context/path issues)
COPY . /app

# Install deps
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
