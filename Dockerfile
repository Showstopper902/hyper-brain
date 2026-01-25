# Updated Dockerfile (RunPod-only; removes Hyperbolic CLI stage)

FROM python:3.11-slim

# Minimal OS deps for runtime + debugging
RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates curl jq openssh-client  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
