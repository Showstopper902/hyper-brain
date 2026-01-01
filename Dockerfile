# ---------- build hyperbolic CLI (PINNED) ----------
FROM golang:1.24.3 AS hypercli
WORKDIR /tmp

# Pin to a known release tag (don’t use @latest)
RUN go install github.com/HyperbolicLabs/hyperbolic-cli@v0.0.3

# ---------- runtime ----------
FROM python:3.11-slim

# system deps (ssh + curl + jq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl jq openssh-client \
 && rm -rf /var/lib/apt/lists/*

# copy hyperbolic CLI binary
COPY --from=hypercli /go/bin/hyperbolic /usr/local/bin/hyperbolic

WORKDIR /app

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
