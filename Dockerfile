# ---------- build hyperbolic CLI (PINNED + correct Go toolchain) ----------
ARG GO_VERSION=1.24.3
FROM golang:${GO_VERSION} AS hypercli

WORKDIR /tmp

# Use the toolchain inside the image (now 1.24.3), don't try to auto-fetch
ENV GOTOOLCHAIN=local

# Pin to a known release tag (do NOT use @latest)
RUN go install github.com/HyperbolicLabs/hyperbolic-cli@v0.0.3


# ---------- runtime ----------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl jq openssh-client \
 && rm -rf /var/lib/apt/lists/*

COPY --from=hypercli /go/bin/hyperbolic /usr/local/bin/hyperbolic

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
