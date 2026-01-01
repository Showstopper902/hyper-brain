# ---------- build hyperbolic CLI (PINNED COMMIT, Go 1.24.x) ----------
FROM golang:1.24 AS hypercli
WORKDIR /src

# git is needed to pin an exact commit
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pin to a specific commit for determinism (from the module pseudo-version you saw)
ARG HYPERBOLIC_CLI_COMMIT=5d3f40498c8c

RUN git clone https://github.com/HyperbolicLabs/hyperbolic-cli.git . \
 && git checkout "${HYPERBOLIC_CLI_COMMIT}" \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/hyperbolic .

# ---------- runtime ----------
FROM python:3.11-slim

# system deps (ssh + curl + jq) used by your worker
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl jq openssh-client \
 && rm -rf /var/lib/apt/lists/*

# copy hyperbolic CLI binary
COPY --from=hypercli /out/hyperbolic /usr/local/bin/hyperbolic

WORKDIR /app

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
