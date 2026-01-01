# ---------- build hyperbolic CLI (pinned) ----------
FROM golang:1.24.3 AS hypercli
ARG HYPERBOLIC_CLI_VERSION=v0.0.3

WORKDIR /src
# Clone the official CLI repo and build it
RUN git clone --depth 1 --branch "${HYPERBOLIC_CLI_VERSION}" \
      https://github.com/HyperbolicLabs/hyperbolic-cli.git /src/hyperbolic-cli

WORKDIR /src/hyperbolic-cli
# Build a static-ish binary (best-effort; CGO disabled)
RUN CGO_ENABLED=0 go build -o /out/hyperbolic .


# ---------- runtime ----------
FROM python:3.11-slim

# system deps (ssh + curl + jq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl jq openssh-client \
 && rm -rf /var/lib/apt/lists/*

# copy hyperbolic CLI binary
COPY --from=hypercli /out/hyperbolic /usr/local/bin/hyperbolic

WORKDIR /app

# install python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
