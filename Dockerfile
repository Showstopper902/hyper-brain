# ---------- build hyperbolic CLI ----------
FROM golang:1.22 AS hypercli
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
# hyperbolic-cli repo (you referenced it)
RUN git clone https://github.com/HyperbolicLabs/hyperbolic-cli.git .
RUN go build -o /out/hyperbolic .

# ---------- brain app ----------
FROM python:3.11-slim

WORKDIR /app

# OS deps: ssh + curl + certs + bash
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client ca-certificates curl bash \
  && rm -rf /var/lib/apt/lists/*

# hyperbolic CLI binary
COPY --from=hypercli /out/hyperbolic /usr/local/bin/hyperbolic

# Copy app
COPY . /app

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
