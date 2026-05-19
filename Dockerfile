# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Cài dependencies hệ thống tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Cài runtime libs (không cần build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy packages đã cài từ stage builder
COPY --from=builder /install /usr/local

# Copy source code (bỏ qua file nhạy cảm — dùng .dockerignore)
COPY . .

# Tạo thư mục output
RUN mkdir -p output learning/models

# Non-root user
RUN useradd -m floodai && chown -R floodai:floodai /app
USER floodai

# ── Mặc định: CLI help ────────────────────────────────────────────────────────
CMD ["python", "main.py", "--help"]

# ── Để chạy web app: ──────────────────────────────────────────────────────────
# docker run --env-file .env -p 5000:5000 flood-pipeline python app.py

# ── Để chạy pipeline: ─────────────────────────────────────────────────────────
# docker run --env-file .env -v /data/floods:/data flood-pipeline \
#   python main.py --config configs/cpu.yaml --input /data --safe
