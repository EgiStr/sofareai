# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app


COPY requirements.txt .

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Increase timeout and use cache mount
RUN --mount=type=cache,id=pip,target=/root/.cache/pip,sharing=locked \
    pip install --default-timeout=1000 -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ src/

CMD ["python", "src/main.py"]
