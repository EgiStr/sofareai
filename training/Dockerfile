# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

COPY training/requirements.txt .
COPY packages/ /packages/

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/home/egistr/google-cloud-sdk/bin:/home/egistr/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/home/egistr/.config/Code/User/globalStorage/github.copilot-chat/copilotCli:/home/egistr/google-cloud-sdk/bin:/home/egistr/.nvm/versions/node/v22.8.0/bin:/home/egistr/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/egistr/snap/code/191/.local/share/coursier/bin:/home/egistr/snap/code/191/.local/share/coursier/bin:/home/egistr/.vscode/extensions/ms-python.debugpy-2025.16.0-linux-x64/bundled/scripts/noConfigScripts"

# Increase timeout and use cache mount
RUN --mount=type=cache,id=pip,target=/root/.cache/pip,sharing=locked     pip install --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cpu &&     pip install --default-timeout=1000 -r requirements.txt &&     pip install /packages/sofare_common

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:/home/egistr/google-cloud-sdk/bin:/home/egistr/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/home/egistr/.config/Code/User/globalStorage/github.copilot-chat/copilotCli:/home/egistr/google-cloud-sdk/bin:/home/egistr/.nvm/versions/node/v22.8.0/bin:/home/egistr/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/egistr/snap/code/191/.local/share/coursier/bin:/home/egistr/snap/code/191/.local/share/coursier/bin:/home/egistr/.vscode/extensions/ms-python.debugpy-2025.16.0-linux-x64/bundled/scripts/noConfigScripts"

COPY training/src/ src/

CMD ["python", "src/train.py"]
