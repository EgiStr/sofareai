# Installation

This guide covers installation for development and production environments.

## Docker Installation (Recommended)

The easiest way to run SOFARE-AI is using Docker Compose.

### Prerequisites

=== "Ubuntu/Debian"

    ```bash
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Install Docker Compose
    sudo apt-get install docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    ```

=== "macOS"

    ```bash
    # Install Docker Desktop
    brew install --cask docker
    
    # Or download from https://www.docker.com/products/docker-desktop
    ```

=== "Windows"

    1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
    2. Enable WSL 2 backend
    3. Restart your computer

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/sofareai/sofareai.git
cd sofareai

# Create environment file
cp .env.example .env

# Build containers
make build

# Start the stack
make up
```

### Verify Installation

```bash
# Check running containers
docker compose ps

# Expected output:
# NAME               STATUS    PORTS
# sofare_ingestion   running   
# sofare_training    running   
# sofare_serving     running   0.0.0.0:8000->8000/tcp
# sofare_mlflow      running   0.0.0.0:5000->5000/tcp
```

Access the services:

- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

## Local Development Setup

For development without Docker:

### 1. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ingestion/requirements.txt
pip install -r training/requirements.txt
pip install -r serving/requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort ruff
```

### 2. Directory Structure

```bash
# Create data directories
mkdir -p data mlflow_data shared_model
```

### 3. Environment Variables

Create a `.env` file:

```env
# Data source configuration
DATA_SOURCE=binance

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys (optional)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
ALPHA_VANTAGE_API_KEY=your_api_key
```

### 4. Start Services Locally

=== "Ingestion"

    ```bash
    cd ingestion
    python -m src.main
    ```

=== "Training"

    ```bash
    cd training
    python -m src.train
    ```

=== "Serving"

    ```bash
    cd serving
    uvicorn src.app:app --reload --port 8000
    ```

=== "MLflow"

    ```bash
    mlflow server \
      --backend-store-uri file:///mlflow_data \
      --default-artifact-root file:///mlflow_data/artifacts \
      --host 0.0.0.0 \
      --port 5000
    ```

## GPU Support

For GPU-accelerated training:

### NVIDIA Docker

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU Docker Compose

Create `docker-compose.gpu.yml`:

```yaml
services:
  training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Troubleshooting

!!! warning "Common Issues"

    === "Port Already in Use"
    
        ```bash
        # Find process using port
        lsof -i :8000
        
        # Kill process
        kill -9 <PID>
        ```
    
    === "Docker Memory Issues"
    
        Increase Docker memory in Docker Desktop settings to at least 4GB.
    
    === "Permission Denied"
    
        ```bash
        # Fix data directory permissions
        sudo chown -R $USER:$USER data/ mlflow_data/
        ```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Configuration Guide](configuration.md)
