# SOFARE-AI

SOFARE-AI is a robust, microservices-based machine learning pipeline designed for cryptocurrency price prediction. It leverages multi-modal data sources, including OHLCV market data, macroeconomic indicators, and safe-haven asset performance, to train and serve advanced deep learning models.

## 🚀 Features

*   **Multi-Modal Data Ingestion:** Real-time and historical data collection from Binance (Crypto), FRED (Macro), and Yahoo Finance (Safe Haven assets).
*   **Advanced ML Architecture:** Utilizes `SofareM3`, a custom model combining Transformer and TCN encoders with attention fusion mechanisms.
*   **MLflow Integration:** Comprehensive experiment tracking, model versioning, and artifact management.
*   **Microservices Architecture:** Modular design with separate containers for Ingestion, Training, and Serving, orchestrated via Docker Compose.
*   **Real-time Serving:** FastAPI-based prediction engine providing low-latency inference.
*   **Drift Detection:** Automated monitoring for data drift to trigger retraining.
*   **Interactive Dashboard:** Visual interface for monitoring predictions, market data, and system health.

## 📂 Project Structure

```
sofareai/
├── data/                # Shared data storage (CSVs)
├── ingestion/           # Data collection service
├── serving/             # Prediction API and Dashboard
├── training/            # Model training and evaluation pipeline
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for analysis
├── docker-compose.yml   # Container orchestration
└── .gitignore           # Git ignore rules
```

## 🛠️ Getting Started

### Prerequisites

*   Docker & Docker Compose
*   Python 3.10+ (for local development)

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sofareai.git
    cd sofareai
    ```

2.  **Build and Run with Docker Compose:**
    ```bash
    docker compose build
    docker compose up -d
    ```

3.  **Access Services:**
    *   **Dashboard:** [http://localhost:8000](http://localhost:8000)
    *   **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   **MLflow UI:** [http://localhost:5000](http://localhost:5000)

## 🧠 Model Architecture

The core model, **SofareM3**, is designed to handle heterogeneous time-series data:
*   **Micro Encoder:** Processes OHLCV and Technical Indicators.
*   **Macro Encoder:** Processes macroeconomic data (Fed Funds Rate, etc.).
*   **Safe Haven Encoder:** Processes assets like Gold, DXY, S&P 500.
*   **Fusion Layer:** Uses attention mechanisms to weigh the importance of different modalities.

## 🤝 Contributing

Contributions are welcome! Please check out the [issues](https://github.com/yourusername/sofareai/issues) or submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
