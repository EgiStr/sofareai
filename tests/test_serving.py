import unittest
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import MagicMock, patch

# Add serving/src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../serving/src')))

from app import app

class TestServingAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('app.mlflow.pytorch.load_model')
    @patch('app.mlflow.tracking.MlflowClient')
    @patch('app.get_latest_data')
    def test_predict_endpoint(self, mock_get_data, mock_mlflow_client, mock_load_model):
        # Mock Data
        import numpy as np
        mock_get_data.return_value = (np.random.rand(60, 6), np.random.rand(3))
        
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow_client.return_value.search_runs.return_value = [mock_run]
        
        # Mock Model
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.5
        mock_load_model.return_value = mock_model
        
        response = self.client.post("/predict")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertIn("model_version", response.json())
        self.assertEqual(response.json()["model_version"], "test_run_id")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

if __name__ == '__main__':
    unittest.main()
