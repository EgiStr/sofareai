import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add training/src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training/src')))

from features import add_technical_indicators

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })

    def test_indicators_added(self):
        df_processed = add_technical_indicators(self.df)
        
        self.assertIn('rsi', df_processed.columns)
        self.assertIn('macd', df_processed.columns)
        self.assertIn('macd_signal', df_processed.columns)
        self.assertIn('macd_diff', df_processed.columns)
        
        # Check that NaN values (due to windowing) are dropped
        self.assertTrue(len(df_processed) < len(self.df))
        self.assertFalse(df_processed.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
