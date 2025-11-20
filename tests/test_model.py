import unittest
import torch
import sys
import os

# Add training/src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training/src')))

from model import MultiModalLSTM

class TestMultiModalModel(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 32
        seq_len = 60
        input_size = 6
        macro_size = 3
        hidden_size = 64
        
        model = MultiModalLSTM(input_size, macro_size, hidden_size)
        
        # Create dummy input
        x_seq = torch.randn(batch_size, seq_len, input_size)
        x_macro = torch.randn(batch_size, macro_size)
        
        # Forward pass
        output = model(x_seq, x_macro)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
