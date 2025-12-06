import unittest
import torch
import sys
import os

# Add packages to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages/sofare_common/src')))

from sofare_common.model import SofareM3

class TestSofareM3(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 32
        seq_len = 60
        micro_input_size = 6
        macro_input_size = 3
        safe_input_size = 4
        
        model = SofareM3(micro_input_size, macro_input_size, safe_input_size)
        
        # Create dummy input
        x_micro = torch.randn(batch_size, seq_len, micro_input_size)
        x_macro = torch.randn(batch_size, macro_input_size)
        x_safe = torch.randn(batch_size, safe_input_size)
        
        # Forward pass
        cls_out, reg_out = model(x_micro, x_macro, x_safe)
        
        # Check output shape
        self.assertEqual(cls_out.shape, (batch_size, 2))
        self.assertEqual(reg_out.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
