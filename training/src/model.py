import torch
import torch.nn as nn

class MultiModalLSTM(nn.Module):
    def __init__(self, input_size, macro_size, hidden_size=64, num_layers=2, output_size=1):
        super(MultiModalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Branch A: Time Series (LSTM)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Branch B: Macro (Dense)
        self.macro_fc = nn.Linear(macro_size, 16)
        
        # Fusion
        self.fusion_fc = nn.Linear(hidden_size + 16, 32)
        self.final_fc = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x_seq, x_macro):
        # LSTM Branch
        h0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        
        lstm_out, _ = self.lstm(x_seq, (h0, c0))
        lstm_out = lstm_out[:, -1, :] # Last time step
        
        # Macro Branch
        macro_out = self.relu(self.macro_fc(x_macro))
        
        # Fusion
        combined = torch.cat((lstm_out, macro_out), dim=1)
        x = self.relu(self.fusion_fc(combined))
        out = self.final_fc(x)
        
        return out
