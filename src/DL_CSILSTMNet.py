import torch
import torch.nn as nn

class CSILSTMNet(nn.Module):
    def __init__(self, csi_input_size, meta_input_size, window_size, num_classes, hidden_size=64, num_layers=2):
        super().__init__()
        self.csi_lstm = nn.LSTM(csi_input_size, hidden_size, num_layers, batch_first=True)
        self.meta_lstm = nn.LSTM(meta_input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, csi_seq, meta_seq):
        # csi_seq: (B, W, csi_input_size)
        # meta_seq: (B, W, meta_input_size)
        _, (h_csi, _) = self.csi_lstm(csi_seq)   # h_csi: (num_layers, B, hidden_size)
        _, (h_meta, _) = self.meta_lstm(meta_seq)
        # Use last layer's hidden state
        h_csi_last = h_csi[-1]   # (B, hidden_size)
        h_meta_last = h_meta[-1] # (B, hidden_size)
        combined = torch.cat([h_csi_last, h_meta_last], dim=1)  # (B, hidden_size*2)
        out = self.fc(combined)  # (B, num_classes)
        return out