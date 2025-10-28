import torch.nn as nn
import torch

# Define DenseNet1D block (simplified)
class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False)
                )
            )
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class DenseNet1D(nn.Module):
    def __init__(self, csi_channels, meta_feature_dim, num_classes):
        super().__init__()
        self.initial_conv = nn.Conv1d(csi_channels, 64, 7, stride=2, padding=3)
        self.dense_block = DenseBlock1D(64, growth_rate=32, n_layers=4)
        self.transition = nn.Sequential(
            nn.BatchNorm1d(64 + 4*32),
            nn.ReLU(inplace=True),
            nn.Conv1d(64 + 4*32, 128, 1),
            nn.AvgPool1d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(meta_feature_dim, 64, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 + 64*2, num_classes)

    def forward(self, csi_seq, meta_seq):
        x = csi_seq.permute(0, 2, 1)
        x = self.initial_conv(x)
        x = self.dense_block(x)
        x = self.transition(x)
        x = self.global_pool(x).squeeze(-1)

        _, (h_n, _) = self.lstm(meta_seq)
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)

        combined = torch.cat([x, h_n], dim=1)
        return self.fc(combined)


