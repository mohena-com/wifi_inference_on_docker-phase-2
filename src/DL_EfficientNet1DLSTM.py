import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super().__init__()
        mid_channels = in_channels * expansion
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(mid_channels, mid_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(mid_channels // 4, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        se_weight = self.se(out)
        out = out * se_weight
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out

class EfficientNet1DLSTM(nn.Module):
    def __init__(self, csi_input_channels, meta_input_size, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(csi_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        self.mbconv1 = MBConv1D(32, 24, stride=2)
        self.mbconv2 = MBConv1D(24, 40, stride=2)
        self.mbconv3 = MBConv1D(40, 80, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(meta_input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(80 + 64*2, num_classes)

    def forward(self, csi_seq, meta_seq):
        # csi_seq shape: (B, W, C), transpose for Conv1d: (B, C, W)
        x = csi_seq.permute(0, 2, 1)
        x = self.stem(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.pool(x).squeeze(-1)

        _, (h_n, _) = self.lstm(meta_seq)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)

        out = torch.cat([x, h_n], dim=1)
        out = self.fc(out)
        return out
