import torch.nn as nn
import torch
import numpy as np  # Used for printing shapes nicely

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
        print(f"🧱 [INIT][DenseBlock] n_layers={n_layers}, out_channels={in_channels + n_layers * growth_rate}")
    def forward(self, x):
        print(f"🔍 [DenseBlock] Input shape (B, C, L): {list(x.shape)}")
        features = [x]
        for i, layer in enumerate(self.layers):
            concatenated = torch.cat(features, dim=1)
            print(f"🔍 [DenseBlock][Layer {i+1}/{len(self.layers)}] Concatenated shape: {list(concatenated.shape)}")
            out = layer(concatenated)
            features.append(out)
            print(f"🔍 [DenseBlock][Layer {i+1}/{len(self.layers)}] Output shape: {list(out.shape)}")
        final_output = torch.cat(features, dim=1)
        print(f"🔍 [DenseBlock] Final output shape: {list(final_output.shape)}")
        return final_output

class DenseNet1D(nn.Module):
    def __init__(self, csi_channels, meta_feature_dim, num_classes):
        print(f"🧱 [INIT][DenseNet1D] CSI={csi_channels}, Meta={meta_feature_dim}, Classes={num_classes}")
        super().__init__()
        # --- CSI Branch Components ---
        self.initial_conv = nn.Conv1d(csi_channels, 64, 7, stride=2, padding=3)
        self.dense_block = DenseBlock1D(64, growth_rate=32, n_layers=4)
        dense_out_channels = 64 + 4 * 32  # 64 + 128 = 192
        self.transition = nn.Sequential(
            nn.BatchNorm1d(dense_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(dense_out_channels, 128, 1),
            nn.AvgPool1d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --- Meta Branch Components ---
        self.lstm = nn.LSTM(meta_feature_dim, 64, 2, batch_first=True, bidirectional=True)
        # --- Final Classifier ---
        self.fc = nn.Linear(128 + 64*2, num_classes)

    def forward(self, csi_seq, meta_seq):
        # --- INPUT DATA ---
        print("\n🧪 [DEBUG] ----- FORWARD START -----")
        print(f"🔍 [Input] CSI seq shape (B, L, C): {list(csi_seq.shape)}")
        print(f"🔍 [Input] Meta seq shape (B, L, F): {list(meta_seq.shape)}")
        # --- CSI Branch (DenseNet1D) ---
        print("\n📡 [CSI] Branch start")
        # B, L, C -> B, C, L
        x = csi_seq.permute(0, 2, 1)
        print(f"📡 [CSI] After permute (B, C, L): {list(x.shape)}")
        x = self.initial_conv(x)
        print(f"📡 [CSI] After initial_conv (B, 64, L/2): {list(x.shape)}")
        x = self.dense_block(x)
        print(f"📡 [CSI] After DenseBlock (B, 192, L/2): {list(x.shape)}")
        x = self.transition(x)
        print(f"📡 [CSI] After Transition (B, 128, L/4): {list(x.shape)}")
        x = self.global_pool(x).squeeze(-1)
        print(f"📡 [CSI] After GlobalPool + squeeze (B, 128): {list(x.shape)}")
        csi_features = x

        # --- Meta Branch (LSTM) ---
        print("\n🧬 [META] Branch start")
        _, (h_n, c_n) = self.lstm(meta_seq)
        print(f"🧬 [META] LSTM h_n shape (D*layers, B, H): {list(h_n.shape)}")
        # Concatenate the final forward and backward layer states
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        print(f"🧬 [META] Final concatenated features (B, 64*2): {list(h_n.shape)}")
        meta_features = h_n

        # --- Combined Branch (FC Layer) ---
        print("\n🧾 [FC] Combining features")
        combined = torch.cat([csi_features, meta_features], dim=1)
        print(f"🧾 [FC] Combined features shape (B, 128 + 128): {list(combined.shape)}")
        output = self.fc(combined)
        # --- OUTGOING DATA ---
        print(f"🧾 [Output] Logits shape (B, num_classes): {list(output.shape)}")
        print("🧪 [DEBUG] ----- FORWARD END -----\n")
        return output


