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
    def __init__(self, csi_channels, meta_feature_dim, num_classes, dropout_p=0.3):
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
        self.lstm = nn.LSTM(
            meta_feature_dim, 
            64,
            num_layers=2,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_p
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(128 + 64*2, num_classes)

    def forward(self, csi_seq, meta_seq):
        # --- INPUT DATA ---
        print("\n🧪 [DEBUG] ----- FORWARD START -----")
        print(f"🔍 [Input] CSI seq shape (B, L, C): {list(csi_seq.shape)}")
        print(f"🔍 [Input] Meta seq shape (B, L, F): {list(meta_seq.shape)}")
        print(f"🔍 [Input] CSI seq shape (csi_seq): {csi_seq}")
        # --- CSI Branch (DenseNet1D) ---
        print("\n📡 [CSI] Branch start")
        x = csi_seq.permute(0, 2, 1)
        print(f"📡 [CSI] After permute (B, C, L): {list(x.shape)}")
        print(f"🔍 [CSI] After permute x : {x}")
        x = self.initial_conv(x)
        print(f"📡 [CSI] After initial_conv (B, 64, L/2): {list(x.shape)}")
        print(f"📡 [CSI] After initial_conv x: {x}")

        x = self.dense_block(x)
        print(f"📡 [CSI] After DenseBlock (B, 192, L/2): {list(x.shape)}")
        print(f"📡 [CSI] After DenseBlock x: {x}")

        x = self.transition(x)
        print(f"📡 [CSI] After Transition (B, 128, L/4): {list(x.shape)}")
        print(f"📡 [CSI] After Transition x: {x}")

        x = self.global_pool(x).squeeze(-1)
        print(f"📡 [CSI] After GlobalPool + squeeze (B, 128): {list(x.shape)}")
        print(f"📡 [CSI] After GlobalPool + squeeze x: {x}")
        csi_features = x

        # --- Meta Branch (LSTM) ---
        print("\n🧬 [META] Branch start")
        _, (h_n, c_n) = self.lstm(meta_seq)
        print(f"🧬 [META] LSTM h_n shape (D*layers, B, H): {list(h_n.shape)}")

        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        print(f"🧬 [META] Final concatenated features (B, 64*2): {list(h_n.shape)}")
        meta_features = h_n

        # --- Combined Branch (FC Layer) ---
        print("\n🧾 [FC] Combining features")
        combined = torch.cat([csi_features, meta_features], dim=1)
        print(f"🧾 [FC] Combined features shape (B, 128 + 128): {list(combined.shape)}")
        print(f"🧾 [FC] Combined features shape combined: {combined}")
        # Extra: stats on combined features
        print(f"    ▶ combined min={combined.min().item():.4f}, "
              f"max={combined.max().item():.4f}, "
              f"mean={combined.mean().item():.4f}")

        # Extra: FC layer parameter shapes
        print(f"    ▶ fc.weight shape: {list(self.fc.weight.shape)}  "
              f"(out_features={self.fc.out_features}, in_features={self.fc.in_features})")
        print(f"    ▶ fc.bias   shape: {list(self.fc.bias.shape)}")

        # Extra: inspect first sample's combined features
        x0 = combined[0]      # shape: (256,)
        print(f"    ▶ combined[0] sample (first 10 values): "
              f"{x0[:10].detach().cpu().numpy()}")
        
        combined = self.dropout(combined)  # dropout active in train mode
        # Compute logits through the FC layer (usual path)
        output = self.fc(combined)

        # Extra: manually recompute logits for first sample
        with torch.no_grad():
            w = self.fc.weight  # (31, 256)
            b = self.fc.bias    # (31,)
            manual_logits0 = torch.matmul(w, x0) + b
            print(f"    ▶ manual logits[0] (from W·x + b) (first 10): "
                  f"{manual_logits0[:10].detach().cpu().numpy()}")

            # Per-sample contribution analysis
            print("\n🧠 [EXPLAIN] Top feature contributions per sample")
            num_samples_to_show = min(3, output.size(0))   # don’t spam logs
    
            for i in range(num_samples_to_show):
                logits_i = output[i]                       # (num_classes,)
                pred_cls = torch.argmax(logits_i).item()   # winning class index
    
                w_cls = w[pred_cls]                        # (256,)
                x_i = combined[i]                          # (256,)
                contrib = w_cls * x_i                      # element-wise contribution
    
                top_vals, top_idx = torch.topk(contrib, 5) # top-5 contributing features
    
                print(f"\n🔍 ----- Sample {i} -----")
                print(f"    ▶ predicted class: {pred_cls}")
                print(f"    ▶ logit[pred_cls]: {logits_i[pred_cls].item():.4f}")
                print(f"    ▶ top-5 feature contributions for class {pred_cls}:")
                for k in range(5):
                    j = top_idx[k].item()
                    print(f"       • feat[{j:3d}]  x={x_i[j].item(): .4f}  "
                          f"w={w_cls[j].item(): .4f}  "
                          f"contrib={top_vals[k].item(): .4f}")
    
        # --- OUTGOING DATA ---
        print(f"\n🧾 [Output] Logits shape (B, num_classes): {list(output.shape)}")
        for a in output:
            print(f"🧬 [Output] Logits sample: {a}")
        print("🧪 [DEBUG] ----- FORWARD END -----\n")
    
        return output



