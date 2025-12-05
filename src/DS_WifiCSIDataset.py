import os
import re
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from torch.utils.data import random_split
from config_reader import ConfigReader 
from numpy import unwrap

###########################################
# Step 2: Dataset with sliding windows
###########################################
class WifiCSIDataset(Dataset):
    def __init__(self, logger, file_list, window_size=128, stride=64, scaler_meta=None, scaler_csi=None):
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        self.logger = logger
        # Fit scalers if not provided
        if scaler_meta is None: 
            self.scaler_meta = StandardScaler()
        else: 
            self.scaler_meta = scaler_meta
        if scaler_csi is None: 
            self.scaler_csi = StandardScaler()
        else: 
            self.scaler_csi = scaler_csi

        # First pass: collect all data for scaling
        all_meta, all_csi = [], []
        i = 0
        #print(f"A_00. Starting first pass for scaling.")
        for f in file_list:
            X_meta, X_csi, _, _, _ = self.load_csv_as_numpy(f)
            print(f"🔍 {i} X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            print(f"    Len X_meta: {len(X_meta[0])}, {len(X_meta[1])} ")
            print(f"    Len X_csi: {len(X_csi[0])}, {len(X_csi[1])} ")

            all_meta.append(X_meta)
            all_csi.append(X_csi)
            i = i+1
        self.logger.critical(f"✅ Completed first pass for scaling.")   
        print(f"🔍 all_meta: {(len(all_meta[0]))} all_csi: {len(all_csi[0])}")

        all_meta = np.vstack(all_meta)
        all_csi = np.vstack(all_csi)
        self.scaler_meta.fit(all_meta)
        self.scaler_csi.fit(all_csi)
        print(f"✅ all_meta: {all_meta.shape} all_csi: {all_csi.shape}")

        # Second pass: windowed sequences
        #print(f"A_01. Starting Second pass: windowed sequences.{file_list}")
        
        for j, f in enumerate(file_list, start=1):
            X_meta, X_csi, y, _, _ = self.load_csv_as_numpy(f)
            X_meta = self.scaler_meta.transform(X_meta)
            X_csi = self.scaler_csi.transform(X_csi)
            T = len(X_meta)
            
            print(f"✅ {j} 🔍 {len(X_meta)}, {len(X_csi)}, {len(y)}, {window_size}, {stride}")
             
            if T < window_size:
                window_size = int(T/2)  # Adjust window size if sequence is shorter

            for c, start in enumerate(range(0, T - window_size + 1, stride), start=1):
                m_seq = X_meta[start:start+window_size]   # (W, 12)
                csi_seq = X_csi[start:start+window_size]  # (W, 99)
                y_seq = {k: v[start:start+window_size] for k, v in y.items()}   
                # print(f"✅Window[{c}] ({start}, {start+window_size}) m_seq:{len(m_seq)}, csi_seq:{len(csi_seq)}, subject:{len(y_seq['subject'])}, activity:{len(y_seq['activity'])}")
                self.samples.append((m_seq, csi_seq, y))
              #  print(f"A_03_{i}. Windowed sample added from {f}, start:{start}")   

        print(f"✅ Completed Second pass: sample length:{len(self.samples)}. Each Sample have {window_size} entries")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        m_seq, csi_seq, y = self.samples[idx]
        # Use the first subject label in the window (or last, depending on your task)
        subject_label = y["subject"][0]  # or y["subject"][-1]
      #  print(f"C_00. __getitem__ idx:{idx} subject_label:{subject_label} ")
       # print(f"C_00. __getitem__ idx:{idx} subject_label:{subject_label} ")
        return {
            "metadata_seq": torch.tensor(m_seq, dtype=torch.float32), # (W, 12)
            "csi_seq": torch.tensor(csi_seq, dtype=torch.float32),   # (W, 99)
            "label": torch.tensor(subject_label, dtype=torch.long)   # shape: (1,)
        }

    def parse_complex(self, s):
        if s is None or s.strip() == "":
            return 0.0 + 0.0j  # treat missing values as 0
        try:
            s = s.replace('+-', '-').replace('-+', '-').replace('i', 'j')
            return complex(s)
        except Exception:
            # If still not parsable, default to 0
            return 0.0 + 0.0j

    def extract_S_C_A_numbers(self, filename):
        """
        Extract subject (Sxx) and class (C03) numbers from filename.
        Example: 'E1_S01_C03_A03_T01.csv' -> (1, 3)
        """
        match = re.search(r'S(\d+).*C(\d+).*A(\d+)', filename)
        if match:
            a, b, c =  int(match.group(1)), int(match.group(2)), int(match.group(3))
            print(f"✅ Extracted: S={a}, C={b}, A={c}")
            return a, b, c
        return None, None
 
    # Place this helper method within the same class (self)
   # from scipy import unwrap # Use 'from numpy import unwrap' if available in your numpy version
    

    def sanitize_phase(self, raw_phase_matrix):
        """
        Applies phase unwrapping and linear trend removal to the raw phase matrix.

        Args:
            raw_phase_matrix (np.ndarray): Array of shape (Time_Steps, Subcarriers).

        Returns:
            np.ndarray: Sanitized phase data.
        """
        T, N = raw_phase_matrix.shape
        sanitized_phase = np.zeros_like(raw_phase_matrix, dtype=np.float32)
        time_index = np.arange(T)

        # Process each subcarrier column-wise
        for subcarrier_index in range(N):
            raw_phase = raw_phase_matrix[:, subcarrier_index]
            
            # 1. Phase Unwrapping
            # Transforms phase from (-pi, pi] to a continuous signal
            unwrapped_phase = unwrap(raw_phase) 
            
            # 2. Linear Trend Removal (Sanitization)
            # Removes the large, static hardware phase offset (CFO/SFO)
            # Fit a 1st-degree polynomial (linear fit: y = mx + c)
            p = np.polyfit(time_index, unwrapped_phase, 1)
            
            # Calculate the linear trend
            linear_trend = np.polyval(p, time_index)
            
            # Subtract the trend to isolate motion-induced phase
            sanitized_phase[:, subcarrier_index] = unwrapped_phase - linear_trend

        return sanitized_phase

    import numpy as np
    import csv
    import os
    # Assuming self.sanitize_phase is defined elsewhere (see helper function below)
    # Note: Ensure you import 'unwrap' directly from 'scipy' or 'numpy' 
    # if you implement the helper function.
    def load_csv_as_numpy(self, filename):
        print(f"B_00. Loading:{ filename}")

        with open(filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            print(f"B_01. Columns: {len(cols)}")
            csi_cols = [c for c in cols if c.startswith('csi_')]
            meta_cols = [
                'timestamp_low','bfee_count','Nrx','Ntx',
                'rssi_a','rssi_b','rssi_c','agc',
                'perm_1','perm_2','perm_3'
            ]
            sa_cols = ['subject', 'activity']
            
              
             # --- MODIFIED: Separate lists for Magnitude and Raw Phase ---
            X_meta, X_mag, X_raw_phase = [], [], []
            subj, class_labels, action_labels = [], [], []
            
            subject, class_label, action_label = self.extract_S_C_A_numbers(os.path.basename(filename))
       
            for i, row in enumerate(reader):
                # metadata
                meta_row = [float(row[c]) for c in meta_cols]
                # --- MODIFIED: Separate Magnitude and Raw Phase Extraction ---
                # --- MODIFIED: Extract MAGNITUDE and RAW PHASE ---
                mag_row, phase_row = [], []
                for c in csi_cols:
                    z = self.parse_complex(row[c])
                    mag_row.append(np.abs(z))   # Magnitude (r)
                    phase_row.append(np.angle(z)) # Raw Phase (theta)

                # Append the row data
                X_meta.append(meta_row)
                X_mag.append(mag_row)
                X_raw_phase.append(phase_row)                                
                
                subj.append(subject)
                class_labels.append(class_label)
                action_labels.append(action_label)
               
              #  print(f"Row {i} loaded. Subject: {subj[-1]} Activity: {act[-1]}"  )             

            # Convert lists to numpy arrays
            X_meta = np.array(X_meta, dtype=np.float32) 
            X_mag = np.array(X_mag, dtype=np.float32) 
            X_raw_phase = np.array(X_raw_phase, dtype=np.float32) # (T, 99)
            X_class_labels = np.array(class_labels, dtype=np.int32)
            X_action_labels = np.array(action_labels, dtype=np.int32)
            
            # --- Magnitude Normalization (row-level z-score) ---
            mag_mean = X_mag.mean(axis=1, keepdims=True)
            mag_std = X_mag.std(axis=1, keepdims=True) + 1e-8

            X_mag = (X_mag - mag_mean) / mag_std if mag_std != 0 else X_mag - mag_mean

            # --- Phase Sanitization ---
            # Unwrap phase column-wise (per subcarrier)
            X_unwrapped_phase = np.unwrap(X_raw_phase, axis=0)

            # Remove linear trend / mean-center row-wise
            phase_mean = X_unwrapped_phase.mean(axis=1, keepdims=True)
            X_sanitized_phase = X_unwrapped_phase - phase_mean

            # --- CRITICAL STEP: Phase Sanitization (Unwrap and Trend Removal) ---
            # The phase data must be processed column-wise (per subcarrier)
            X_sanitized_phase = self.sanitize_phase(X_raw_phase) # (T, 99)

            # --- Combine Magnitude and Phase ---
            # Final CSI feature matrix: (T, 2*Nsub + 2 labels)
            X_csi = np.concatenate(
                (X_mag, X_sanitized_phase, 
                X_class_labels[:, None], X_action_labels[:, None]),
                axis=1, dtype=np.float32
            ) 


            print(f"🔍  X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            # X_csi.shape will now be (T, 198) if the number of subcarriers is 99
            
            print(f"✅  Subject: {len(subj)}, Class: {len(class_labels)}")
            y = {"subject": subj, "class": class_labels}

        return X_meta, X_csi, y, meta_cols, csi_cols

