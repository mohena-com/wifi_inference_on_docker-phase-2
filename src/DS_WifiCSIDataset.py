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
        for f in file_list:
            X_meta, X_csi, _, _, _ = self.load_csv_as_numpy(f)
            self.logger.debug(f"A_00_{i}. X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            print(f"A_00_{i}. X_meta: {X_meta} ")            
            print(f"A_01_{i}. X_csi: {X_csi}")

            all_meta.append(X_meta)
            all_csi.append(X_csi)
            i = i+1
        self.logger.critical(f"A_00. Completed first pass for scaling.")   
        self.logger.debug(f"A_01. all_meta: {(len(all_meta[0]))} all_csi: {len(all_csi)}")

        all_meta = np.vstack(all_meta)
        all_csi = np.vstack(all_csi)
        self.scaler_meta.fit(all_meta)
        self.scaler_csi.fit(all_csi)
        self.logger.debug(f"A_02. all_meta: {all_meta.shape} all_csi: {all_csi.shape}")

        # Second pass: windowed sequences
        for f in file_list:
            X_meta, X_csi, y, _, _ = self.load_csv_as_numpy(f)
            X_meta = self.scaler_meta.transform(X_meta)
            X_csi = self.scaler_csi.transform(X_csi)
            T = len(X_meta)
            print(f"A_01_{i}. X_meta: {X_meta} ")
            print(f"A_01_{i}. X_csi: {X_csi} ")
            print(f"A_01_{i}. y: {y} ")
            print(f"A_01_window_size:{window_size}. stride: {stride}. T: {T}")
            
            for start in range(0, T - window_size + 1, stride):
                m_seq = X_meta[start:start+window_size]   # (W, 12)
                csi_seq = X_csi[start:start+window_size]  # (W, 99)
                self.samples.append((m_seq, csi_seq, y))
                self.logger.debug(f"A_03_{i}. Windowed sample added from {f}, start:{start}")   
        self.logger.critical(f"A_01. Completed Second pass: windowed sequences. sample length:{len(self.samples)}")
        print(f"A_01. Completed Second pass: windowed sequences. sample  :{self.samples}")
       # self.logger.info(f"dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        m_seq, csi_seq, y = self.samples[idx]
        # Use the first subject label in the window (or last, depending on your task)
        subject_label = y["subject"]  # or y["subject"][-1]
        print(f"C_00. __getitem__ idx:{idx} subject_label:{subject_label} ")
        return {
            "metadata_seq": torch.tensor(m_seq, dtype=torch.float32), # (W, 12)
            "csi_seq": torch.tensor(csi_seq, dtype=torch.float32),   # (W, 99)
            "label": torch.tensor(subject_label, dtype=torch.long)   # shape: ()
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

    def extract_S_A_numbers(self, filename):
        """
        Extract subject (Sxx) and activity (Axx) numbers from filename.
        Example: 'E1_S01_C03_A03_T01.csv' -> (1, 3)
        """
        match = re.search(r'S(\d+).*A(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def load_csv_as_numpy(self, filename):
        print(f"B_00. Loading:{ filename}")
        with open(filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            print(f"B_01. Columns: {cols}")
            csi_cols = [c for c in cols if c.startswith('csi_')]
            meta_cols = [
                'timestamp_low','bfee_count','Nrx','Ntx',
                'rssi_a','rssi_b','rssi_c','agc',
                'perm_1','perm_2','perm_3'
            ]
            sa_cols = ['subject', 'activity']
            
            X_meta, X_csi = [], []
            subj, act = [], []
            s, a = self.extract_S_A_numbers(os.path.basename(filename))
            fromrow = False
            if s is None or a is None:
                fromrow = True
            else:
                fromrow = False

            for row in reader:
                # metadata
                meta_row = [float(row[c]) for c in meta_cols]
                # CSI as magnitudes
                csi_row = [abs(self.parse_complex(row[c])) for c in csi_cols]
                X_meta.append(meta_row)
                X_csi.append(csi_row)

                
                if fromrow:
                    sa_row = [int(row[c]) for c in sa_cols]
                    subj.append(sa_row[0])
                    act.append(sa_row[1])
                else:
                    subj.append(s)
                    act.append(a)
                    

            X_meta = np.array(X_meta, dtype=np.float32)  # (T, 12)
            X_csi = np.array(X_csi, dtype=np.float32)    # (T, 99)
            self.logger.debug(f"B_01. X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            
            self.logger.debug(f"B_02. Subject: {len(subj)}, Activity: {len(act)}")
            y = {"subject": subj, "activity": act}
            print(f"B_03. y subject: {y['subject']} activity: {y['activity']}"  )
            

            return X_meta, X_csi, y, meta_cols, csi_cols

