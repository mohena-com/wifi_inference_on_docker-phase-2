import os
import re
import csv
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from numpy import unwrap

###########################################
# Step 2: Dataset with sliding windows
###########################################
class WifiCSIDataset(Dataset):
    def __init__(self, logger, file_list, window_size=128, stride=64):
        """
        WifiCSIDataset builds windowed sequences from raw CSV files.

        - No normalization is applied here.
        - CSI is built as [magnitude | phase] for each subcarrier.
        - Normalization (meta + CSI mag/phase) is applied later in __getitem__
          if scalers are attached via set_scalers().
        """
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        self.logger = logger

        # scalers will be attached later via set_scalers(...)
        self.scaler_meta = None
        self.scaler_mag = None
        self.scaler_phase = None
        self.n_subcarriers = None  # will be set in set_scalers

        # Build samples (raw, unnormalized)
        for f in file_list:
            X_meta, X_csi, y, _, _ = self.load_csv_as_numpy(f)

            T = len(X_meta)
            for start in range(0, T - window_size + 1, stride):
                m_seq = X_meta[start:start + window_size]   # (W, n_meta)
                csi_seq = X_csi[start:start + window_size]  # (W, 2 * n_subcarriers)
                self.samples.append((m_seq, csi_seq, y))

        self.logger.critical(f"Dataset initialized with {len(self.samples)} windowed samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single window:
          - metadata_seq: (W, n_meta)
          - csi_seq:      (W, 2 * n_subcarriers) -> [mag | phase]
          - label:        scalar in [0..29]
        Normalization is applied here if scalers are set.
        """
        m_seq, csi_seq, y = self.samples[idx]  # numpy arrays
        
        # 1) META normalization (if scaler_meta is set)
        if self.scaler_meta is not None:
            # m_seq: (W, n_meta)
            m_seq = self.scaler_meta.transform(m_seq)

        # 2) CSI mag/phase normalization (if scalers are set)
        if self.scaler_mag is not None and self.scaler_phase is not None:
            if self.n_subcarriers is None:
                # Infer once from csi_seq shape
                self.n_subcarriers = csi_seq.shape[1] // 2

            n_sc = self.n_subcarriers
            mag = csi_seq[:, :n_sc]        # (W, n_sc)
            phase = csi_seq[:, n_sc:]      # (W, n_sc)

            mag = self.scaler_mag.transform(mag)
            phase = self.scaler_phase.transform(phase)

            csi_seq = np.concatenate([mag, phase], axis=1)  # back to (W, 2*n_sc)

        # Subject label: 1..30 -> 0..29
        subject_label = int(y["subject"][0]) - 1
        #print("IDX:", idx, "SUB:", subject_label)
        return {
            "metadata_seq": torch.tensor(m_seq, dtype=torch.float32),
            "csi_seq":      torch.tensor(csi_seq, dtype=torch.float32),
            "label":        torch.tensor(subject_label, dtype=torch.long)
        }

    # =============================
    # Attach scalers after dataset is built
    # =============================
    def set_scalers(self, scaler_meta, scaler_mag, scaler_phase, n_subcarriers=None):
        """
        Attach pre-fitted scalers to the dataset.

        - scaler_meta  : StandardScaler for metadata (per column)
        - scaler_mag   : StandardScaler for CSI magnitude (per subcarrier)
        - scaler_phase : StandardScaler for CSI phase (per subcarrier)
        - n_subcarriers: number of subcarriers (if None, inferred in __getitem__)
        """
        self.scaler_meta = scaler_meta
        self.scaler_mag = scaler_mag
        self.scaler_phase = scaler_phase
        self.n_subcarriers = n_subcarriers
        self.logger.info("Meta + CSI (mag/phase) scalers attached to WifiCSIDataset")

    # =============================
    # Helper methods
    # =============================
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
        Extract subject (Sxx), class (Cxx), and action (Axx) numbers from filename.
        Example: 'E1_S01_C03_A03_T01.csv' -> (1, 3, 3)
        """
        match = re.search(r'S(\d+).*C(\d+).*A(\d+)', filename)
        if match:
            a, b, c =  int(match.group(1)), int(match.group(2)), int(match.group(3))
             
            return a, b, c
        return None, None, None

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
            unwrapped_phase = unwrap(raw_phase)

            # 2. Linear Trend Removal (Sanitization)
            p = np.polyfit(time_index, unwrapped_phase, 1)
            linear_trend = np.polyval(p, time_index)

            sanitized_phase[:, subcarrier_index] = unwrapped_phase - linear_trend

        return sanitized_phase

    def load_csv_as_numpy(self, filename):
        """
        Loads a single CSV and returns:
          - X_meta : (T, n_meta)
          - X_csi  : (T, 2 * N_subcarriers) = [mag | phase]
          - y      : dict with 'subject' (list of subject IDs per row)
        No normalization is done here.
        """
        self.logger.debug(f"📂 Loading: {filename}")

        with open(filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames

            csi_cols = [c for c in cols if c.startswith('csi_')]
            meta_cols = [
                'timestamp_low', 'bfee_count', 'Nrx', 'Ntx',
                'rssi_a', 'rssi_b', 'rssi_c', 'agc',
                'perm_1', 'perm_2', 'perm_3'
            ]

            X_meta, X_mag, X_raw_phase = [], [], []
            subj, class_labels, action_labels = [], [], []

            subject, class_label, action_label = self.extract_S_C_A_numbers(os.path.basename(filename))

            for row in reader:
                # metadata
                meta_row = [float(row[c]) for c in meta_cols]

                # CSI: magnitude + raw phase
                mag_row, phase_row = [], []
                for c in csi_cols:
                    z = self.parse_complex(row[c])
                    mag_row.append(np.abs(z))      # Magnitude
                    phase_row.append(np.angle(z))  # Raw Phase

                X_meta.append(meta_row)
                X_mag.append(mag_row)
                X_raw_phase.append(phase_row)

                subj.append(subject)
                class_labels.append(class_label)
                action_labels.append(action_label)

            # Convert to numpy arrays
            X_meta = np.array(X_meta, dtype=np.float32)           # (T, n_meta)
            X_mag = np.array(X_mag, dtype=np.float32)             # (T, N_subcarriers)
            X_raw_phase = np.array(X_raw_phase, dtype=np.float32) # (T, N_subcarriers)

            # --- Phase processing ---
            # Option 1: simple unwrap only (current)
            X_unwrapped_phase = np.unwrap(X_raw_phase, axis=0)

            # Option 2: full sanitization (uncomment if you prefer)
            # X_unwrapped_phase = self.sanitize_phase(X_raw_phase)

            X_sanitized_phase = X_unwrapped_phase

            # Combine magnitude and phase into CSI feature matrix
            X_csi = np.concatenate(
                (X_mag, X_sanitized_phase),
                axis=1, dtype=np.float32
            )

            self.logger.debug(f"B_01. X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            self.logger.debug(f"B_02. Subject: {len(subj)}, Class: {len(class_labels)}")

            y = {"subject": subj}

        return X_meta, X_csi, y, meta_cols, csi_cols


###########################################
# Helper functions to build scalers
###########################################

def build_scaler_meta(train_dataset):
    """
    Build StandardScaler for metadata using ONLY training dataset.
    Assumes each sample is (m_seq, csi_seq, y) with:
      m_seq shape -> (W, n_meta)
    """
    all_meta = []
    for (m_seq, _, _) in train_dataset.samples:
        all_meta.append(m_seq)

    all_meta = np.vstack(all_meta)  # (total_rows, n_meta)

    scaler_meta = StandardScaler().fit(all_meta)
    print("✔ scaler_meta created successfully")

    return scaler_meta


def build_scalers_csi_mag_phase(train_dataset):
    """
    Build separate StandardScalers for CSI magnitude and phase.
    Assumes each csi_seq is:
      csi_seq shape -> (W, 2 * N_subcarriers)
      [0:N]   -> magnitude
      [N:2N]  -> phase
    """
    all_mag = []
    all_phase = []

    # Infer N_subcarriers from first sample
    first_csi = train_dataset.samples[0][1]
    n_sc = first_csi.shape[1] // 2

    for (_, csi_seq, _) in train_dataset.samples:
        mag = csi_seq[:, :n_sc]
        phase = csi_seq[:, n_sc:]

        all_mag.append(mag)
        all_phase.append(phase)

    all_mag = np.vstack(all_mag)      # (total_frames, n_sc)
    all_phase = np.vstack(all_phase)  # (total_frames, n_sc)

    scaler_mag = StandardScaler().fit(all_mag)
    scaler_phase = StandardScaler().fit(all_phase)

    print("✔ scaler_mag and scaler_phase created successfully")
    print(f"   Magnitude dims = {n_sc}, Phase dims = {n_sc}")

    return scaler_mag, scaler_phase
