import os
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import psutil
from mlflow.models.signature import infer_signature
import itertools
from sklearn.preprocessing import label_binarize

# Import your custom classes
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM
# EfficientNet1D would be imported similarly

# --- Logging setup (use your CSI_ID.py pattern) ---
cr = ConfigReader("csi_id_config.properties")

def parse_run_name(run_name: str) -> dict:
    """
    Convert a model run name back into hyperparameters dictionary.
    Example: 'DenseNet1D_lr1e-3_bs64_adam_wd1e-4_ep20' ->
    {
        'model_name': 'DenseNet1D',
        'lr': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'weight_decay': 0.0001,
        'epochs': 20
    }
    """
    try:
        # Split by underscore
        parts = run_name.split('_')
        
        # Extract model name (everything before _lr)
        model_idx = next(i for i, part in enumerate(parts) if part.startswith('lr'))
        model_name = '_'.join(parts[:model_idx])
        
        params = {
            'model_name': model_name,
            'optimizer': None,  # Will be set later
            'lr': 0,
            'batch_size': 0,
            'weight_decay': 0,
            'epochs': 0
        }
        
        # Parse each part
        for part in parts[model_idx:]:
            if part.startswith('lr'):
                # Handle learning rate
                lr_str = part[2:]  # Remove 'lr'
                if 'e-' in lr_str:
                    # Handle scientific notation (e.g., '1e-3')
                    params['lr'] = float(lr_str)
                else:
                    # Handle decimal notation with 'p' (e.g., '0p01' -> 0.01)
                    params['lr'] = float(lr_str.replace('p', '.'))
                    
            elif part.startswith('bs'):
                # Handle batch size
                params['batch_size'] = int(part[2:])
                
            elif part.startswith('wd'):
                # Handle weight decay
                wd_str = part[2:]
                if 'e-' in wd_str:
                    params['weight_decay'] = float(wd_str)
                else:
                    params['weight_decay'] = float(wd_str.replace('p', '.'))
                    
            elif part.startswith('ep'):
                # Handle epochs
                params['epochs'] = int(part[2:])
                
            # Handle optimizer (adam, adamw, sgd etc.)
            elif part in ('adam', 'adamw', 'sgd'):
                params['optimizer'] = part
        
        return params
        
    except Exception as e:
        # If parsing fails, return None or raise an exception
        raise ValueError(f"Failed to parse run name '{run_name}': {str(e)}")


run_name = "best_overall_model_final_MobileNetV3_1D_LSTM_lr5e-04_bs16_adam_wd1e-04_ep100_valacc0.9656.pt"
params = parse_run_name(run_name)
print(params)