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

cr = ConfigReader("config/gait_id_config.properties")

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

def create_model_instance(model_class, model_name, batch, device):
    print(f"Creating model instance for {model_class} / {model_name} ")
        # Model instantiation according to constructor
    model = None
    if model_name == "CSILSTMNet":
        model = model_class(
            csi_input_size=batch["csi_seq"].shape[2],
            meta_input_size=batch["metadata_seq"].shape[2],
            window_size=batch["metadata_seq"].shape[1],
            num_classes=31
        ) 
    elif model_name in ["DenseNet1D", "MobileNetV3_1D_LSTM"]:
        model = model_class(
            csi_channels=batch["csi_seq"].shape[2],
            meta_feature_dim=batch["metadata_seq"].shape[2],
            num_classes=31
        ) 
    elif model_name == "EfficientNet1DLSTM":
        model = model_class(
            csi_input_channels=batch["csi_seq"].shape[2],
            meta_input_size=batch["metadata_seq"].shape[-1],
            num_classes=31
        ) 
    else:
        raise ValueError("Unknown model")
    return model.to(device)




# Instantiate model depending on parsed model name
def instantiate_from_runname(params, device=None):
    """Given params from parse_run_name, pick the model class and create an instance using create_model_instance.

    This function looks for known model identifiers inside params['model_name'] and maps them to
    the imported model classes above.
    """
    raw_name = params.get('model_name', '')

    # known model keys and corresponding classes
    model_map = {
        'CSILSTMNet': CSILSTMNet,
        'DenseNet1D': DenseNet1D,
        'MobileNetV3_1D_LSTM': MobileNetV3_1D_LSTM,
        'EfficientNet1DLSTM': EfficientNet1DLSTM,
    }

    # find which known model appears in the raw_name
    chosen_key = None
    for key in model_map.keys():
        if key in raw_name:
            chosen_key = key
            break

    if chosen_key is None:
        raise ValueError(f"Could not infer model class from run name '{raw_name}'")

    model_class = model_map[chosen_key]

    # Build a synthetic sample batch with shapes matching expectations
    # These sizes are typical for your dataset; adapt if needed.
    batch_size = max(1, int(params.get('batch_size', 1)))
    window_size = 128
    csi_channels = 90
    meta_features = 12

    # Torch tensors with shape (batch, window, features)
    sample_batch = {
        'csi_seq': torch.zeros((batch_size, window_size, csi_channels), dtype=torch.float32),
        'metadata_seq': torch.zeros((batch_size, window_size, meta_features), dtype=torch.float32),
        'label': torch.zeros((batch_size,), dtype=torch.long)
    }

    # create model instance
    model = create_model_instance(model_class, chosen_key, sample_batch, device)
    return model

def get_best_model_and_params(best_model_fname=None):
    run_name = "best_overall_model_final_MobileNetV3_1D_LSTM_lr5e-04_bs16_adam_wd1e-04_ep100_valacc0.9656.pt"
    params = parse_run_name(run_name)
    print(params)
    model_instance = None
    total_params = 0
    try:
        try:
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                device = torch.device('mps')
                # optional: improve matmul precision on MPS
                try:
                    torch.set_float32_matmul_precision('high')
                except Exception:
                    pass
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        except Exception:
            device = torch.device('cpu')
        model_instance = instantiate_from_runname(params, device=device)
        print(f"device used: {device}")
        print(f"Created model instance: {model_instance.__class__.__name__}")
        total_params = sum(p.numel() for p in model_instance.parameters())
        print(f"Total parameters: {total_params}")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
    return model_instance, params, total_params, device

'''
run_name = "best_overall_model_final_MobileNetV3_1D_LSTM_lr5e-04_bs16_adam_wd1e-04_ep100_valacc0.9656.pt"
get_best_model_and_params(run_name)


dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
'''

