from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime
import sys
import os
import warnings
import pandas as pd
from config_reader import ConfigReader
from flask_cors import CORS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Flask app
app = Flask(__name__)
CORS(app)  # <-- Add here

# Load config and model at startup
CONFIG_FILE = os.environ.get('CONFIG_FILE', 'config/gait_id_config.properties')
config = ConfigReader(CONFIG_FILE)
print(f"0. Using config file: CONFIG_FILE:{CONFIG_FILE} config:{config}")
 
best_model_pattern = config.get('best_model_pattern')   
print(f"Model path: {config.get('model_save_path')}")
model_save_dir = Path(config.get('model_save_path'))
print(f"Model save directory: {model_save_dir.resolve()}")

model_files = list(model_save_dir.glob(best_model_pattern))
print(f"Found model files: {model_files}")
if not model_files:
    raise FileNotFoundError("No model files found")

# Assume exactly one file matches the pattern; pick the first entry
best_model_path = model_files[0]
print(f"Loading best model from: {best_model_path}")

from CSI_Model_Eval_helper import get_best_model_and_params   
model_instance, params, total_params, device = get_best_model_and_params(str(best_model_path))

print(f"Loaded model: {model_instance} from {best_model_path}")


print(f"INIT DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
import json
def create_json_message(val_true, val_pred, val_prob):
    # Convert arrays/lists to serializable format if needed (e.g., lists)
    message = {
        "val_true": val_true.tolist() if hasattr(val_true, 'tolist') else val_true,
        "val_pred": val_pred.tolist() if hasattr(val_pred, 'tolist') else val_pred,
        "val_prob": val_prob.tolist() if hasattr(val_prob, 'tolist') else val_prob
    }
    json_message = json.dumps(message)
    return json_message


def get_test_loader(test_dataset, batch_size, device):   
    import os
    from torch.utils.data import DataLoader

    num_workers = 0 if device.type in ["mps", "cpu"] else min(4, max(1, (os.cpu_count() or 4) // 2))
    pin_mem = True if device.type != "cpu" else False
    print(f"Creating DataLoader with num_workers={num_workers}, pin_memory={pin_mem}, batch_size={batch_size} ")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )
    print(f"Created DataLoader with {len(test_loader)} batches.")
    print(f"test_loader  :{test_loader}:")
    for i, b in enumerate(test_loader):
        print(f"Batch Values {i} : {b['label'].tolist()}")
        lst = b['label'].tolist()
        for val in lst:
            print(f"  Label value: {val}")


  #  batch = next(iter(test_loader))
    
  #  print(f"Sample batch keys: {batch.keys()}")
    
    return test_loader

def evaluate_model_on_input_data(input_data):
    from CSI_Model_Eval_helper import get_best_model_and_params
    criterion = nn.CrossEntropyLoss()
    model, params, total_params, device = get_best_model_and_params()  
    print(f"Evaluating model on input data with params: {params} on device: {device}")
    model.eval()
    print(f"Model loaded for evaluation: {model}")
    test_loader = get_test_loader(input_data, params['batch_size'], device)
    print(f"Test loader created with {test_loader} ")
 #   print(f"batches : {batch}")
    running_loss, correct, total = 0.0, 0, 0
    val_true, val_pred, val_prob = [], [], []
    non_blocking_flag = True if device.type == "cuda" else False
    with torch.no_grad():
        for batch in test_loader:
            print(f"Processing batch with keys: {batch.keys()}")
            csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
            print(f"Batch shapes - csi_seq: {csi_seq.shape}")

            meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
            print(f"Batch shapes - meta_seq: {meta_seq.shape}")

            labels = batch["label"].squeeze().to(device, non_blocking=non_blocking_flag)
            print(f"Batch shapes - labels: {labels.shape}")

            if torch.isnan(csi_seq).any() or torch.isinf(csi_seq).any():
                csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(meta_seq).any() or torch.isinf(meta_seq).any():
                meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"After NaN/Inf check - csi_seq: {csi_seq.shape}, meta_seq: {meta_seq.shape}, labels: {labels.shape}")    
            # same per-batch normalization used in training
            try:
                mean = csi_seq.mean(dim=(0, 1), keepdim=True)
                std = csi_seq.std(dim=(0, 1), keepdim=True) + 1e-8
                csi_seq = (csi_seq - mean) / std
            except Exception:
                pass
            print(f"After normalization - csi_seq: {csi_seq.shape}")    
            outputs = model(csi_seq, meta_seq)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(f"Batch results - loss: {loss.item()}, correct: {(preds == labels).sum().item()}/{labels.size(0)}")   
            val_true.extend(labels.cpu().numpy().tolist())
            val_pred.extend(preds.cpu().numpy().tolist())
            val_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy().tolist())
            print(f"Accumulated results - running_loss: {running_loss}, correct: {correct}/{total}")
    return val_true, val_pred, val_prob

def setup_logging(log_file_path='/tmp/uploads/app.log'):

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode="w"),
        ],
    )
    logger = logging.getLogger()
    logger.debug("Logger initialized")
    return logger

import logging
log_file_path='/tmp/uploads/app.log'
logger = setup_logging(log_file_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']  # 'file' is the name in your form

    # Create a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    uploaded_file.save(temp.name)

    # You can now use temp.name for further processing
    # Don't forget to close or cleanup the temp file when done

    return f'Temp file created at {temp.name}'


from flask import send_from_directory

@app.route('/gaitid/index.html', methods=['GET'])
def serve_index():
    return send_from_directory('static', 'index.html')

 
import glob
import os
from werkzeug.utils import secure_filename
import os
from DS_WifiCSIDataset import WifiCSIDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
@app.route('/gaitid/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist('file')  # if multiple files, or just request.files.values()
    logger.info(f"Received {len(uploaded_files)} files for prediction.  {uploaded_files}  ")
    saved_file_paths = []
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join('/tmp/uploads', filename)
        uploaded_file.save(save_path)
        saved_file_paths.append(save_path)

    filelist = glob.glob(os.path.join('/tmp/uploads', '**', '*.csv'), recursive=True)
    logger.info(f"Predict: Found {len(filelist)} CSV files in /tmp/uploads for dataset creation.")  
    # Now pass the saved file paths to WifiCSIDataset
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=1)
    
    i = 0
    for a in dataset.samples:        
        m_seq = a[0]
        csi_seq = a[1]
        label = a[2]
        print(f"-----------------------------dataset.samples[{i}]--------------------------------------------------------:")       
        print(f"Metadata Sequence: {m_seq}")
        print(f"CSI Sequence: {csi_seq}")   
        print(f"Label: {label}")
        for l in label:
            print(f"-->Label : {l} ") 
        i += 1

    # Continue with your logic using dataset...
    print(f"Dataset created with {len(dataset)} samples from uploaded files.")
    
    # You can add more processing logic here if needed
    val_true, val_pred, val_prob = evaluate_model_on_input_data(dataset)
    
    # Delete uploaded files after processing
    for file_path in saved_file_paths:
        try:
            os.remove(file_path)
            logger.info(f"Deleted uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
    val_json = create_json_message(val_true, val_pred, val_prob)
    print(f"Returning JSON: {val_json}")
    return val_json

    #return {"message": "Files processed and dataset created. Check logs for details."}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)