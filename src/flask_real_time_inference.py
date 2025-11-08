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

# -------------------- LOAD TRAINED WEIGHTS --------------------
import torch

try:
    checkpoint = torch.load(best_model_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    best_model_instance.load_state_dict(state_dict)
    best_model_instance.eval()
    print(f"[INFO] Loaded model weights from: {best_model_path}")

except Exception as e:
    print(f"[WARNING] Could not load model weights from {best_model_path}: {e}")
# ---------------------------------------------------------------




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
# --- drop-in replacement for evaluate_model_on_input_data in flask_real_time_inference.py ---
def evaluate_model_on_input_data(test_loader, model, device, params=None):
    """
    Evaluate model on a DataLoader (works for inference and validation).
    Returns: val_true (optional), val_pred, val_prob
    """
    import torch.nn.functional as F

    criterion = nn.CrossEntropyLoss()
    print(f"Evaluating model on DataLoader with params: {params} on device: {device}")

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    val_true, val_pred, val_prob = [], [], []
    non_blocking_flag = (device.type == "cuda")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"\n🧩 Processing batch {batch_idx + 1}/{len(test_loader)}")

            # --- Move inputs to device ---
            csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
            meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)

            # --- Clean invalid values ---
            csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)

            # --- Forward pass ---
            outputs = model(csi_seq, meta_seq)
            for a in outputs:
                print(f"0==>output :{a}")
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            print(f"CSI shape: {csi_seq.shape}, META shape: {meta_seq.shape}, outputs: {outputs.shape}")
            for a, b in zip(probs, preds):
                print(f"1==>probs :{a}, pred: {b}")

            # --- Always store predictions and probabilities ---
            val_pred.extend(preds.cpu().numpy().tolist())
            val_prob.extend(probs.cpu().numpy().tolist())

            # --- Fetch label (prefer subject) ---
            labels = None
            if "label" in batch and batch["label"].numel() > 0:
                labels = batch["label"]     # keep batch dim
            elif "subject" in batch:
                subj_tensor = batch["subject"]
                if subj_tensor is not None and subj_tensor.numel() > 0:
                    labels = subj_tensor     # keep batch dim

            print(f"Actual label value: {labels.squeeze().item() if labels.numel() == 1 else labels.squeeze().tolist()}")


            # --- Compute loss only if valid label exists ---
            if labels is not None and labels.numel() > 0:
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)  # ensure (N,)
                labels = labels.long().to(device, non_blocking=non_blocking_flag)
                # safety: batch should match
                assert outputs.size(0) == labels.size(0), f"batch mismatch: {outputs.size()} vs {labels.size()}"
                loss = criterion(outputs, labels)
                running_loss += float(loss.item())
                correct += (preds == labels).sum().item()
                print(f"✅  VERIFICATION labels:{labels.squeeze().tolist()} preds:{preds.squeeze().tolist()} ")
                total += labels.size(0)
                val_true.extend(labels.cpu().numpy().tolist())
                print(f"Batch {batch_idx + 1}: loss={loss.item():.4f}, acc={(preds == labels).sum().item()}/{labels.size(0)}")
            else:
                print(f"Batch {batch_idx + 1}: No valid label/subject found → inference-only mode.")
    
    print(f"probs:{len(val_prob)},  pred:{len(val_pred)},   true:{len(val_true)}")
    for a, b, c in zip(val_prob, val_pred, val_true):
        print(f"probs:{a}, pred:{b},  true:{c}")
    # --- Summary ---
    if total > 0:
        avg_loss = running_loss / len(test_loader)
        acc = 100.0 * correct / total
        print(f"\n✅ Validation complete: Avg Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
    else:
        print(f"\n✅ Inference complete: {len(val_pred)} predictions generated.")

    return val_true if len(val_true) > 0 else None, val_pred, val_prob



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

import os
import pandas as pd
from flask import request, jsonify
from werkzeug.utils import secure_filename
import glob 
from DS_WifiCSIDataset import WifiCSIDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

@app.route('/gaitid/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist('file')
    print(f"Received {len(uploaded_files)} files for prediction: {[f.filename for f in uploaded_files]}")

    upload_dir = '/tmp/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    saved_file_paths = []

    # --- Helper: pad CSVs if too short ---
    def pad_csv_if_needed(csv_path, min_rows=128):
        """Pads short CSVs with last row to reach min_rows."""
        try:
            df = pd.read_csv(csv_path)
            current_len = len(df)
            if current_len < min_rows:
                pad_rows = min_rows - current_len
                last_row = df.iloc[-1:]
                pad_df = pd.concat([last_row] * pad_rows, ignore_index=True)
                df = pd.concat([df, pad_df], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Padded {os.path.basename(csv_path)} from {current_len} → {len(df)} rows.")
            else:
                print(f"{os.path.basename(csv_path)} already has {current_len} rows — no padding needed.")
        except Exception as e:
            logger.error(f"Padding failed for {csv_path}: {e}")

    # --- Save and pad uploaded files ---
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join(upload_dir, filename)
        uploaded_file.save(save_path)
        print(f"Saved uploaded file to {save_path}")

        pad_csv_if_needed(save_path, min_rows=128)
        saved_file_paths.append(save_path)

    # --- Load dataset and evaluate ---
    try:
        test_dataset = WifiCSIDataset(
            logger=logger,
            file_list=saved_file_paths,
            window_size=128,
            stride=64
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluate on the uploaded dataset
        # AFTER
        val_true, val_pred, val_prob = evaluate_model_on_input_data(
            test_loader, model_instance, device, params
        )

        # --- Prepare structured results ---
        results = []
        for i, fpath in enumerate(saved_file_paths):
            result_entry = {
                "file": os.path.basename(fpath),
                "predicted_label": int(val_pred[i]) if i < len(val_pred) else None,
                "probabilities": val_prob[i] if i < len(val_prob) else None
            }
            results.append(result_entry)

        print(f"Prediction complete for {len(saved_file_paths)} file(s).")
        return jsonify({
            "message": "Prediction successful",
            "total_files": len(saved_file_paths),
            "results": results
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


 

    filelist = glob.glob(os.path.join('/tmp/uploads', '**', '*.csv'), recursive=True)
    print(f"Predict: Found {len(filelist)} CSV files in /tmp/uploads for dataset creation.")  
    # Now pass the saved file paths to WifiCSIDataset
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
    
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

    
    print(f"Dataset created with {len(dataset)} samples from uploaded files.")
    
     
    val_true, val_pred, val_prob = evaluate_model_on_input_data(dataset)
    
    # Delete uploaded files after processing
    for file_path in saved_file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
    val_json = create_json_message(val_true, val_pred, val_prob)
    print(f"Returning JSON: {val_json}")
    return val_json

 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)