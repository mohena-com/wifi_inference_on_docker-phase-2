from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
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

from CSI_Model_Eval_helper import get_best_model_and_params
from CSI_Model_Eval_helper import get_best_model_path
best_model_path = get_best_model_path(config)
model_instance, params, total_params, device = get_best_model_and_params(str(best_model_path))

print(f"Loaded model: {model_instance} from {best_model_path}")
print(f"params: {params} device {device}")
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

from dataclasses import dataclass, asdict
from typing import List

@dataclass
class BatchResult:
    batch: int
    true_value: List[int]
    predicted_value: List[int]
    correct: int
    total: int
    accuracy: str
    loss: float


@dataclass
class InferenceResponse:
    file_list: List[str]
    batches: List[BatchResult]




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

# --- drop-in replacement for evaluate_model_on_input_data in flask_real_time_inference.py ---


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
    print("Logger initialized")
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

# Paste this into flask_real_time_inference.py replacing the old /gaitid/predict function

from datetime import datetime
from collections import defaultdict, Counter
import json
import numpy as np
import torch.nn.functional as F

@app.route('/gaitid/predict', methods=['POST'])
def predict():
    """
    Accepts uploaded CSV file(s) via multipart form 'file'.
    Runs inference using the existing model_instance and returns a rich JSON:
      {
        "batches": [ { batch_index, windows:[{true,pred,correct,top3}], loss, acc }, ... ],
        "summary": { total_windows, total_correct, total_incorrect, notes },
        "per_file_summary": [ { file, majority_predicted, majority_actual, n_windows }, ... ]
      }
    Also writes prediction_result_<ts>.json and prediction_result_latest.json to upload_dir.
    """
    import os
    from werkzeug.utils import secure_filename
    import traceback
    import torch
    import math

    upload_dir = '/tmp/uploads'
    os.makedirs(upload_dir, exist_ok=True)

    uploaded_files = request.files.getlist('file')
    logger.info(f"Received {len(uploaded_files)} files for prediction: {[f.filename for f in uploaded_files]}")

    saved_file_paths = []

    # Helper: pad CSVs to min_rows if needed
    def pad_csv_if_needed(csv_path, min_rows=128):
        try:
            df = pd.read_csv(csv_path)
            current_len = len(df)
            if current_len == 0:
                logger.warning(f"{os.path.basename(csv_path)} is empty. Skipping.")
                return
            if current_len < min_rows:
                pad_rows = min_rows - current_len
                last_row = df.iloc[-1:]
                pad_df = pd.concat([last_row] * pad_rows, ignore_index=True)
                df = pd.concat([df, pad_df], ignore_index=True)
                df.to_csv(csv_path, index=False)
                logger.info(f"Padded {os.path.basename(csv_path)} from {current_len} → {len(df)} rows.")
            else:
                logger.info(f"{os.path.basename(csv_path)} has {current_len} rows — no padding needed.")
        except Exception as e:
            logger.exception(f"Padding failed for {csv_path}: {e}")

    # Save uploads
    try:
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({"error": "No files uploaded"}), 400

        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.filename)
            save_path = os.path.join(upload_dir, filename)
            uploaded_file.save(save_path)
            logger.info(f"Saved uploaded file to {save_path}")
            pad_csv_if_needed(save_path, min_rows=128)
            saved_file_paths.append(save_path)

    except Exception as e:
        logger.exception("Failed saving uploaded files")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    # Build dataset, loader and run inference
    try:
        batch_size = 16
        test_dataset = WifiCSIDataset(logger=logger, file_list=saved_file_paths, window_size=128, stride=64)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Debug
        logger.info(f"DEBUG: dataset size (len): {len(test_dataset)}")
        logger.info(f"DEBUG: expected batches (ceil): {math.ceil(len(test_dataset) / batch_size)}")
        logger.info(f"DEBUG: expected batches (floor): {len(test_dataset) // batch_size}")

        # We'll iterate test_loader here, but leverage the existing evaluate_model_on_input_data
        # which already prints batch info and returns (val_true, val_pred, val_prob).
        # However we need per-window top3 and per-batch grouping — so re-run inference loop here
        # but we reuse model_instance and device to ensure consistency.

        model_instance.eval()
        device_local = device
        all_windows = []         # list of window dicts in order
        batch_summaries = []
        non_blocking_flag = (device.type == "cuda")
        inference_response = InferenceResponse(file_list=[], batches=[])

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):

                

                print(f"\n🧩 Processing batch {batch_idx + 1}/{len(test_loader)}")
                # Move inputs
                csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
                meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
                b_labels = batch["label"].squeeze().to(device, non_blocking=non_blocking_flag)
                print(f"🧩 Batch {batch_idx + 1} labels:          {b_labels.cpu().numpy().tolist()}")

                # guard against NaN/Inf values in inputs/labels
                if torch.isnan(csi_seq).any() or torch.isinf(csi_seq).any():
                    csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                if torch.isnan(meta_seq).any() or torch.isinf(meta_seq).any():
                    meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                if torch.isnan(b_labels).any() or torch.isinf(b_labels).any():
                    b_labels = torch.nan_to_num(b_labels, nan=0).long()

                # same per-batch normalization used in training
                try:
                    mean = csi_seq.mean(dim=(0, 1), keepdim=True)
                    std = csi_seq.std(dim=(0, 1), keepdim=True) + 1e-8
                    csi_seq = (csi_seq - mean) / std
                except Exception:
                    pass

                
                # Forward
                with torch.no_grad():  # disable gradient tracking for speed & memory efficiency
                    outputs = model_instance(csi_seq, meta_seq)   # (N, C)
                    probs_tensor = torch.softmax(outputs, dim=1)  # (N, C)
                    preds_tensor = torch.argmax(outputs, dim=1)   # (N,)               
                print(f"🧩 Predictions for Batch {batch_idx + 1}: {preds_tensor.cpu().numpy().tolist()} ")
 
                b_files = batch.get("file")
                b_starts = batch.get("start")
                b_wins = batch.get("window_size")
                print(f"💻 Batch {batch_idx + 1}: Processing {len(preds_tensor)} windows")
                # compute batch-level loss if labels exist
                batch_loss = None
                batch_correct = 0
                batch_total = 0
                if b_labels is not None and hasattr(b_labels, "numel") and b_labels.numel() > 0:
                    if b_labels.dim() == 0:
                        b_labels = b_labels.unsqueeze(0)
                    b_labels = b_labels.long().to(device_local)
                    criterion = torch.nn.CrossEntropyLoss()
                    try:
                        batch_loss = float(criterion(outputs, b_labels).item())
                    except Exception:
                        batch_loss = None
                    batch_correct = int((preds_tensor == b_labels).sum().item())
                    batch_total = int(b_labels.size(0))
                    print(f"✅  VERIFICATION of Prediction for Batch {batch_idx + 1} ")
                    print(f"    ✅ {batch_correct}/{batch_total} correct, ❌ loss: {batch_loss}")

                # convert to cpu numpy
                preds = preds_tensor.cpu().numpy().tolist()
                probs_np = probs_tensor.cpu().numpy()  # shape (N, C)
                print(f"🔄 Batch {batch_idx + 1}: Processing {len(preds)} windows")
                # iterate windows in this batch
                for i in range(len(preds)):
                    pred = int(preds[i])
                    lab = int(b_labels[i])
                    # mapping index -> subject id (adjust if needed)                    
                    print(f"💻 Window {i + 1}/{len(preds)}: pred_={pred}, pred_raw={lab}, {pred==lab}"  )

                inference_response.batches.append(BatchResult(
                 batch=batch_idx + 1,
                 true_value=b_labels.cpu().numpy().tolist(),
                 predicted_value=preds,
                 correct=batch_correct,
                 total=batch_total,
                 accuracy=f"{batch_correct}/{batch_total}",
                 loss=batch_loss
                ))
        # Finished all batches    
        
        # optional cleanup of uploaded csvs (your existing cleanup_files)
        try:
            file_list = cleanup_files()
        except Exception:
            logger.exception("cleanup_files failed")
        inference_response.file_list=file_list
        from flask import jsonify
       # response_data = inference_result()
        return jsonify(inference_response), 200

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

 
# Helper route to serve latest JSON (so Angular can GET /gaitid/prediction_result_latest.json)
@app.route('/gaitid/prediction_result_latest.json', methods=['GET'])
def serve_prediction_latest():
    upload_dir = '/tmp/uploads'
    try:
        latest_path = os.path.join(upload_dir, "prediction_result_latest.json")
        if not os.path.exists(latest_path):
            return jsonify({"error": "no prediction file found"}), 404
        return send_from_directory(upload_dir, "prediction_result_latest.json")
    except Exception as e:
        logger.exception("serve_prediction_latest failed")
        return jsonify({"error": str(e)}), 500






 
def cleanup_files():
    filelist = glob.glob(os.path.join('/tmp/uploads', '**', '*.csv'), recursive=True)
    print(f"Predict: Found {len(filelist)} CSV files in /tmp/uploads for dataset creation.")  
    # Delete uploaded files after processing
    for file in filelist:
        try:
            os.remove(file)
            print(f"Deleted uploaded file: {file}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file}: {e}")
    return filelist


 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)