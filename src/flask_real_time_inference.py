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
from CSI_Model_Eval_helper import get_best_model_and_params, get_best_model_path
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

config = None

"""
Initialize model once at startup. Returns (model_instance, device, params, total_params, best_model_path).
"""
CONFIG_FILE = os.environ.get('CONFIG_FILE', 'config/gait_id_config.properties')
print(f"🧩 Using config file: {CONFIG_FILE}", "config")

# read config and determine best model path
config = ConfigReader(CONFIG_FILE)

state_dict = {
    "model_state" : None, 
    "scaler_meta" : None,
    "scaler_mag" : None,
    "scaler_phase" : None, 
    "feature_info" : None
}

# Flask app
app = Flask(__name__)
CORS(app)  # <-- Add here 
import torch
def init_model():
    
    best_model_path = get_best_model_path(config)
    print(f"📦 Best model path: {best_model_path}", "model_path")
    # get model skeleton and metadata (do not load weights yet)
    model_instance, params, total_params, device = get_best_model_and_params(str(best_model_path))
    print(f"📦 Model class: {model_instance.__class__.__name__}  path: {best_model_path}", "load_model")
    print(f"ℹ️ params: {params} total_params: {total_params} device: {device}", "info")
    print(f"📦 Model instance: {model_instance}")

    # load checkpoint safely
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
    except Exception as e:
        print(f"❌ Failed to load checkpoint from {best_model_path}: {e}", "error")
        raise

    # determine the correct state dict
    state_dict = None
    if isinstance(checkpoint, dict):
        # prefer common keys
        for key in ("state_dict", "model_state_dict", "model", "scaler_meta","scaler_mag" , "scaler_phase"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"📦 Found '{key}' in checkpoint; using it as state_dict" )
                break
        if state_dict is None:
            # sometimes checkpoint is the state_dict already, or contains nested keys
            # if it looks like a state_dict (mapping of tensors), use it directly
            state_dict = checkpoint
            print("📦 Using checkpoint dict as state_dict (fallback)" )
    else:
        # checkpoint is not a dict — assume it's the state dict object itself
        state_dict = checkpoint
        print(f"📦 Checkpoint is not a dict; using as state_dict")

    # load weights into model and set eval mode
    try:
        model_instance.load_state_dict(state_dict)
        print(f"✅ Loaded state_dict {state_dict}")
        #model_state  = state_dict["model_state"]
        scaler_meta  = state_dict["scaler_meta"]
        scaler_mag   = state_dict["scaler_mag"]
        scaler_phase = state_dict["scaler_phase"]
        #feature_info = state_dict["feature_info"]   
       
        #model_instance.eval()
        print(f"✅ Loaded model weights from: {best_model_path}", "success")
    except Exception as e:
        print(f"❌ Error when loading state_dict into model: {e}", "error")
        raise

    return model_instance, device, params, total_params, best_model_path

import threading

_model_lock = threading.Lock()
model_instance = device = params = total_params = best_model_path = None
is_model_loaded = False
def ensure_model_loaded():
    global model_instance, device, params, total_params, best_model_path
    if model_instance is None:
        with _model_lock:
            if model_instance is None :
                print(f"📦 INIT model START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                model_instance, device, params, total_params, best_model_path = init_model()
                print(f"📦 INIT model DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                is_model_loaded = True
 

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
    ensure_model_loaded()
    upload_dir = '/tmp/uploads'
    os.makedirs(upload_dir, exist_ok=True)

    uploaded_files = request.files.getlist('file')
    logger.info(f"📂 Received {len(uploaded_files)} files for prediction: {[f.filename for f in uploaded_files]}")

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
            logger.exception(f"❌ Padding failed for {csv_path}: {e}")

    # Save uploads
    try:
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({"error": "No files uploaded"}), 400

        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.filename)
            save_path = os.path.join(upload_dir, filename)
            uploaded_file.save(save_path)
            logger.info(f"💾 Saved uploaded file to {save_path}")
            pad_csv_if_needed(save_path, min_rows=128)
            saved_file_paths.append(save_path)

    except Exception as e:
        logger.exception("❌ Failed saving uploaded files")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    # Build dataset, loader and run inference
    try:
        w_size = config.get_int('window_size', 64)
        s_size = config.get_int('stride', 32)
        # batch_size = int(params.get('batch_size', 8))
        batch_size = config.get_int('batch_size', 8)

        print(f"🧩 Using window_size={w_size}, stride={s_size}, batch_size={batch_size} for dataset creation."  )
        print(f"🧩 Creating test dataset and loader with batch_size={batch_size}")
        test_dataset = WifiCSIDataset(logger=logger, file_list=saved_file_paths, window_size=w_size, stride=s_size)        

        test_dataset.set_scalers(state_dict["scaler_meta"], state_dict["scaler_mag"], state_dict["scaler_phase"] )
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Debug
        print(f"🔍 dataset size (len): {len(test_dataset)}")
        print(f"🔍 expected batches (ceil): {math.ceil(len(test_dataset) / batch_size)}")
        print(f"🔍 expected batches (floor): {len(test_dataset) // batch_size}")

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
            for batch_idx, batch in enumerate(test_loader, start=1):               
                print(f"")
                print(f"🧩 Processing batch {batch_idx}/{len(test_loader)}")
                # Move inputs
                csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
                meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
                b_labels = batch["label"].reshape(-1).to(device, non_blocking=non_blocking_flag).long()
                print(f"    🧩 Labels:       {b_labels.cpu().numpy().tolist()}")

                # guard against NaN/Inf values in inputs/labels
                if torch.isnan(csi_seq).any() or torch.isinf(csi_seq).any():
                    csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                if torch.isnan(meta_seq).any() or torch.isinf(meta_seq).any():
                    meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)
                if torch.isnan(b_labels).any() or torch.isinf(b_labels).any():
                    b_labels = torch.nan_to_num(b_labels, nan=0).long()           

                # Forward
                #with torch.no_grad():  # disable gradient tracking for speed & memory efficiency
                outputs = model_instance(csi_seq, meta_seq)   # (N, C)
                print(f"    🔍 outputs : {outputs}")

                probs_tensor = torch.softmax(outputs, dim=1)  # (N, C)
                print(f"    🔍 probs_tensor : {probs_tensor}")

                preds_tensor = torch.argmax(outputs, dim=1)   # (N,)               
                print(f"    🔍 Predictions : {preds_tensor.cpu().numpy().tolist()} ")
 
                b_files = batch.get("file")
                b_starts = batch.get("start")
                b_wins = batch.get("window_size")
                #print(f"💻 Batch {batch_idx }: Processing {len(preds_tensor)} windows")
                # compute batch-level loss if labels exist
                batch_loss = None
                batch_correct = 0
                batch_total = 0

                # convert to cpu numpy
                preds = preds_tensor.cpu().numpy().tolist()
                probs_np = probs_tensor.cpu().numpy()  # shape (N, C)
                print(f"    🔄 Processing")
                # iterate windows in this batch
                if b_labels.dim() == 0:
                    b_labels = b_labels.unsqueeze(0)
                for i in range(len(b_labels)):
                    pred = int(preds[i])
                    lab = int(b_labels[i])
                    # mapping index -> subject id (adjust if needed)
                    print(f"        💻 Window {i + 1}/{len(preds)}: pred={pred}, label={lab}, {pred==lab}")

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
                print(f"    ✅ VERIFICATION of Prediction  ✅ {batch_correct}/{batch_total} correct, ❌ loss: {batch_loss}")

                inference_response.batches.append(BatchResult(
                 batch=batch_idx,
                 true_value=b_labels.cpu().numpy().tolist(),
                 predicted_value=preds,
                 correct=batch_correct,
                 total=batch_total,
                 accuracy=f"{batch_correct}/{batch_total}",
                 loss=batch_loss
                ))

        try:
            file_list = cleanup()
        except Exception:
            logger.exception("cleanup_files failed")
        inference_response.file_list=file_list
        from flask import jsonify
       # response_data = inference_result()
        return jsonify(inference_response), 200

    except Exception as e:
        logger.exception("❌ Prediction failed")
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
        logger.exception("❌ serve_prediction_latest failed")
        return jsonify({"error": str(e)}), 500

def cleanup():
    filelist = glob.glob(os.path.join('/tmp/uploads', '**', '*.csv'), recursive=True)
    print(f"🧩 Predict: Found {len(filelist)} CSV 📂 files in /tmp/uploads for dataset creation.")  
    # Delete uploaded files after processing
    for file in filelist:
        try:
            os.remove(file)
            print(f"❌ Deleted uploaded 📂 file: {file}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file}: {e}")
    
    model_instance = device = params = total_params = best_model_path = None


    return filelist

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)
