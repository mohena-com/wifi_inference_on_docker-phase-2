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

def evalute_model_on_input_data(model, input_data):
    from CSI_Model_Eval_helper import get_best_model_and_params

    model_instance, params, total_params, device = get_best_model_and_params()  
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    val_true, val_pred, val_prob = [], [], []
    non_blocking_flag = True if device.type == "cuda" else False
    with torch.no_grad():
        for batch in test_loader:
            csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
            meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
            labels = batch["label"].squeeze().to(device, non_blocking=non_blocking_flag)

            if torch.isnan(csi_seq).any() or torch.isinf(csi_seq).any():
                csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(meta_seq).any() or torch.isinf(meta_seq).any():
                meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)

            # same per-batch normalization used in training
            try:
                mean = csi_seq.mean(dim=(0, 1), keepdim=True)
                std = csi_seq.std(dim=(0, 1), keepdim=True) + 1e-8
                csi_seq = (csi_seq - mean) / std
            except Exception:
                pass

            outputs = model(csi_seq, meta_seq)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_true.extend(labels.cpu().numpy().tolist())
            val_pred.extend(preds.cpu().numpy().tolist())
            val_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy().tolist())
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

@app.route('/gaitid/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist('file')  # if multiple files, or just request.files.values()
    print(f"Received {len(uploaded_files)} files for prediction.  {uploaded_files}  ")
    saved_file_paths = []
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join('/tmp/uploads', filename)
        uploaded_file.save(save_path)
        saved_file_paths.append(save_path)

    filelist = glob.glob(os.path.join('/tmp/uploads', '**', '*.csv'), recursive=True)

    # Now pass the saved file paths to WifiCSIDataset
    dataset = WifiCSIDataset(logger, filelist, window_size=1, stride=64)

    # Continue with your logic using dataset...
    print(f"Dataset created with {len(dataset)} samples from uploaded files.")
    return jsonify({"message":  len(dataset)})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)