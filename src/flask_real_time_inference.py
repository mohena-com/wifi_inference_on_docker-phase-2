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

 

@app.route('/gaitid/predict', methods=['POST'])
def predict():
    # List of file keys (input names)
    file_keys = request.files.keys()
    print("Uploaded file field names:", file_keys)

    # If you want to access all files
    uploaded_files = request.files.to_dict()
    for key, file in uploaded_files.items():
        print(f"File field: {key}, Filename: {file.filename}")

    # Process the specific file or return a message
    if 'file' in uploaded_files:
        file = uploaded_files['file']
        # read/process the file
        data = file.read()
        return {"message": f"Received file {file.filename} of size {len(data)} bytes."}
    else:
        return {"error": "No files uploaded"}, 400

 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)