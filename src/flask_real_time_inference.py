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
    print(f"predict called")
    data = request.get_json()
    print(f"Received data: {data}")
    return jsonify(data)

@app.route('/gait_id/predict1', methods=['POST'])
def predict1():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        if not data or 'csi_data' not in data:
            return jsonify({'error': 'Missing csi_data in request'}), 400
        csi_data = data['csi_data']
        print(f"Received csi_data: {csi_data}")
        # Handle input shape: flatten if needed, then reshape
        arr = np.array(csi_data)
        # If shape is (batch, 1, 90), squeeze to (batch, 90)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        # If shape is (batch, 90), expand last dim to (batch, 90, 1)
        if arr.ndim == 2 and arr.shape[1] == 90:
            arr = arr[..., np.newaxis]
        # Now arr should be (batch, 90, 1)
        processed_data = arr.astype(np.float32)
        print(f"Processed csi_data shape: {processed_data.shape}")
        predicted_class, confidence = predict_activity(model, processed_data)

        print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        result = [
            {
                'predicted_activity': activity_labels[int(predicted_class[i])],
                'confidence': float(confidence[i])
            }
            for i in range(len(predicted_class))
        ]
        print(f"Prediction result: {result}")

        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)