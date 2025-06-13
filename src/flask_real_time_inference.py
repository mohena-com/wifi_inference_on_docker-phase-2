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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_best_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def preprocess_csi_data(csi_data, input_shape):
    try:
        csi_data = np.array(csi_data).reshape(-1, *input_shape)
        csi_data = csi_data.astype(np.float32)
        return csi_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def predict_activity(model, csi_data):
    try:
        predictions = model.predict(csi_data, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise

# Flask app
app = Flask(__name__)

# Load config and model at startup
CONFIG_FILE = os.environ.get('CONFIG_FILE', 'config/har_infer_config.properties')
config = ConfigReader(CONFIG_FILE)
print(f"Using config file: {CONFIG_FILE}")
print(f"Using config file: {config}")
input_shape = config.get_tuple('input_shape')
print(f"Input shape for model: {input_shape}")
print(f"model path: {config.get('model_save_path')}")

model_save_dir = Path(config.get('model_save_path'))
print(f"Model save directory: {model_save_dir.resolve()}")

best_model_pattern = config.get('best_model_pattern')
model_files = list(model_save_dir.glob(best_model_pattern))
print(f"Found model files: {model_files}")
if not model_files:
    raise FileNotFoundError("No model files found")
fold_numbers = [int(str(f).split('_')[-1].split('.')[0]) for f in model_files]
best_fold = max(fold_numbers)
best_model_path = model_save_dir / f'best_model_fold_{best_fold}.keras'
print(f"Loading best model from: {best_model_path}")
model = load_best_model(str(best_model_path))
print(f"Loaded model: {model} from {best_model_path}")
activity_labels = [
    "Walking", "Running", "Sitting", "Standing", "Lying",
    "Climbing Up", "Climbing Down", "Jumping", "Falling", "Idle"
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'csi_data' not in data:
            return jsonify({'error': 'Missing csi_data in request'}), 400
        csi_data = data['csi_data']

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

        predicted_class, confidence = predict_activity(model, processed_data)
        result = [
            {
                'predicted_activity': activity_labels[int(predicted_class[i])],
                'confidence': float(confidence[i])
            }
            for i in range(len(predicted_class))
        ]
        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)