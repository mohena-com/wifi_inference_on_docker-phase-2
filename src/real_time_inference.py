import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys
from config_reader import ConfigReader
import time
import warnings
import pandas as pd
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_logging(log_dir):
    """clear previous logging handlers"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'real_time_inference_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    
    return log_file

def load_best_model(model_path, config):
    """Load the best trained model"""
    try:
        # We assume best model is a PyTorch .pt file in the configured models folder.
        model_file = Path(model_path)

        # If provided path doesn't point to a file, search the configured model directory for .pt files
        if not model_file.exists():
            model_dir = Path(config.get('model_save_path', 'models'))
            logging.info(f"Model file {model_path} not found. Searching for .pt files in {model_dir}")
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

            candidates = sorted(model_dir.glob('*.pt')) + sorted(model_dir.glob('*.pth'))
            if not candidates:
                raise FileNotFoundError(f"No .pt/.pth model files found in {model_dir}")

            # choose the most recently modified .pt/.pth file
            model_file = max(candidates, key=lambda p: p.stat().st_mtime)

        logging.info(f"Loading PyTorch model from: {model_file}")

        # Load with torch
        try:
            import torch
            from torch import nn
        except Exception as e:
            logging.error("PyTorch is required to load .pt models but is not available: %s", e)
            raise

        loaded = torch.load(str(model_file), map_location='cpu')

        # If the saved object is an nn.Module, use it. If it's a checkpoint dict containing state_dict,
        # we cannot reconstruct the architecture here — user should save the full model or provide the class.
        if isinstance(loaded, nn.Module):
            pt_model = loaded
        elif isinstance(loaded, dict) and 'model_state_dict' in loaded:
            raise RuntimeError("Loaded checkpoint contains 'model_state_dict'. Provide the model class to load state_dict.")
        else:
            # In many cases torch.save(model) serializes the module object itself; treat as module
            pt_model = loaded

        pt_model.eval()

        class TorchModelWrapper:
            """Wrap a PyTorch model to provide a predict(x) API that returns numpy probabilities."""
            def __init__(self, model):
                self.model = model

            def predict(self, x, verbose=0):
                import torch
                with torch.no_grad():
                    t = torch.from_numpy(np.asarray(x)).float()
                    out = self.model(t)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    try:
                        arr = out.cpu().numpy()
                    except Exception:
                        arr = out.detach().cpu().numpy()
                    # If outputs look like logits (batch x classes), convert to probabilities
                    if arr.ndim == 2:
                        e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
                        probs = e / np.sum(e, axis=1, keepdims=True)
                        return probs
                    return arr

        wrapped = TorchModelWrapper(pt_model)
        logging.info(f"PyTorch model loaded and wrapped for predict() from: {model_file}")
        return wrapped

    except Exception as e:
        logging.error(f"Error loading PyTorch model: {e}")
        raise


def load_test_data(base_dir, gait_filename):
    from DS_WifiCSIDataset import WifiCSIDataset
    import os
    import glob
    """Load test data from CSV file"""
    try:
        filelist = glob.glob(os.path.join(base_dir, '**', gait_filename), recursive=True)
        dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)    
        
        logging.info(f"Loaded test data: {len(dataset)} samples")
        return dataset
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def preprocess_csi_data(csi_data, config):
    """Preprocess CSI data for inference"""
    try:
        # Reshape data according to model's input shape
        input_shape = config.get_tuple('input_shape')
        csi_data = csi_data.reshape(-1, *input_shape)
        
        # Normalize data if needed (using the same normalization as training)
        csi_data = csi_data.astype(np.float32)
        
        return csi_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def predict_activity(model, csi_data, config):
    """Make prediction on CSI data"""
    try:
        # Get prediction
        predictions = model.predict(csi_data, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        
        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise

def process_test_data(model, test_data, config):
    """Process test data for inference"""
    try:
       
        logging.info("Starting inference on test data...")
        print("\nTest Data Inference Started")
        print("Processing each sample...\n")
        
        total_samples = len(test_data)
        correct_predictions = 0
        
        for i, (sample, true_label) in enumerate(zip(test_data, test_labels)):
            try:
                # Preprocess data
                processed_data = preprocess_csi_data(sample.reshape(1, -1), config)
                
                # Get prediction
                predicted_class, confidence = predict_activity(model, processed_data, config)
                
                # Print results
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                predicted_activity = activity_labels[predicted_class[0]]
                true_activity = activity_labels[int(true_label)]
                conf = confidence[0] * 100
                
                # Check if prediction is correct
                is_correct = predicted_class[0] == int(true_label)
                if is_correct:
                    correct_predictions += 1
                
                print(f"[{timestamp}] Sample {i+1}/{total_samples}")
                print(f"    True Activity: {true_activity}")
                print(f"    Predicted Activity: {predicted_activity} (Confidence: {conf:.2f}%)")
                print(f"    {'✓ Correct' if is_correct else '✗ Incorrect'}\n")
                
                # Add small delay to simulate real-time processing
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue
        
        # Print final accuracy
        accuracy = (correct_predictions / total_samples) * 100
        print(f"\nInference Complete!")
        print(f"Total Samples: {total_samples}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
                
    except Exception as e:
        logging.error(f"Error in test data processing: {e}")
        raise



def main():
    
    # Setup logging
    log_file = setup_logging(f"{args.base_path}/logs")
    logging.info(f"Logging initialized. Log file: {log_file}")
    logging.info(f"Running in Colab: {in_colab}")
    
    # Load configuration
    config = ConfigReader(f"../config/har_infer_config.properties")
    
    # Find and load best model
    model_save_dir = Path(config.get('model_save_path'))
    try:        
        model = load_best_model(str(best_model_path), config)
        logging.info(f"Successfully loaded best model from: {best_model_path}")
        
        # Load test data
        base_dir = config.get("local_data_path")
        gait_filename = config.get("file_name_for_gait")

        test_data = load_test_data(base_dir, gait_filename)

        # Process test data
        process_test_data(model, test_data, config)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        return

if __name__ == "__main__":
    main()

# python3.11 real_time_inference.py /Users/sanjeev/VNIT/FINAL_PRJ_PHASE2 wifi_project har_config.properties False

# python3.11 real_time_inference.py /Users/sanjeev/VNIT/FINAL_PRJ_PHASE2 wifi_project har_config.properties False