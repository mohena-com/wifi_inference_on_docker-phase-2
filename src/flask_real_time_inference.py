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
''' 
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
'''
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
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            print(f"CSI shape: {csi_seq.shape}, META shape: {meta_seq.shape}, outputs: {outputs.shape}")
            #for i, (o, a, b) in enumerate(zip(outputs, probs, preds), start=1):
             #   print(f"{i}==>outputs: {o}, probs: {a}, pred: {b}")

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
                print(f"✅  VERIFICATION of Prediction for Batch {batch_idx + 1}")
                for y, p in zip(labels.squeeze().tolist(), preds.squeeze().tolist()):
                    print(f"     true:{y}  pred:{p} = {y == p} Prediction")
                total += labels.size(0)
                val_true.extend(labels.cpu().numpy().tolist())
                print(f"Batch {batch_idx + 1}: loss={loss.item():.4f}, acc={(preds == labels).sum().item()}/{labels.size(0)}")
            else:
                print(f"Batch {batch_idx + 1}: No valid label/subject found → inference-only mode.")
    
    print(f"probs:{len(val_prob)},  pred:{len(val_pred)},   true:{len(val_true)}")
   # for a, b, c in zip(val_prob, val_pred, val_true):
    #    print(f"probs:{a}, pred:{b},  true:{c}")
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

@app.route('/gaitid/predict', methods=['POST'])
def predict():
    import os
    import json
    import math
    from datetime import datetime
    from collections import defaultdict, Counter
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            if current_len < min_rows and current_len > 0:
                pad_rows = min_rows - current_len
                last_row = df.iloc[-1:]
                pad_df = pd.concat([last_row] * pad_rows, ignore_index=True)
                df = pd.concat([df, pad_df], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Padded {os.path.basename(csv_path)} from {current_len} → {len(df)} rows.")
            elif current_len == 0:
                print(f"Warning: {os.path.basename(csv_path)} is empty; skipping padding.")
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

    # --- Load dataset and evaluate (we'll iterate test_loader directly to build full JSON) ---
    try:
        batch_size = 16
        test_dataset = WifiCSIDataset(
            logger=logger,
            file_list=saved_file_paths,
            window_size=128,
            stride=64
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Debug prints
        print("DEBUG: dataset size (len):", len(test_dataset))
        print("DEBUG: expected batches (ceil):", math.ceil(len(test_dataset) / batch_size))
        print("DEBUG: expected batches (floor = drop_last True):", len(test_dataset) // batch_size)

        all_windows = []         # list of per-window dicts (order preserved)
        batch_summaries = []     # stores batch-level metadata (size, loss, correct, total)
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        model_instance.eval()
        device_local = device  # use existing device variable

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move inputs
                csi_seq = batch["csi_seq"].to(device_local)
                meta_seq = batch["metadata_seq"].to(device_local)

                # forward pass
                outputs = model_instance(csi_seq, meta_seq)        # (N, C)
                probs_tensor = F.softmax(outputs, dim=1)           # (N, C)
                preds_tensor = torch.argmax(outputs, dim=1)        # (N,)

                # optional labels
                b_labels = batch.get("label")         # index (0..)
                b_labels_raw = batch.get("label_raw") # raw subject id (e.g., 1..)
                b_files = batch.get("file")
                b_starts = batch.get("start")
                b_wins = batch.get("window_size")

                # compute loss+accuracy for this batch if labels exist
                batch_loss = None
                batch_correct = 0
                batch_total = 0
                if b_labels is not None and hasattr(b_labels, "numel") and b_labels.numel() > 0:
                    # normalize shapes
                    if b_labels.dim() == 2 and b_labels.size(-1) == 1:
                        b_labels = b_labels.squeeze(-1)
                    elif b_labels.dim() == 0:
                        b_labels = b_labels.unsqueeze(0)
                    b_labels = b_labels.long().to(device_local)

                    criterion = nn.CrossEntropyLoss()
                    batch_loss = float(criterion(outputs, b_labels).item())
                    running_loss += batch_loss

                    batch_correct = int((preds_tensor == b_labels).sum().item())
                    batch_total = int(b_labels.size(0))
                    running_correct += batch_correct
                    running_total += batch_total

                # extract numpy arrays for loop
                preds = preds_tensor.cpu().numpy().tolist()
                probs_np = probs_tensor.cpu().numpy()  # shape (N, C)

                # ensure b_files/stars/wins/lables_raw indexing works for batch_size>=1
                for i in range(len(preds)):
                    pred_idx = int(preds[i])
                    # mapping index -> raw subject id: adjust if your mapping differs
                    pred_raw = int(pred_idx + 1)

                    prob_row = probs_np[i].tolist()
                    # top-3 probabilities
                    topk = list(map(int, np.argsort(prob_row)[::-1][:3]))
                    top3 = [{"label_idx": int(k), "label_raw": int(k + 1), "prob": float(prob_row[k])} for k in topk]

                    # true values (if exist)
                    true_idx = None
                    true_raw = None
                    if b_labels is not None and hasattr(b_labels, "numel") and b_labels.numel() > 0:
                        try:
                            # handle tensor/list indexing
                            if hasattr(b_labels, "cpu"):
                                true_idx = int(b_labels.cpu().numpy().tolist()[i])
                            else:
                                true_idx = int(b_labels[i])
                            if b_labels_raw is not None and hasattr(b_labels_raw, "numel") and b_labels_raw.numel() > 0:
                                if hasattr(b_labels_raw, "cpu"):
                                    true_raw = int(b_labels_raw.cpu().numpy().tolist()[i])
                                else:
                                    true_raw = int(b_labels_raw[i])
                            else:
                                true_raw = int(true_idx + 1) if true_idx is not None else None
                        except Exception:
                            true_idx = None
                            true_raw = None

                    # file / start / window_size
                    # b_files may be list or a single repeated scalar
                    if isinstance(b_files, (list, tuple)):
                        file_val = str(b_files[i])
                    else:
                        # if it's tensor or scalar
                        try:
                            file_val = str(b_files[i])
                        except Exception:
                            file_val = str(b_files)
                    try:
                        if hasattr(b_starts, "cpu"):
                            start_val = int(b_starts.cpu().numpy().tolist()[i])
                        else:
                            start_val = int(b_starts[i]) if isinstance(b_starts, (list, tuple)) else int(b_starts)
                    except Exception:
                        start_val = None
                    try:
                        if hasattr(b_wins, "cpu"):
                            win_val = int(b_wins.cpu().numpy().tolist()[i])
                        else:
                            win_val = int(b_wins[i]) if isinstance(b_wins, (list, tuple)) else int(b_wins)
                    except Exception:
                        win_val = None

                    all_windows.append({
                        "file": os.path.basename(file_val) if file_val is not None else None,
                        "start": start_val,
                        "window_size": win_val,
                        "true_idx": true_idx,
                        "true_raw": true_raw,
                        "pred_idx": pred_idx,
                        "pred_raw": pred_raw,
                        "probs": prob_row,
                        "top3": top3
                    })

                batch_summaries.append({
                    "batch_index": batch_idx + 1,
                    "batch_size": len(preds),
                    "loss": batch_loss,
                    "correct": batch_correct,
                    "total": batch_total
                })

        # final overall stats
        total_windows = len(all_windows)
        total_correct = sum(1 for w in all_windows if (w["true_idx"] is not None and w["true_idx"] == w["pred_idx"]))
        total_incorrect = total_windows - total_correct

        # Build batches array (slice the all_windows list according to batch_summaries)
        batches = []
        cursor = 0
        for bs in batch_summaries:
            cnt = bs["batch_size"]
            slice_windows = all_windows[cursor: cursor + cnt]
            cursor += cnt
            # per-window JSON for UI (only include what UI needs)
            windows_json = []
            for w in slice_windows:
                windows_json.append({
                    "true": w["true_raw"],
                    "pred": w["pred_raw"],
                    "correct": (w["true_idx"] is not None and w["true_idx"] == w["pred_idx"]),
                    "top3": w["top3"],
                    # optionally include small probs (top3 already includes probs)
                    # omit full w["probs"] to reduce JSON size if you want
                })
            batches.append({
                "batch_index": bs["batch_index"],
                "windows": windows_json,
                "loss": bs["loss"],
                "acc": f"{bs['correct']}/{bs['batch_size']}" if bs.get("total", 0) else None
            })

        # Build per-file majority-vote summary
        by_file_preds = defaultdict(list)
        by_file_truths = defaultdict(list)
        for w in all_windows:
            fname = w["file"] or "unknown"
            if w["pred_raw"] is not None:
                by_file_preds[fname].append(w["pred_raw"])
            if w["true_raw"] is not None:
                by_file_truths[fname].append(w["true_raw"])

        per_file_summary = []
        files_union = sorted(set(list(by_file_preds.keys()) + list(by_file_truths.keys())))
        for fname in files_union:
            preds = by_file_preds.get(fname, [])
            truths = by_file_truths.get(fname, [])
            maj_pred = Counter(preds).most_common(1)[0][0] if preds else None
            maj_true = Counter(truths).most_common(1)[0][0] if truths else None
            per_file_summary.append({
                "file": fname,
                "majority_predicted": int(maj_pred) if maj_pred is not None else None,
                "majority_actual": int(maj_true) if maj_true is not None else None,
                "n_windows": len(preds)
            })

        # Build summary object
        summary = {
            "total_windows": total_windows,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "notes": f"probs:{total_windows}, pred:{total_windows}, true:{total_windows}"
        }

        result_json = {
            "batches": batches,
            "summary": summary,
            "per_file_summary": per_file_summary
        }

        # Save JSONs: timestamped + latest
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_ts = os.path.join(upload_dir, f"prediction_result_{ts}.json")
        out_latest = os.path.join(upload_dir, "prediction_result_latest.json")
        try:
            with open(out_ts, "w") as fh:
                json.dump(result_json, fh, indent=2)
            with open(out_latest, "w") as fh2:
                json.dump(result_json, fh2, indent=2)
            print(f"[INFO] Saved prediction JSON -> {out_ts} and {out_latest}")
        except Exception as e:
            logger.error(f"Failed to save prediction JSON: {e}")

        print(f"Prediction complete for {len(saved_file_paths)} file(s).")
        # optional cleanup call if you have it
        try:
            cleanup_files()
        except Exception:
            pass

        return jsonify(result_json), 200

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500



@app.route('/gaitid/predict1', methods=['POST'])
def predict1():
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

        import math
        # expected batches:
        print("DEBUG: expected batches (ceil):", math.ceil(len(test_dataset) / 16))
        print("DEBUG: expected batches (floor = drop_last True):", len(test_dataset) // 16)

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

        cleanup_files()

        return jsonify({
            "message": "Prediction successful",
            "total_files": len(saved_file_paths),
            "results": results
        })

    except Exception as e:
        logger.exception("Prediction failed")
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


 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)