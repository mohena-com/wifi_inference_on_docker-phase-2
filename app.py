# app.py
import os
import glob
import io
import json
import time
import logging
from pathlib import Path
from werkzeug.utils import secure_filename

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Project imports (assumes these files are in project root)
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_DenseNet1D import DenseNet1D
from DL_CSILSTMNet import CSILSTMNet
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csi_infer")

# ---- Flask app ----
app = Flask(__name__, static_folder="web_ui", static_url_path="/ui")
CORS(app)

# ---- Config / env ----
CR_PATH = os.environ.get("CSI_CONFIG_PATH", "csi_id_config.properties")
cr = ConfigReader(CR_PATH) if Path(CR_PATH).exists() else ConfigReader("csi_id_config.properties")
LOCAL_DATA_PATH = Path(os.environ.get("LOCAL_DATA_PATH", cr.get("local_data_path") or "./data"))
FILE_PATTERN = cr.get("file_name_for_gait") or "*C03*.csv"
DEFAULT_PAYLOAD_PATH = os.environ.get("MODEL_PAYLOAD_PATH", str(Path(cr.get("output_path") or ".") / "best_payload.pt"))

# ---- device ----
def get_device():
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
logger.info(f"Using device: {device}")

# ---- model map + globals ----
MODEL_CLASS_MAP = {
    "DenseNet1D": DenseNet1D,
    "CSILSTMNet": CSILSTMNet,
    "EfficientNet1DLSTM": EfficientNet1DLSTM,
    "MobileNetV3_1D_LSTM": MobileNetV3_1D_LSTM,
}

_model_bundle = {
    "model": None,
    "model_name": None,
    "params": None,
    "scaler_meta": None,
    "scaler_csi": None,
    "window_size": int(os.environ.get("WINDOW_SIZE", 128)),
    "stride": int(os.environ.get("STRIDE", 64)),
    "num_classes": int(os.environ.get("NUM_CLASSES", 31))
}

# ---- payload loader utilities ----
def load_payload(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"payload not found: {path}")
    logger.info(f"Loading payload from {path}")
    data = torch.load(path, map_location="cpu")
    # prefer structured payload
    if isinstance(data, dict) and "state_dict" in data:
        return data
    # else assume the file is a raw state dict
    return {"state_dict": data, "model_name": None, "params": {}}

def create_model_from_payload(payload):
    model_name = payload.get("model_name")
    if not model_name:
        raise ValueError("payload missing 'model_name' entry required to instantiate model")
    if model_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown model_name in payload: {model_name}")
    params = payload.get("params", {}) or {}
    csi_channels = params.get("csi_channels", params.get("csi_input_channels", 99))
    meta_dim = params.get("meta_dim", params.get("meta_input_size", 12))
    num_classes = params.get("num_classes", _model_bundle["num_classes"])

    cls = MODEL_CLASS_MAP[model_name]
    if model_name == "CSILSTMNet":
        model = cls(csi_input_size=csi_channels, meta_input_size=meta_dim,
                    window_size=_model_bundle["window_size"], num_classes=num_classes)
    elif model_name in ("DenseNet1D", "MobileNetV3_1D_LSTM"):
        model = cls(csi_channels=csi_channels, meta_feature_dim=meta_dim, num_classes=num_classes)
    elif model_name == "EfficientNet1DLSTM":
        model = cls(csi_input_channels=csi_channels, meta_input_size=meta_dim, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type")
    return model

def load_model(payload_path=DEFAULT_PAYLOAD_PATH):
    payload = load_payload(payload_path)
    model = create_model_from_payload(payload)
    state = payload["state_dict"]
    try:
        model.load_state_dict(state)
    except Exception:
        # attempt to strip 'module.' prefix if present
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    _model_bundle["model"] = model
    _model_bundle["model_name"] = payload.get("model_name")
    _model_bundle["params"] = payload.get("params", {})
    _model_bundle["scaler_meta"] = payload.get("scaler_meta", None)
    _model_bundle["scaler_csi"] = payload.get("scaler_csi", None)
    logger.info(f"Model loaded: {_model_bundle['model_name']}")
    return True

# try load at startup
try:
    if os.path.exists(DEFAULT_PAYLOAD_PATH):
        load_model(DEFAULT_PAYLOAD_PATH)
except Exception as e:
    logger.exception("startup model load failed: %s", e)

# ---- dataset/parsing/scaling helpers ----
def parse_single_csv_bytes(file_bytes):
    tmp_path = "_tmp_infer_upload.csv"
    with open(tmp_path, "wb") as fh:
        fh.write(file_bytes)
    ds = WifiCSIDataset(logger=logger, file_list=[], window_size=_model_bundle["window_size"], stride=_model_bundle["stride"])
    X_meta, X_csi, y, _, _ = ds.load_csv_as_numpy(tmp_path)
    os.remove(tmp_path)
    return X_meta, X_csi, y

def ensure_scalers():
    if _model_bundle["scaler_meta"] is not None and _model_bundle["scaler_csi"] is not None:
        return
    from sklearn.preprocessing import StandardScaler
    scaler_meta = StandardScaler()
    scaler_csi = StandardScaler()

    files = glob.glob(str(LOCAL_DATA_PATH / "**" / FILE_PATTERN), recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found using pattern {FILE_PATTERN} under {LOCAL_DATA_PATH}")
    meta_acc, csi_acc = [], []
    for f in files:
        try:
            ds = WifiCSIDataset(logger=logger, file_list=[f], window_size=_model_bundle["window_size"], stride=_model_bundle["stride"])
            Xm, Xc, _, _, _ = ds.load_csv_as_numpy(f)
            meta_acc.append(Xm)
            csi_acc.append(Xc)
        except Exception:
            logger.exception("skipping file during scaler fit: %s", f)
    meta_all = np.vstack(meta_acc)
    csi_all = np.vstack(csi_acc)
    scaler_meta.fit(meta_all)
    scaler_csi.fit(csi_all)
    _model_bundle["scaler_meta"] = scaler_meta
    _model_bundle["scaler_csi"] = scaler_csi
    logger.info("Fitted scalers from training files (fallback)")

def make_windows_and_tensors(X_meta, X_csi):
    W = _model_bundle["window_size"]
    S = _model_bundle["stride"]
    samples = []
    T = len(X_meta)
    for start in range(0, T - W + 1, S):
        m_seq = X_meta[start:start+W]
        c_seq = X_csi[start:start+W]
        samples.append((
            torch.tensor(m_seq, dtype=torch.float32).unsqueeze(0),
            torch.tensor(c_seq, dtype=torch.float32).unsqueeze(0)
        ))
    return samples

def predict_on_windows(windows, topk=3):
    model = _model_bundle["model"]
    if model is None:
        raise RuntimeError("No model loaded")
    soft = torch.nn.Softmax(dim=1)
    results = []
    for meta_t, csi_t in windows:
        meta_t = meta_t.to(device)
        csi_t = csi_t.to(device)
        with torch.no_grad():
            try:
                out = model(csi_t, meta_t)
            except Exception:
                out = model(meta_t, csi_t)
            probs = soft(out).cpu().numpy()[0]
            top_idx = probs.argsort()[-topk:][::-1]
            top = [{"label": int(i), "prob": float(probs[i])} for i in top_idx]
            results.append({"top": top})
    return results

# ---- endpoints ----
@app.route("/models", methods=["GET"])
def models_info():
    return jsonify({
        "loaded": _model_bundle["model"] is not None,
        "model_name": _model_bundle["model_name"],
        "params": _model_bundle["params"] or {}
    })

@app.route("/reload_model", methods=["POST"])
def reload_model_endpoint():
    req = request.get_json() or {}
    path = req.get("payload_path", DEFAULT_PAYLOAD_PATH)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": f"payload not found: {path}"}), 400
    try:
        load_model(path)
        return jsonify({"ok": True, "msg": f"loaded {path}"})
    except Exception as e:
        logger.exception("Reload model failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if _model_bundle["model"] is None:
        return jsonify({"error": "no model loaded; call /reload_model or set MODEL_PAYLOAD_PATH"}), 400
    if "file" not in request.files:
        return jsonify({"error": "no file provided; attach file field named 'file'"}), 400
    f = request.files["file"]
    fname = secure_filename(f.filename or "upload.csv")
    buf = f.read()
    try:
        X_meta, X_csi, y = parse_single_csv_bytes(buf)
    except Exception as e:
        logger.exception("Failed to parse CSV")
        return jsonify({"error": f"failed to parse csv: {e}"}), 500

    try:
        ensure_scalers()
    except Exception as e:
        logger.exception("Scaler ensure failed")
        return jsonify({"error": f"failed to ensure scalers: {e}"}), 500

    scaler_meta = _model_bundle["scaler_meta"]
    scaler_csi = _model_bundle["scaler_csi"]
    X_meta_scaled = scaler_meta.transform(X_meta)
    X_csi_scaled = scaler_csi.transform(X_csi)

    windows = make_windows_and_tensors(X_meta_scaled, X_csi_scaled)
    if not windows:
        return jsonify({"error": "no windows generated; check window_size vs file length"}), 400

    preds = predict_on_windows(windows, topk=3)
    return jsonify({
        "file": fname,
        "n_windows": len(preds),
        "predictions": preds
    })

# serve web UI at /ui/
@app.route("/ui/<path:filename>")
def ui_static(filename):
    return send_from_directory("web_ui", filename)

if __name__ == "__main__":
    # for local debug only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
