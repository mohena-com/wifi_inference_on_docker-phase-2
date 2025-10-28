import os
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import psutil
from mlflow.models.signature import infer_signature
import itertools
from sklearn.preprocessing import label_binarize

# Import your custom classes
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM
# EfficientNet1D would be imported similarly

# --- Logging setup (use your CSI_ID.py pattern) ---
cr = ConfigReader("csi_id_config.properties")




def setup_logging(log_file_path):

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

def create_model_instance(model_class, model_name, batch, device):
    print(f"Creating model instance for {model_class} / {model_name} ")
        # Model instantiation according to constructor
    model = None
    if model_name == "CSILSTMNet":
        model = model_class(
            csi_input_size=batch["csi_seq"].shape[2],
            meta_input_size=batch["metadata_seq"].shape[2],
            window_size=batch["metadata_seq"].shape[1],
            num_classes=31
        ) 
    elif model_name in ["DenseNet1D", "MobileNetV3_1D_LSTM"]:
        model = model_class(
            csi_channels=batch["csi_seq"].shape[2],
            meta_feature_dim=batch["metadata_seq"].shape[2],
            num_classes=31
        ) 
    elif model_name == "EfficientNet1DLSTM":
        model = model_class(
            csi_input_channels=batch["csi_seq"].shape[2],
            meta_input_size=batch["metadata_seq"].shape[-1],
            num_classes=31
        ) 
    else:
        raise ValueError("Unknown model")
    return model.to(device)

def train_and_evaluate(model_class, model_name, train_dataset, test_dataset, device, params, checkpoint_dir, logger):
    logger.info(f"START T-N-E {model_name} USING LEARNING RATE {params['lr']}")
    """Train and evaluate for one set of params, return metrics, best ckpt, and full history."""
    criterion = nn.CrossEntropyLoss()

    num_epochs = params["epochs"]
    best_val_acc, best_epoch = 0, 0
    best_model_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []      

    b_size = int(params['batch_size'])

    # DataLoader options tuned for typical desktop/laptop (adjust num_workers)
    # device-specific flags
    non_blocking_flag = True if device.type == "cuda" else False
    pin_mem = True if device.type != "cpu" else False

    # DataLoader options tuned for typical desktop/laptop (adjust num_workers)
    num_workers = 0 if device.type == "mps" or device.type == "cpu" else min(4, max(1, (os.cpu_count() or 4)//2))

    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers>0))
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers>0))

    batch = next(iter(train_loader))
    model = create_model_instance(model_class, model_name, batch, device)
 
    total_params = sum(p.numel() for p in model.parameters())

    if params["optimizer"] == "adam" or params["optimizer"] == "adamw":
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9, weight_decay=params["weight_decay"])
    logger.info(f"Model parameter count: {total_params}")
    mlflow.log_params(params)    
    mlflow.log_param("parameter_count", total_params)
   
   # logger.info(f"{model_name} USING LEARNING RATE {params['lr']}")
    csi_seq = None
    meta_seq = None
    outputs = None
    for epoch in range(num_epochs):
        start_time = time.time()
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_true, train_pred = [], []
        for batch in train_loader:
            csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
            meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
            labels = batch["label"].squeeze().to(device, non_blocking=non_blocking_flag).long()

            # guard against NaN/Inf values in inputs/labels
            if torch.isnan(csi_seq).any() or torch.isinf(csi_seq).any():
                csi_seq = torch.nan_to_num(csi_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(meta_seq).any() or torch.isinf(meta_seq).any():
                meta_seq = torch.nan_to_num(meta_seq, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                labels = torch.nan_to_num(labels, nan=0).long()

            # simple per-batch normalization for csi_seq (avoid division by zero)
            try:
                mean = csi_seq.mean(dim=(0, 1), keepdim=True)
                std = csi_seq.std(dim=(0, 1), keepdim=True) + 1e-8
                csi_seq = (csi_seq - mean) / std
            except Exception:
                # fallback: skip normalization if shape unexpected
                pass


            optimizer.zero_grad()
            try:
                outputs = model(csi_seq, meta_seq)
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Detected NaN/Inf loss on training batch; skipping this batch.")
                    continue
                loss.backward()
            except Exception as e:
                logger.exception(f"Exception during forward/backward: {e}. Skipping batch.")
                continue

            # gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_true.extend(labels.cpu().numpy().tolist())
            train_pred.extend(preds.cpu().numpy().tolist())

        
        # compute epoch train metrics
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        logger.debug(f"running loss: {running_loss}, len train_loader: {len(train_loader)}, train_loss: {train_loss}, train_acc: {train_acc}")

        # log training metrics
        train_precision = precision_score(train_true, train_pred, average="weighted", zero_division=0) if train_true else 0.0
        train_recall = recall_score(train_true, train_pred, average="weighted", zero_division=0) if train_true else 0.0
        train_f1 = f1_score(train_true, train_pred, average="weighted", zero_division=0) if train_true else 0.0

        mlflow.log_metrics({
            "Training Loss Curve": train_loss,
            "Training Accuracy Curve": train_acc,
            "Training Precision Curve": train_precision,
            "Training Recall Curve": train_recall,
            "Training F1 Score": train_f1
        }, step=epoch)

        # Validation phase
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

        val_loss = running_loss / max(1, len(test_loader))
        val_acc = correct / max(1, total)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # log validation metrics
        val_precision = precision_score(val_true, val_pred, average="weighted", zero_division=0) if val_true else 0.0
        val_recall = recall_score(val_true, val_pred, average="weighted", zero_division=0) if val_true else 0.0
        val_f1 = f1_score(val_true, val_pred, average="weighted", zero_division=0) if val_true else 0.0

        mlflow.log_metrics({
            "Validation Loss Curve": val_loss,
            "Validation Accuracy Curve": val_acc,
            "Validation Precision Curve": val_precision,
            "Validation Recall Curve": val_recall,
            "Validation F1 Score": val_f1
        }, step=epoch)

        # ROC-AUC (robust handling)
        try:
            val_true_np = np.array(val_true)
            val_prob_np = np.array(val_prob)
            unique_classes = np.unique(val_true_np)
            if len(unique_classes) and val_prob_np.size:
                if len(unique_classes) == val_prob_np.shape[1]:
                    val_auc = roc_auc_score(val_true_np, val_prob_np, multi_class='ovr', average='weighted')
                else:
                    y_true_bin = label_binarize(val_true_np, classes=unique_classes)
                    val_prob_subset = val_prob_np[:, unique_classes]
                    val_auc = roc_auc_score(y_true_bin, val_prob_subset, multi_class='ovr', average='weighted')
                mlflow.log_metric('Validation AUC Curve', float(val_auc), step=epoch)
        except Exception as e:
            logger.debug(f"ROC-AUC computation skipped/failed: {e}")

        # log lr
        for param_group in optimizer.param_groups:
            mlflow.log_metric("Learning Rate Over Epochs", param_group.get("lr", 0.0), step=epoch)

        end_time = time.time()
        hw_metrics = {
            "Epoch Duration in Seconds": end_time - start_time,
            "CPU Utilization Percent": psutil.cpu_percent(),
            "Memory Used GB": psutil.virtual_memory().used / (1024 ** 3),
        }
        mlflow.log_metrics(hw_metrics, step=epoch)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Model {model_name}, "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, "
            f"Epoch Duration: {end_time-start_time:.2f} sec"
        )

        # Save best model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            do_signature_logging(model, model_name, csi_seq, meta_seq, params, logger, device)

    learning_rate = float(params.get('lr', 0.0))
    logger.fatal(f"Training complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
    if best_model_state is not None:
        best_model_fname = f"{checkpoint_dir}/inner_best_model_{params['model_name']}_epoch{best_epoch}.pt"
        torch.save(best_model_state, best_model_fname)
        logger.fatal(f"SAVED BEST MODEL {best_model_fname} USING LEARNING RATE {learning_rate}")
        mlflow.log_artifact(best_model_fname)

    return train_losses, val_losses, train_accs, val_accs, best_model_state, best_val_acc, best_epoch, learning_rate, model, test_loader
#

def plot_stats(history, save_path, logger):
    epochs = range(1, len(history['accuracy']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    logger.info(f"Saved stats plot to {save_path}")
    plt.close()

import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cm_cr(cm, cr_report, model_name, params, plot_path):
    # --- Robust Confusion Matrix Plot ---
    fig_width = max(12, 0.5 * cm.shape[0])
    fig_height = max(9, 0.5 * cm.shape[0])
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        annot_kws={"size": 6 if cm.shape[0] > 20 else 10}
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    tick_font_size = 6 if cm.shape[0] > 20 else 9
    plt.xticks(np.arange(cm.shape[1]) + 0.5, np.arange(1, cm.shape[1]+1), rotation=90, fontsize=tick_font_size)
    plt.yticks(np.arange(cm.shape[0]) + 0.5, np.arange(1, cm.shape[0]+1), rotation=0, fontsize=tick_font_size)
    plt.tight_layout()
    cm_path = f"{plot_path}/{make_run_name(params)}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()

    # --- Robust Classification Report Plot ---
    # Always treat as a string for wide compatibility, autoscale height for #lines
    report_str = str(cr_report)
    n_lines = report_str.count('\n') + 1
    plt.figure(figsize=(12, min(max(n_lines * 0.4, 6), 48)))
    plt.text(0, 1, report_str, fontsize=10, family='monospace', verticalalignment='top')
    plt.axis('off')
    plt.title('Classification Report')
    cr_path = f"{plot_path}/{make_run_name(params)}_classification_report.png"
    plt.savefig(cr_path, bbox_inches='tight')
    plt.close()

    # Log artifacts to MLflow
    import mlflow
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(cr_path)

def format_weight_decay_for_name(wd) -> str:
    """
    Return a compact string for wd suitable for filenames.
    Examples:
      0       -> "0"
      1e-05   -> "1e-05"
      0.001   -> "1e-03"
      0.01    -> "0p01"
      0.1     -> "0p1"
    """
    wd = float(wd)
    if wd == 0.0:
        return "0"
    if wd < 1e-2:
        # scientific format for very small values (remove '+' in exponent)
        s = f"{wd:.0e}"
        return s.replace("+", "")
    # readable decimal with '.' -> 'p' for filenames
    return str(wd).replace(".", "p")

def format_lr_for_name(lr) -> str:
    """
    Format learning rate for filenames:
      0       -> "0"
      0.0001  -> "1e-04"
      5e-05   -> "5e-05"
      0.01    -> "0p01"
      0.1     -> "0p1"
    """
    lr = float(lr)
    if lr == 0.0:
        return "0"
    if lr < 1e-2:
        s = f"{lr:.0e}"
        return s.replace("+", "")
    return str(lr).replace(".", "p")



def make_run_name(params):
    """
    Create a short, readable, and unique run/file name based on model and key hyperparameters.
    Example: 'DenseNet1D_lr1e-3_bs64_adam_wd1e-4_ep20'
    """
    lr = params.get("lr", 0)
    bs = params.get("batch_size", 0)
    opt = params.get("optimizer", "opt")
    wd = params.get("weight_decay", 0)
    epochs = params.get("epochs", 0)
    model_name = params.get("model_name", 'Invalid Model')

    # Sanitize numbers for filenames (avoid scientific notation & dots)
    lr_str = format_lr_for_name(lr)
    wd_str = format_weight_decay_for_name(wd)

    run_name = f"{model_name}_lr{lr_str}_bs{bs}_{opt}_wd{wd_str}_ep{epochs}"
    return run_name

def run_mlop_pipeline(cr, exp_path, plot_path, log_path, checkpoint_dir, log_filename, device):
    logger = setup_logging(log_filename)
    print(f"logger {logger}")
    # --- Dataset loading (as in DS_WifiCSIDataset.py) ---
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
    logger.critical(f"Dataset loaded with {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    best_overall_model = (None, None)
   # device = torch.device("mps" if torch.backends.mps.is_available() else
    #                     "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")  
    # --- Model Definitions & hyperparameter grid ---

    from model_definitions import var_model_definitions

    model_defs = var_model_definitions  
    best_model_no = 0
    learning_rate = None
    # --- MLflow experiment ---
    mlflow.set_experiment(cr.get("experiment_name"))
    best_overall = {"val_acc": -1}
    stats_summary = {}
    non_blocking_flag = True if device.type == "cuda" else False
    print(f"model_defs {model_defs}")
    for model_name, (model_class, param_grid) in model_defs.items():
        param_keys, param_vals = zip(*param_grid.items())
        logger.info(" ")
        logger.info("N")
        logger.info("E")
        logger.info("W")
        logger.info(f"STARTING TRAINING FLOW : {model_name}:{param_keys}:{param_vals}")
        with mlflow.start_run(run_name=f"{model_name}_main") as parent_run:
            for combo in itertools.product(*param_vals):
                params = dict(zip(param_keys, combo))
                params['model_name'] = model_name
            
                logger.info(f"START RUN MLFLOW : {make_run_name(params)}")                
                    
                with mlflow.start_run(run_name=f"{make_run_name(params)}", nested=True) as child_run:
                    train_losses, val_losses, train_accs, val_accs, best_model_state, best_val_acc, best_epoch, learning_rate, model, test_loader = train_and_evaluate(
                        model_class, model_name, train_dataset, test_dataset, device, params, checkpoint_dir, logger)

                history = {
                    'accuracy': train_accs,
                    'val_accuracy': val_accs,
                    'loss': train_losses,
                    'val_loss': val_losses
                }
                stats_summary[(model_name, str(params))] = {
                    "val_acc": best_val_acc,
                    "epoch": best_epoch,
                    "history": history
                }
                plot_stats(history, save_path=f"{plot_path}/{make_run_name(params)}_stats.png", logger=logger)

                # Save confusion matrix, classification report on test set
                if best_model_state is None:
                    logger.warning("No best model state found, skipping test evaluation.")
                    continue

                model.load_state_dict(best_model_state)
                all_preds, all_labels = [], []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        csi_seq = batch["csi_seq"].to(device, non_blocking=non_blocking_flag)
                        meta_seq = batch["metadata_seq"].to(device, non_blocking=non_blocking_flag)
                        labels = batch["label"].squeeze().to(device, non_blocking=True)
                        outputs = model(csi_seq, meta_seq)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        labels_np = labels.cpu().numpy()
                        all_preds.extend(preds.tolist())
                        all_labels.extend(labels_np.tolist())
                cm = confusion_matrix(all_labels, all_preds)
                cr_report = classification_report(all_labels, all_preds, zero_division=0)
                np.save(f"{plot_path}/{make_run_name(params)}_cm.npy", cm)
                mlflow.log_artifact(f"{plot_path}/{make_run_name(params)}_cm.npy")
                plot_cm_cr(cm, cr_report, model_name, params,  plot_path)

                logger.info(f"Confusion matrix:\n{cm}")
                logger.info(f"Classification report:\n{cr_report}")
                # Save model checkpoint for best overall if needed
                if best_val_acc > best_overall["val_acc"]:
                    best_model_no += 1
                    best_overall = {
                        "model": model_name,
                        "params": params,
                        "val_acc": best_val_acc,
                        "epoch": best_epoch,
                        "lr": learning_rate
                    }
                    # store state and metadata to save once later
                    best_overall_model_state = best_model_state
                    best_overall_model_meta = {
                        "model_name": model_name,
                        "params": params,
                        "model_class": model_class
                    }
                    logger.info(f"New best overall model (deferred save): {model_name} val_acc={best_val_acc:.4f} epoch={best_epoch}")

                mlflow.log_metric("Top Validation Accuracy", best_val_acc)

    # single final save/log of best overall model (if any)
    if best_overall.get("val_acc", -1) >= 0 and best_overall_model_state is not None:
        final_name = make_run_name(best_overall["params"])
        final_path = f"{checkpoint_dir}/best_overall_model_final_{final_name}_valacc{best_overall['val_acc']:.4f}.pt"
        # save state + metadata so you can reconstruct model later
        payload = {
            "state_dict": best_overall_model_state,
            "model_name": best_overall_model_meta["model_name"],
            "params": best_overall_model_meta["params"]
        }
        torch.save(payload, final_path)
        mlflow.log_artifact(final_path)
        logger.fatal(f"SAVED & LOGGED SINGLE BEST OVERALL MODEL: {final_path}")

        # attempt to log a pytorch model with signature for real-world inference
        try:
            # create a small sample batch and instantiate model, then delegate signature logging
            sample_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
            sample_batch = next(iter(sample_loader))
            bc = best_overall_model_meta["model_class"]
            sample_model = create_model_instance(bc, best_overall_model_meta["model_name"], sample_batch, device)
            sample_model.load_state_dict(payload["state_dict"])
            sample_model.eval()

            # use centralized helper to log model + signature
            do_signature_logging(sample_model,
                                 best_overall_model_meta["model_name"],
                                 sample_batch["csi_seq"],
                                 sample_batch["metadata_seq"],
                                 payload["params"],
                                 logger,
                                 device)
            logger.info("Logged PyTorch model + signature to MLflow for production use.")
        except Exception as e:
            logger.exception(f"Failed to log PyTorch model with signature: {e}")
    logger.info(f"best_overall : {best_overall}")

def do_signature_logging(model, model_name, csi_seq, meta_seq, params, logger, device):
    """Log model signature to MLflow with proper resource cleanup."""
    import tempfile
    import os
    import json
    import gc

    logger.debug("0. Starting signature logging")
    model.eval()
    non_blocking = True if device.type == "cuda" else False
    try:
        with torch.no_grad():
            # ensure inputs are on device
            inp_csi = csi_seq.to(device, non_blocking=non_blocking)
            inp_meta = meta_seq.to(device, non_blocking=non_blocking)
            example_output = model(inp_csi, inp_meta)

        # Convert tensors to numpy on CPU
        csi_np = inp_csi.cpu().numpy()
        meta_np = inp_meta.cpu().numpy()
        op_np = example_output.cpu().numpy()

        # Build an input example for signature inference (try concat, fallback to dict)
        try:
            input_example = np.concatenate([csi_np, meta_np], axis=-1)
            input_is_array = True
        except Exception:
            input_example = {"csi_seq": csi_np, "metadata_seq": meta_np}
            input_is_array = False

        # Try to infer signature but tolerate failures
        signature = None
        try:
            signature = infer_signature(input_example, out_np)
        except Exception as e:
            logger.debug(f"Signature inference failed: {e}")
            signature = None

        # Save minimal artifacts in a temp dir and upload via mlflow.log_artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # save model state dict (smaller than logging full model environment)
            state_path = os.path.join(tmp_dir, "state_dict.pth")
            torch.save(model.state_dict(), state_path)

            # save examples
            if input_is_array:
                np.save(os.path.join(tmp_dir, "input_example.npy"), input_example)
            else:
                np.save(os.path.join(tmp_dir, "csi_example.npy"), input_example["csi_seq"])
                np.save(os.path.join(tmp_dir, "meta_example.npy"), input_example["metadata_seq"])
            np.save(os.path.join(tmp_dir, "output_example.npy"), op_np)

            # save metadata + signature (if available)
            meta = {
                "model_name": model_name,
                "params": params,
                "signature": signature.to_dict() if signature is not None else None
            }
            with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            # Upload the directory as a single artifact tree (less fd churn than log_model)
            mlflow.log_artifacts(tmp_dir, artifact_path=f"best_model_{model_name}")
            logger.debug(f"Logged artifacts for model {model_name} to MLflow (artifact_path=best_model_{model_name})")

    except Exception as e:
        logger.exception(f"Failed to log model signature/artifacts: {e}")
    finally:
        # device-aware cleanup
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    logger.debug("4. Signature logging completed")

def do_signature_logging1(model, model_name, csi_seq, meta_seq, params, logger, device):

    import numpy as np
    import tempfile
    import shutil
    import os
    
    logger.debug("0. do_signature_logging")
    # Prepare input example matching your model's expected input

    example_output = None
    model.eval()
    logger.debug("1. do_signature_logging")
    with torch.no_grad():
        # ensure inputs are on device before forward
        inp_csi = csi_seq.to(next(model.parameters()).device) if not csi_seq.device == next(model.parameters()).device else csi_seq
        inp_meta = meta_seq.to(next(model.parameters()).device) if not meta_seq.device == next(model.parameters()).device else meta_seq
        example_output = model(inp_csi, inp_meta)

    # Convert tensors to numpy
    csi_np = csi_seq.cpu().numpy()
    meta_np = meta_seq.cpu().numpy()
    op_np = example_output.cpu().numpy()

    # Concatenate along the last axis (feature dimension)
    combined_input = np.concatenate([csi_np, meta_np], axis=-1)
    logger.debug("2. do_signature_logging")

    # Infer signature and log model
    signature = infer_signature(combined_input, op_np)
    logger.debug("3. do_signature_logging")

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=f"best_model_{model_name}.{params['model_name']}",   
        signature=signature
    )
    logger.debug("4. do_signature_logging")

    logger.debug(f"5. do_signature_logging Logged model with signature to MLflow for {model_name} with params {params}")
def get_device():
    """
    Return a torch.device choosing MPS (Apple), then CUDA, then CPU.
    Also set a few backend flags appropriate for the chosen device.
    """
    # prefer MPS on Apple silicon
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            # improve matmul precision on MPS (PyTorch 1.12+)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            return device
    except Exception:
        pass

    if torch.cuda.is_available():
        # CUDA path
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

def set_system_resources():
    import os
    # limit OpenMP/BLAS threads to avoid many shared-memory fds
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import torch
    # limit torch intra/inter op threads
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # try to raise soft fd limit (Unix/macOS)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(4096, soft)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        # ignore if not permitted
        pass

if __name__ == "__main__":
    set_system_resources()
    now = time.strftime("%Y%m%d_%H%M%S")
    base_dir = cr.get("local_data_path")
    gait_filenme = cr.get("file_name_for_gait")
    filelist = glob.glob(os.path.join(base_dir, '**', gait_filenme), recursive=True)

    exp_path = f"{cr.get('output_path')}/{cr.get('experiment_name')}_{now}"
    os.makedirs(exp_path, exist_ok=True)

    plot_path = f"{exp_path}/plots"
    os.makedirs(plot_path, exist_ok=True)
    print(f"Plot path: {plot_path}")

    log_path = f"{exp_path}/logs"
    os.makedirs(log_path, exist_ok=True)
    print(f"Log path: {log_path}")

    checkpoint_dir = f"{exp_path}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint path: {checkpoint_dir}") 

    log_filename = f"{log_path}/{cr.get('experiment_name')}_run_{now}.log"
    print(f"Log file: {log_filename}")

    # select device using helper
    device = get_device()
    print(f"Using device: {device}")

    print("Starting MLOps pipeline...")
    run_mlop_pipeline(cr, exp_path, plot_path, log_path, checkpoint_dir, log_filename, device)
    print("MLOps pipeline completed.")


# for additional metrices - 
# https://www.perplexity.ai/search/apply-pca-on-a-dataset-for-dim-XHDCWLULS3S6upRT9nu0ig#54