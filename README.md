# CSI Inference — Deployment (Flask + Gunicorn + Docker)

## Quick local dev
1. Create virtualenv:
   python -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Put your model payload (best_payload.pt) under ./checkpoints or set MODEL_PAYLOAD_PATH env var.
4. Run:
   python app.py
5. Open UI:
   http://localhost:8000/ui/index.html

## Docker (CPU)
docker compose up --build

## Docker (GPU)
DOCKERFILE=Dockerfile.gpu docker compose up --build
(note: ensure host has nvidia runtime & image base matches CUDA driver)

## Environment vars
- MODEL_PAYLOAD_PATH: path inside container to saved payload (default /app/checkpoints/best_payload.pt)
- LOCAL_DATA_PATH: path to training data used to fit scalers if not saved in payload
- WINDOW_SIZE, STRIDE, NUM_CLASSES: override defaults
