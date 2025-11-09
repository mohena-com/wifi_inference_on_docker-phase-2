#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

 
HOST_PORT=8080
CONTAINER_PORT=8080
UPLOAD_WAIT_SECONDS=5
HEALTH_URL="http://localhost:${HOST_PORT}/gaitid/index.html"
MAX_HEALTH_RETRIES=3   # total ~60s if sleep 5s between tries
SLEEP_BETWEEN_RETRIES=5
CONTAINER_NAME="wifi_inference_frontend"
IMAGE_NAME="wifi_inference_frontend:latest"
DOCKERFILE="Dockerfile.frontend"

echo "🧩    Updating repository..."
echo
git pull --ff-only


echo "🔨    Building Docker image (${IMAGE_NAME}) from ${DOCKERFILE}..."
echo
docker build --pull -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

echo "🚀    Starting new container from image: ${IMAGE_NAME}"
echo

docker run -e PYTHONUNBUFFERED=1 \
  -d --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"