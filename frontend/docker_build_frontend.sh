#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

HOST_PORT=8080
CONTAINER_PORT=80
HEALTH_URL="http://localhost:${HOST_PORT}/gaitid/index.html"
MAX_HEALTH_RETRIES=3        # number of retries for health check
SLEEP_BETWEEN_RETRIES=5     # seconds between retries
CONTAINER_NAME="wifi_inference_frontend"
IMAGE_NAME="wifi_inference_frontend:latest"
DOCKERFILE="Dockerfile.frontend"

# Helper: print to stderr
err() { printf '%s\n' "$*" >&2; }

# Ensure docker is available
if ! command -v docker >/dev/null 2>&1; then
  err "Docker not found in PATH. Install Docker and retry."
  exit 1
fi

echo "🧩 Updating repository (git pull --ff-only)..."
# If you want the deploy to continue even when no git remote is configured, don't fail on git errors:
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git pull --ff-only || echo "Warning: git pull failed or no updates."
else
  echo "Not a git repo, skipping git pull."
fi

echo
echo "🔨 Building Docker image (${IMAGE_NAME}) from ${DOCKERFILE}..."
docker build --pull -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

# If a container with the same name exists, stop and remove it (safe replacement)
if docker ps -a --format '{{.Names}}' | grep -xq "${CONTAINER_NAME}"; then
  echo "🛑 Stopping existing container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" || true
  echo "🧹 Removing existing container ${CONTAINER_NAME}..."
  docker rm "${CONTAINER_NAME}" || true
fi

echo
echo "🚀 Starting new container from image: ${IMAGE_NAME}"
# Bind to all interfaces explicitly (0.0.0.0) so LAN access works
# docker run -e PYTHONUNBUFFERED=1 \
#  -d --restart unless-stopped \
#  -p "0.0.0.0:${HOST_PORT}:${CONTAINER_PORT}" \
#  --name "${CONTAINER_NAME}" \
#  "${IMAGE_NAME}"

docker run -d --restart unless-stopped \
  -p 0.0.0.0:${HOST_PORT}:${CONTAINER_PORT} \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"



echo "⏳ Waiting a moment for the container to initialize..."
sleep 2

echo "Info:  docker logs -f  wifi_inference_frontend"
echo 

# Health check with retries
echo "🔎 Checking health URL: ${HEALTH_URL}"
i=0
until [ "${i}" -ge "${MAX_HEALTH_RETRIES}" ]; do
  # use curl if available, else use docker exec to curl inside container if installed
  if command -v curl >/dev/null 2>&1; then
    if curl --fail --silent --show-error --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "✅ Health check passed (URL reachable)."
      exit 0
    fi
  else
    # fallback: try HTTP from within container (if curl is installed in image)
    if docker exec "${CONTAINER_NAME}" sh -c "command -v curl >/dev/null 2>&1 && curl --fail --silent --show-error --max-time 5 ${HEALTH_URL} >/dev/null 2>&1"; then
      echo "✅ Health check passed (via container)."
      exit 0
    fi
  fi

  i=$((i + 1))
  echo "Attempt ${i}/${MAX_HEALTH_RETRIES} failed; sleeping ${SLEEP_BETWEEN_RETRIES}s..."
  sleep "${SLEEP_BETWEEN_RETRIES}"
done

err "❌ Health check failed after ${MAX_HEALTH_RETRIES} attempts. Check container logs:"
err "  docker logs -f ${CONTAINER_NAME}"
exit 2

