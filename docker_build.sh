#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

CONTAINER_NAME="gait_id_inference_container"
IMAGE_NAME="gait_id_inference:latest"
DOCKERFILE="Dockerfile.api"
HOST_PORT=5002
CONTAINER_PORT=5002
UPLOAD_WAIT_SECONDS=5
HEALTH_URL="http://localhost:${HOST_PORT}/gaitid/index.html"
MAX_HEALTH_RETRIES=3   # total ~60s if sleep 5s between tries
SLEEP_BETWEEN_RETRIES=5

echo "🧩    Updating repository..."
echo
git pull --ff-only

echo "🔨    Building Docker image (${IMAGE_NAME}) from ${DOCKERFILE}..."
echo
docker build --pull -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

# Stop & remove old container if present (safe)
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
  echo "🛑  Stopping existing container: ${CONTAINER_NAME}"
  echo
  docker stop "${CONTAINER_NAME}" || true

  echo "🧹  Removing existing container: ${CONTAINER_NAME}"
  echo
  docker rm -f "${CONTAINER_NAME}" || true
else
  echo "🧩  No existing container named ${CONTAINER_NAME}"
  echo
fi

echo "🚀    Starting new container from image: ${IMAGE_NAME}"
echo

docker run -e PYTHONUNBUFFERED=1 \
  --env "CONFIG_FILE=config/gait_id_config.properties" \
  -d --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"

# Wait a little and poll health endpoint (with retries)
echo "⏳    Waiting ${UPLOAD_WAIT_SECONDS}s for container to initialize..."
echo
sleep "${UPLOAD_WAIT_SECONDS}"

echo "🔍    Checking health endpoint ${HEALTH_URL}"
echo
success=0
for i in $(seq 1 ${MAX_HEALTH_RETRIES}); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "✅    Health check passed (attempt ${i})."
    echo
    success=1
    break
  else
    echo "⏳    Health check failed (attempt ${i}). Retrying in ${SLEEP_BETWEEN_RETRIES}s..."
    echo
    sleep "${SLEEP_BETWEEN_RETRIES}"
  fi
done

if [[ ${success} -ne 1 ]]; then
  echo "❌    ERROR: Health check did not succeed after $((MAX_HEALTH_RETRIES * SLEEP_BETWEEN_RETRIES))s."
  echo
  echo "❌    Showing last 200 lines of container logs for debugging:"
  echo
  docker logs --tail 200 "${CONTAINER_NAME}" || true
  exit 1
fi

echo "✅    Deployment complete. Container '${CONTAINER_NAME}' running and healthy."
echo
echo "👉    Access the application at: http://localhost:${HOST_PORT}/gaitid/index.html"
echo
echo "👉    View logs with: docker logs -f --tail 100 ${CONTAINER_NAME}"