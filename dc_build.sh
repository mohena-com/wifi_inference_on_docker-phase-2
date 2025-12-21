export DOCKER_BUILDKIT=1
  
#!/usr/bin/env bash
set -euo pipefail

# Friendly script to build and run docker-compose with better defaults and logs.
rm -rf /System/Volumes/Data/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DEV_1.0/wifi_inference_on_docker-phase-2/buildcache/*
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "🧩 Updating repository..."
cd "$REPO_ROOT"
git pull --ff-only || {
  echo "⚠ git pull failed or no changes."
}

echo
echo "🛑 Stopping existing containers (if any)..."
docker-compose down || echo "No containers to stop or docker-compose down failed."

echo
echo "🧩 Rebuilding images (using BuildKit, pulling latest base images)..."
# Use BuildKit for faster builds / better caching. Pull base images and build without cache to avoid stale base images.
 
# Try to pull latest base images first (helps avoid interactive keychain issues)
echo "-> Pulling base images (best-effort)..."
docker-compose pull --ignore-pull-failures || echo "Note: some images failed to pull (this may be OK)."

# Build all services with no cache to ensure fresh layers. Remove --no-cache if you want faster incremental builds.
docker-compose build --pull --no-cache

echo
echo "🚀 Starting containers (detached)..."
docker-compose up -d --remove-orphans

echo
echo "✅ Services started (give them a moment to become healthy)..."
sleep 2

echo "Frontend at http://webmaster-ai:8080"
echo "API at http://webmaster-ai:5002"
echo "MLFLOW Tracking URI at http://webmaster-ai:5001"

echo

echo "📦 Container status:"
docker-compose ps
echo

echo "📜 Following combined logs (press Ctrl+C to stop)..."
# Follow logs for all services (tail last 200 lines then stream)
docker-compose logs -f --tail=200
