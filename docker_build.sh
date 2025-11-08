#!/bin/bash
echo "Updating repository..."
git pull
echo 
echo "Building Docker image..." 
docker build -t gait_id_inference:latest -f Dockerfile .
echo 
echo "Restarting Docker container..."
docker stop gait_id_inference_container
echo 
echo "Removing old Docker container..."
docker rm gait_id_inference_container
echo 
echo "Starting new Docker container..."
docker run -e PYTHONUNBUFFERED=1 -d -p 5002:5002 --name gait_id_inference_container gait_id_inference:latest
echo 
sleep 5
echo "🧩 Testing the deployment..."
curl -X GET http://localhost:5002/gaitid/index.html
echo 
echo 
echo "Deployment complete."
