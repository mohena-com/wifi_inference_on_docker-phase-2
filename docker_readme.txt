docker run -d -p 5001:5001 --name har_inference_container har_inference:latest
docker run -e PYTHONUNBUFFERED=1 -d -p 5001:5001 --name har_inference_container har_inference:latest
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"csi_data": [[[[0.1],[0.2],...]]]}'


docker stop har_inference_container
docker rm har_inference_container

docker tag har_inference pandeysanjeev/har_inference:latest
docker push pandeysanjeev/har_inference:latest
