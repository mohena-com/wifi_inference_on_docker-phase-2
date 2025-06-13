docker run -d -p 5000:5000 --name har_inference_container har_inference:latest

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"csi_data": [[[[0.1],[0.2],...]]]}'


docker stop har_inference_container
docker rm har_inference_container