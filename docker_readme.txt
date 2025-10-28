docker stop gait_id_inference_container
docker rm gait_id_inference_container


docker run -d -p 5002:5002 --name gait_id_inference_container gait_id_inference:latest
docker run -e PYTHONUNBUFFERED=1 -d -p 5002:5002 --name gait_id_inference_container gait_id_inference:latest
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"csi_data": [[[[0.1],[0.2],...]]]}'




docker tag gait_id_inference pandeysanjeev/gait_id_inference:latest
docker push pandeysanjeev/gait_id_inference:latest
