#!/bin/bash

# Define absolute host path (update if your directory path changes)
HOST_PROJECT_PATH="/Users/sanjeev/VNIT/FINAL_PROJECT"

# Define container path
CONTAINER_PROJECT_PATH="/app"

# Run Docker container
docker run --rm \
  -v "$HOST_PROJECT_PATH":"$CONTAINER_PROJECT_PATH" \
  -v "$HOST_PROJECT_PATH/logs":"$CONTAINER_PROJECT_PATH/logs" \
  har_inference \
  "$CONTAINER_PROJECT_PATH" wifi_project har_config.properties False
