
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/uploads
# Copy source code and other necessary folders
COPY src/ /app/
COPY config/ /app/config/
COPY models/ /app/models/
 
# Ensure static directory exists and copy static files
RUN mkdir -p /app/static
COPY src/index.html /app/static/index.html
COPY src/script.js /app/static/script.js

# Expose the port your Flask app runs on
EXPOSE 5002

# Set environment variable for unbuffered output (for Docker logs)
ENV PYTHONUNBUFFERED=1

CMD ["python", "flask_real_time_inference.py"]