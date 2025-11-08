
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/uploads
# Copy source code and other necessary folders
COPY src/confg_reader.py /app/confg_reader.py
COPY src/CSI_ID_MLOPS_advanced_v1.3.py /app/CSI_ID_MLOPS_advanced_v1.3.py
COPY src/CSI_Model_Eval_helper.py /app/CSI_Model_Eval_helper.py
COPY src/DL_CSILSTMNet.py /app/DL_CSILSTMNet.py
COPY src/DL_DenseNet1D.py /app/DL_DenseNet1D.py
COPY src/DL_EfficientNet1DLSTM.py /app/DL_EfficientNet1DLSTM.py
COPY src/DL_MobileNetV3.py /app/DL_MobileNetV3.py
COPY src/DS_WifiCSIDataset.py /app/DS_WifiCSIDataset.py
COPY src/flask_real_time_inference.py /app/lask_real_time_inference.py

COPY config/gait_id_config.properties /app/config/
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



 