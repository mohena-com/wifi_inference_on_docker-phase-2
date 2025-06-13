
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY sample_test_data/ sample_test_data/

EXPOSE 5000

CMD ["python", "src/flask_real_time_inference.py"]