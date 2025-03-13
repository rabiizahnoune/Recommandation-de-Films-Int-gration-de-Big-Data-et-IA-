FROM bitnami/spark:3.5.0

WORKDIR /opt/spark_jobs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter requests numpy
COPY . .

