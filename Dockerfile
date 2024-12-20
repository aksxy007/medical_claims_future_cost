FROM apache/airflow:2.7.1-python3.10

# Switch to the root user to install additional packages
USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (airflowuser) and set permissions
# RUN useradd -m -u 1000 airflowuser \
#     && mkdir -p /opt/airflow/logs \
#     && mkdir -p /opt/airflow/dags \
#     && mkdir -p /opt/airflow/plugins \
#     && chown -R airflowuser:airflowuser /opt/airflow \
#     && chmod -R 777 /opt/airflow/logs

# Switch to the airflowuser
USER airflow

# Set the working directory
WORKDIR /opt/airflow

# Copy requirements.txt to the Docker image
COPY requirements.txt /requirements.txt

# Install Python dependencies as airflowuser
RUN pip install --no-cache-dir -r /requirements.txt

# Set environment variables to prevent pip warnings
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Set the default command to the Airflow entrypoint
ENTRYPOINT ["/entrypoint"]
CMD ["webserver"]
