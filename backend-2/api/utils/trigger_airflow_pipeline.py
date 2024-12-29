import httpx
import os
from requests.auth import HTTPBasicAuth
import logging
import ast
import json
# Configure logging
logging.basicConfig(level=logging.INFO)

# Airflow API Configuration
AIRFLOW_API_URL = 'http://localhost:8080/api/v1/dags/parent_dag/dagRuns'
AIRFLOW_USERNAME = os.getenv('AIRFLOW_USERNAME', 'airflow')
AIRFLOW_PASSWORD = os.getenv('AIRFLOW_PASSWORD', 'airflow')


async def trigger_airflow_pipeline(message: dict):
    """
    Triggers the Airflow pipeline by making a POST request to the Airflow API.

    Args:
        message (dict): The configuration to send to the DAG.

    Returns:
        dict: Response message with the status of the request.
    """
    
    if not isinstance(message, dict):
        message = json.loads(message)
        # raise ValueError("Message must be a dictionary.")
    
    conf = {"conf": message}

    # Log the configuration
    logging.info(f"Triggering Airflow pipeline with config: {conf}")

    try:
        async with httpx.AsyncClient() as client:
            # Make an asynchronous POST request to Airflow to trigger the DAG
            response = await client.post(
                AIRFLOW_API_URL,
                json=conf,  # Pass the config as part of the request body
                auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                timeout=10.0  # Optional: Set a timeout for the request
            )

            # Log the response
            logging.info(f"Airflow response status: {response.status_code}, content: {response.text}")

            # If the response from Airflow is successful
            if response.status_code == 200:
                return {
                    "message": "Pipeline triggered successfully",
                    "dag_run_id": response.json().get("dag_run_id"),
                }
            else:
                logging.error(f"Failed to trigger pipeline: {response.text}")
                return {
                    "message": "Failed to trigger pipeline",
                    "status_code": response.status_code,
                    "detail": response.text,
                }
    except httpx.RequestError as e:
        logging.error(f"HTTP request error: {str(e)}")
        return {"message": "Error triggering pipeline", "detail": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {"message": "Error triggering pipeline", "detail": str(e)}
