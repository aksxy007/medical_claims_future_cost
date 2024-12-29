# endpoints/trigger.py
from fastapi import APIRouter, HTTPException
from Models.RunConfig import Config
import httpx
import os
import json
from requests.auth import HTTPBasicAuth
import pprint 
# Initialize the router for the trigger endpoint
router = APIRouter()


# URL for the Airflow REST API to trigger the DAG
AIRFLOW_API_URL = 'http://localhost:8080/api/v1/dags/parent_dag/dagRuns'

# Define the Airflow API Key (if necessary)
AIRFLOW_API_KEY = os.getenv('AIRFLOW_API_KEY')  # You can store this in environment variables
AIRFLOW_USERNAME = 'airflow'
AIRFLOW_PASSWORD = 'airflow'


@router.post("/trigger-pipeline")
async def trigger_pipeline(req: Config):
    """
    Endpoint to trigger the Airflow parent DAG with the provided config.
    """
    conf={"conf":req.dict()}
    print(type(conf))
    print(conf)
    # print("My Config",req.dict())
    try:
        async with httpx.AsyncClient() as client:
            # Make an asynchronous POST request to Airflow to trigger the DAG
            response = await client.post(
                AIRFLOW_API_URL,
                json=conf,  # Pass the config as part of the request body
                auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
            )

            print("Airflow return",response.json())
            # If the response from Airflow is successful
            if response.status_code == 200:
                return {"message": "Pipeline triggered successfully", "dag_run_id": response.json().get("dag_run_id")}
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to trigger pipeline")
    except Exception as e:
        # Handle any other errors
        raise HTTPException(status_code=500, detail=f"Error triggering pipeline: {str(e)}")
    

@router.get("/health")
def healthCheck(req):
    return {"message":"up and running"}
