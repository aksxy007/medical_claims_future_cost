import sys
import os

# Add the path to the modeling scripts to sys.path
scripts_path = os.path.join('/opt/airflow', 'modelling_pipeline', 'modelling_scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json
import os
from DataAcquisition import DataAcquisition  # Import the DataAcquisition class
from Scoring import Scoring  # Import your Scoring class

# Define the configuration path
CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'
BASE_OUTPUT_PATH = '/opt/airflow/modelling_pipeline/modelling_output'
def load_config():
    """
    Load the configuration from the config.json file.
    """
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return config

def acquire_oot_data(config, **kwargs):
    """
    Use DataAcquisition class to fetch OOT data from Snowflake.
    """
    data_acquisition = DataAcquisition(config)
    oot_data = data_acquisition.fetch_oot_data()  # Fetch OOT data
    oot_data_folder = os.path.join(BASE_OUTPUT_PATH,config.get('output_folder','output'),'Score','data')
    os.makedirs(oot_data_folder,exist_ok=True)
    oot_data.to_csv(os.path.join(oot_data_folder,"oot_data.csv"), index=False)  # Save to CSV
    data_acquisition.close_connection()  # Close Snowflake connection
    return os.path.join(oot_data_folder,"oot_data.csv")  # Path to the acquired OOT data

def score_oot_data(oot_data_path, config,**kwargs):
    """
    Use the Scoring class to score the OOT data.
    """
    import pandas as pd
    oot_data = pd.read_csv(oot_data_path)
    
    # Create Scoring object and score the data
    scoring = Scoring(config,BASE_OUTPUT_PATH)
    evaluation_results = scoring.score(oot_data)

    # Log evaluation results
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
    

    return evaluation_results

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='scoring_dag',
    default_args=default_args,
    description='DAG for Scoring Out-Of-Time Data',
    schedule_interval=None,  # Set as needed
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['scoring_pipeline'],
) as dag:

    # Step 1: Acquire the OOT Data from Snowflake using DataAcquisition
    acquire_oot_data_task = PythonOperator(
        task_id='acquire_oot_data',
        python_callable=acquire_oot_data,
        op_args=[load_config()],  # Pass the config
        provide_context=True,
    )

    # Step 2: Score the OOT Data using the Scoring class
    score_task = PythonOperator(
        task_id='score_oot_data',
        python_callable=score_oot_data,
        op_args=["{{ task_instance.xcom_pull(task_ids='acquire_oot_data') }}", load_config()],
        provide_context=True,
    )

    # Set task dependencies
    acquire_oot_data_task >> score_task
