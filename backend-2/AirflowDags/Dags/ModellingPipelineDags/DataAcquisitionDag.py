import sys
import os

# Add the path to the modeling scripts to sys.path
scripts_path = os.path.join('/opt/airflow', 'modelling_pipeline', 'modelling_scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import os
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from DataAcquisition import DataAcquisition



# Load configuration file
CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'
BASE_OUTPUT_PATH = '/opt/airflow/modelling_pipeline/modelling_output'
def load_config():
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return config

def acquire_training_data(**kwargs):
    """
    Task to fetch training data and save it to the output folder.
    """
    config = load_config()
    output_folder = os.path.join(BASE_OUTPUT_PATH,config.get('output_folder','output'), 'data')
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize DataAcquisition class
    data_acquisition = DataAcquisition(config)
    
    try:
        # Fetch training data
        training_data = data_acquisition.fetch_training_data()
        training_data_path = os.path.join(output_folder, 'training_data.csv')
        training_data.to_csv(training_data_path, index=False)
        print(f"Training data saved to: {training_data_path}")
    finally:
        # Close connection
        data_acquisition.close_connection()

def acquire_oot_data(**kwargs):
    """
    Task to fetch OOT data if provided and save it to the output folder.
    """
    config = load_config()
    oot_table_name = config.get('oot_table_name', None)
    
    if not oot_table_name:
        print("OOT table name not specified in the config. Skipping OOT data acquisition.")
        return  # Exit the task gracefully if no OOT table name is provided
    
    output_folder = os.path.join(config['input_folder'], 'data')
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize DataAcquisition class
    data_acquisition = DataAcquisition(config)
    
    try:
        # Fetch OOT data
        oot_data = data_acquisition.fetch_oot_data()
        oot_data_path = os.path.join(output_folder, 'oot_data.csv')
        oot_data.to_csv(oot_data_path, index=False)
        print(f"OOT data saved to: {oot_data_path}")
    finally:
        # Close connection
        data_acquisition.close_connection()

# Define the DAG
with DAG(
    'data_acquisition_dag',
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 12, 17),
    },
    schedule_interval=None,
    catchup=False,
) as dag:
    
    # Task 1: Fetch training data
    fetch_training_task = PythonOperator(
        task_id='fetch_training_data',
        python_callable=acquire_training_data,
    )
    
    # Task 2: Fetch OOT data (only if specified)
    fetch_oot_task = PythonOperator(
        task_id='fetch_oot_data',
        python_callable=acquire_oot_data,
    )

    # Task dependencies
    fetch_training_task >> fetch_oot_task
