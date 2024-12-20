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
import pandas as pd
from FeatureExploration import FeatureExploration

# Path to the configuration file
CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'
BASE_OUTPUT_PATH = '/opt/airflow/modelling_pipeline/modelling_output'
def load_config():
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return config


def perform_feature_exploration(**kwargs):
    """Perform feature exploration and save results."""
    # Load configuration
    config = load_config()
    output_folder = os.path.join(BASE_OUTPUT_PATH,config.get('output_folder'))
    prep_folder = os.path.join(output_folder,"Prep")
    
    # Load processed data from Data Preparation step
    prepared_data_path = os.path.join(prep_folder, 'prepared_data.csv')
    print(f"Loading prepared_data from {prepared_data_path}")
    df = pd.read_csv(prepared_data_path)
    
    # Initialize Feature Exploration
    target = config.get('target_column', 'target')  # Define your target column here
    print("target from config",target)
    feature_exploration = FeatureExploration(df, config,BASE_OUTPUT_PATH)
    
    # Compute feature importance and select features
    feature_exploration.compute_feature_importance()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='feature_exploration_dag',
    default_args=default_args,
    description='DAG for feature exploration and importance computation',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['feature_exploration'],
) as dag:

    # Step 1: Perform Feature Exploration
    perform_exploration = PythonOperator(
        task_id='perform_feature_exploration',
        python_callable=perform_feature_exploration,
    )
