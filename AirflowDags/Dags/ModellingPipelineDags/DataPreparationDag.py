import sys
import os

# Add the path to the modeling scripts to sys.path
scripts_path = os.path.join('/opt/airflow', 'modelling_pipeline', 'modelling_scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from DataPreparation import DataPreparation  # Replace with the actual module containing your class

# Configuration file path

CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'
BASE_OUTPUT_PATH = '/opt/airflow/modelling_pipeline/modelling_output'

def load_config():
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return config

def prepare_data(**kwargs):
    """Main function to prepare data using the DataPreparation class."""
    # Load config and dataset
    config = load_config()
    output_folder = os.path.join(BASE_OUTPUT_PATH,config['output_folder'], 'data')
    target = config.get('target_column', '')
    prep_folder = os.path.join(BASE_OUTPUT_PATH, config['output_folder'],"Prep")
    os.makedirs(prep_folder,exist_ok=True)
    df = pd.read_csv(os.path.join(output_folder,"training_data.csv"))
    
    # Instantiate and execute data preparation tasks
    prep = DataPreparation(df, config, target)
    prep.handle_missing_values()
    prep.encode_categorical_features()
    prep.standardize_or_normalize()
    prep.save_df(os.path.join(prep_folder,'prepared_data.csv'))
                 
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='data_preparation_dag',
    default_args=default_args,
    description='DAG for preparing data before model training',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['data_preparation'],
) as dag:

    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        provide_context=True,
    )

    task_prepare_data
