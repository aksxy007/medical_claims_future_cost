import sys
import os

# Add the path to the modeling scripts to sys.path
scripts_path = os.path.join('/opt/airflow', 'modelling_pipeline', 'modelling_scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import json
from TrainModels import TrainModels

# Path to the configuration file
CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'
BASE_OUTPUT_PATH = '/opt/airflow/modelling_pipeline/modelling_output'
def load_config():
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return config

def train_models_task(**kwargs):
    """
    Task to execute the TrainModels class.
    This will perform the entire model training and selection process.
    """
    config = load_config()
    trainer = TrainModels(config,base_path=BASE_OUTPUT_PATH)
    trainer.run()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='build_dag',
    default_args=default_args,
    description='DAG for model building and selection',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['build', 'train_models'],
) as dag:

    # Task: Train Models
    train_models = PythonOperator(
        task_id='train_models',
        python_callable=train_models_task,
    )
