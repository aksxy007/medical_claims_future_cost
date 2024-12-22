from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import json

# Load configuration file
CONFIG_PATH = '/opt/airflow/modelling_pipeline/config/config.json'

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        raise Exception(f"Failed to load config file: {e}")

# Decision function for branching
def decide_step(step_name, **kwargs):
    config = load_config()
    step_config = config.get(step_name, {})
    enabled = step_config.get("enabled", "N")
    print(f"Step {step_name} is {'enabled' if enabled == 'Y' else 'disabled'} in the configuration.")
    return f'trigger_{step_name}' if enabled == "Y" else f'skip_{step_name}'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='parent_dag',
    default_args=default_args,
    description='Parent DAG to orchestrate the entire pipeline',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['parent_pipeline'],
) as dag:

    # Step 1: Trigger Data Acquisition DAG
    trigger_data_acquisition = TriggerDagRunOperator(
        task_id='trigger_data_acquisition',
        trigger_dag_id='data_acquisition_dag',
        wait_for_completion=True,
    )

    # Step 2: Check and trigger Data Preparation DAG
    branch_data_preparation = BranchPythonOperator(
        task_id='branch_data_preparation',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'data_preparation'},
    )
    trigger_data_preparation = TriggerDagRunOperator(
        task_id='trigger_data_preparation',
        trigger_dag_id='data_preparation_dag',
        wait_for_completion=True,
    )
    skip_data_preparation = DummyOperator(task_id='skip_data_preparation')

    # Step 3: Check and trigger Feature Exploration DAG
    branch_feature_exploration = BranchPythonOperator(
        task_id='branch_feature_exploration',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'feature_exploration'},
    )
    trigger_feature_exploration = TriggerDagRunOperator(
        task_id='trigger_feature_exploration',
        trigger_dag_id='feature_exploration_dag',
        wait_for_completion=True,
    )
    skip_feature_exploration = DummyOperator(task_id='skip_feature_exploration')

    # Step 4: Check and trigger Model Building DAG
    branch_model_building = BranchPythonOperator(
        task_id='branch_model_building',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'build'},
    )
    trigger_model_building = TriggerDagRunOperator(
        task_id='trigger_model_building',
        trigger_dag_id='build_dag',
        wait_for_completion=True,
    )
    skip_model_building = DummyOperator(task_id='skip_build')

    # Step 5: Check and trigger Scoring DAG
    branch_scoring = BranchPythonOperator(
        task_id='branch_score',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'score'},
    )
    trigger_scoring = TriggerDagRunOperator(
        task_id='trigger_score',
        trigger_dag_id='scoring_dag',
        wait_for_completion=True,
    )
    skip_scoring = DummyOperator(task_id='skip_score')

    # Final step: End pipeline
    end_pipeline = DummyOperator(task_id='end_pipeline')

    # Define task dependencies
    trigger_data_acquisition >> branch_data_preparation
    branch_data_preparation >> trigger_data_preparation >> branch_feature_exploration
    branch_data_preparation >> skip_data_preparation >> branch_feature_exploration

    branch_feature_exploration >> trigger_feature_exploration >> branch_model_building
    branch_feature_exploration >> skip_feature_exploration >> branch_model_building

    branch_model_building >> trigger_model_building >> branch_scoring
    branch_model_building >> skip_model_building >> branch_scoring

    branch_scoring >> trigger_scoring >> end_pipeline
    branch_scoring >> skip_scoring >> end_pipeline
