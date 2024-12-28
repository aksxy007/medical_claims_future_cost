from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import json
import logging
# Decision function for branching
def decide_step(step_name, config, **kwargs):
    config=eval(config)
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
    start_date=datetime(2024, 12, 26),
    catchup=False,
    tags=['parent_pipeline'],
) as dag:

    # Assume the config is passed dynamically by Backend 2
    print('{{dag_run.conf}}')
    config = '{{ dag_run.conf.get("config")}}' # Accessing the config passed by Backend 
    print("config",config)
    

    logging.info(f"Config being passed: {config}")
    # Step 1: Trigger Data Acquisition DAG
    trigger_data_acquisition = TriggerDagRunOperator(
        task_id='trigger_data_acquisition',
        trigger_dag_id='data_acquisition_dag',
        wait_for_completion=True,
        conf={"config": config}
        # conf=config,  # Pass the user-specific config
    )

    # Step 2: Check and trigger Data Preparation DAG
    branch_data_preparation = BranchPythonOperator(
        task_id='branch_data_preparation',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'data_preparation', 'config': config},  # Pass config dynamically
    )
    trigger_data_preparation = TriggerDagRunOperator(
        task_id='trigger_data_preparation',
        trigger_dag_id='data_preparation_dag',
        wait_for_completion=True,
        conf={"config": config}
    )
    skip_data_preparation = DummyOperator(task_id='skip_data_preparation')

    # Step 3: Check and trigger Feature Exploration DAG
    branch_feature_exploration = BranchPythonOperator(
        task_id='branch_feature_exploration',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'feature_exploration', 'config': config},  # Pass config dynamically
    )
    trigger_feature_exploration = TriggerDagRunOperator(
        task_id='trigger_feature_exploration',
        trigger_dag_id='feature_exploration_dag',
        wait_for_completion=True,
        conf={"config": config}
    )
    skip_feature_exploration = DummyOperator(task_id='skip_feature_exploration')

    # Step 4: Check and trigger Model Building DAG
    branch_model_building = BranchPythonOperator(
        task_id='branch_model_building',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'build', 'config': config},  # Pass config dynamically
    )
    trigger_model_building = TriggerDagRunOperator(
        task_id='trigger_build',
        trigger_dag_id='build_dag',
        wait_for_completion=True,
        conf={"config": config}
    )
    skip_model_building = DummyOperator(task_id='skip_build')

    # Step 5: Check and trigger Scoring DAG
    branch_scoring = BranchPythonOperator(
        task_id='branch_score',
        python_callable=decide_step,
        trigger_rule='all_done',
        op_kwargs={'step_name': 'build', 'config': config},
        
    )
    trigger_scoring = TriggerDagRunOperator(
        task_id='trigger_score',
        trigger_dag_id='scoring_dag',
        wait_for_completion=True,
        conf={"config": config}
        
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
