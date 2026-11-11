"""Reproducible, feature-only Wisconsin clustering workflow."""
from datetime import datetime, timedelta, timezone

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.clustering_pipeline import (
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_predict,
)

with DAG(
    dag_id="MLOPS_Airflow_Lab1",
    description="Wisconsin feature clustering with elbow-selected K-Means",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    default_args={
        "owner": "Adit Vakil",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["lab1", "clustering", "wisconsin"],
) as dag:
    load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=data_preprocessing,
        op_args=[load.output],
    )
    train = PythonOperator(
        task_id="select_and_save_model",
        python_callable=build_save_model,
        op_args=[preprocess.output],
    )
    predict = PythonOperator(
        task_id="reload_and_predict",
        python_callable=load_model_predict,
        op_args=[train.output],
    )

    load >> preprocess >> train >> predict
