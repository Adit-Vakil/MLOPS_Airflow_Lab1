"""Small operator and scheduling exercise from the Lab 1 README."""
from datetime import datetime, timezone

from airflow.sdk import DAG, Param
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator


def describe_run(message):
    print(f"Sandbox parameter: {message}")
    return {"message": message}


with DAG(
    dag_id="MLOPS_Airflow_Lab1_Sandbox",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="0 9 * * *",
    catchup=False,
    is_paused_upon_creation=True,
    params={
        "message": Param(
            "Airflow Lab 1 operator exercise",
            type="string",
            minLength=1,
            maxLength=200,
        )
    },
    default_args={"owner": "Adit Vakil"},
    tags=["lab1", "sandbox"],
) as dag:
    shell = BashOperator(
        task_id="show_execution_time",
        bash_command="date -u",
    )
    python = PythonOperator(
        task_id="describe_run",
        python_callable=describe_run,
        op_kwargs={"message": "{{ params.message }}"},
    )
    shell >> python
