# MLOPS_Airflow_Lab1

An Airflow pipeline that uses K-Means and the elbow method to cluster
feature-only measurements from scikit-learn's Breast Cancer Wisconsin dataset.
The split uses seed 42 and contains 455 training rows and 114 prediction rows,
each with 30 numeric features. Diagnosis labels are excluded.

The DAG validates the CSVs, fits a StandardScaler on training data, evaluates
k from 1 to 10, and saves the elbow-selected model with its fitted scaler.
The final task reloads the pipeline and writes a cluster assignment for every
prediction row. Tasks exchange artifact paths through XCom. Each run writes to
its own directory under working_data/.

## Run locally

From the repository root, start Docker Desktop. On a fresh checkout, create
the local environment file and mount directories. Keep .env private.

    python3 -c 'import os, base64, secrets; from pathlib import Path; key=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(); fd=os.open(".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, f"AIRFLOW_UID={os.getuid()}\nFERNET_KEY={key}\n".encode()); os.close(fd)'
    mkdir -p logs plugins config working_data

The custom image pins Airflow 3.4.0, Python 3.12, pandas 2.2.3,
scikit-learn 1.6.1, and kneed 0.8.5.

    docker compose config --quiet
    docker compose build
    docker compose up airflow-init
    docker compose up -d

Open http://localhost:8080. The local template uses airflow / airflow.
Use a fresh run ID for each trigger.

    docker compose exec airflow-scheduler airflow dags unpause MLOPS_Airflow_Lab1
    docker compose exec airflow-scheduler airflow dags trigger MLOPS_Airflow_Lab1 --run-id local_001
    docker compose exec airflow-scheduler airflow tasks states-for-dag-run MLOPS_Airflow_Lab1 local_001

## Verification

All four tasks succeeded in verification_001 and verification_002. Both runs
selected k=3 and produced identical predictions across separate output
directories. The 114 rows were assigned to clusters in counts of 73, 27,
and 14. Run reports are saved in evidence/.

Six tests passed, covering valid input, invalid input, saved-model predictions,
and failure when no elbow is detected.

    docker compose exec -T -e PYTHONPATH=/opt/airflow/dags airflow-worker python - < tests/test_validation.py
    docker compose exec -T -e PYTHONPATH=/opt/airflow/dags airflow-worker python - < tests/test_model.py

The dataset export script reproduced both CSVs byte for byte. To regenerate
them in a new directory while the worker is running:

    docker compose exec -T airflow-worker python - --output-dir /opt/airflow/working_data/regenerated < scripts/export_dataset.py

The separate sandbox DAG demonstrates BashOperator, PythonOperator, a
validated message parameter, and a daily 09:00 UTC schedule. It starts paused.
Both tasks succeeded in a manual test.

    docker compose exec airflow-scheduler airflow dags test MLOPS_Airflow_Lab1_Sandbox

## Reference and limitations

This project adapts the workflow from [the professor's Airflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1).
This implementation saves the elbow-selected candidate, applies the same
fitted scaler at prediction time, and saves all cluster assignments. Elbow
selection is a heuristic. Repeatability establishes that the workflow
reproduces its results, while cluster quality requires separate evaluation.

To stop Airflow while retaining the database volume, run docker compose down.
