FROM apache/airflow:3.4.0-python3.12

RUN pip install --no-cache-dir \
    "apache-airflow==3.4.0" \
    "pandas==2.2.3" \
    "scikit-learn==1.6.1" \
    "kneed==0.8.5"
