"""Feature-only clustering pipeline with artifacts isolated by Airflow run."""
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path("/opt/airflow/working_data")


def load_data(run_id):
    train = pd.read_csv(DATA_DIR / "breast_cancer_train.csv")
    predict = pd.read_csv(DATA_DIR / "breast_cancer_predict.csv")

    if train.empty or predict.empty:
        raise ValueError("Training and prediction data must be nonempty")
    if not train.columns.equals(predict.columns):
        raise ValueError("Training and prediction columns must match in order")
    if {"target", "diagnosis", "label"}.intersection(train.columns):
        raise ValueError("Diagnosis labels must not be clustering features")

    for name, frame in (("training", train), ("prediction", predict)):
        if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
            raise ValueError(f"{name} features must be numeric")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{name} data contains missing or infinite values")

    run_key = hashlib.sha256(run_id.encode()).hexdigest()
    folder = OUTPUT_DIR / run_key
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False)
    predict.to_csv(folder / "predict.csv", index=False)
    return {"folder": str(folder), "run_id": run_id}


def data_preprocessing(metadata):
    folder = Path(metadata["folder"])
    train = pd.read_csv(folder / "train.csv")
    predict = pd.read_csv(folder / "predict.csv")

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_predict = scaler.transform(predict)

    joblib.dump(scaler, folder / "scaler.joblib")
    np.save(folder / "train_scaled.npy", scaled_train)
    np.save(folder / "predict_scaled.npy", scaled_predict)
    return metadata


def build_save_model(metadata):
    import json

    from kneed import KneeLocator
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline

    folder = Path(metadata["folder"])
    train = np.load(folder / "train_scaled.npy")
    max_k = min(10, len(train) - 1, len(np.unique(train, axis=0)))
    if max_k < 3:
        raise ValueError("Need at least three distinct training samples")

    candidates = list(range(1, max_k + 1))
    models = {}
    inertias = []
    for k in candidates:
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=20,
            max_iter=300, random_state=42,
        ).fit(train)
        models[k] = model
        inertias.append(float(model.inertia_))

    elbow = KneeLocator(
        candidates, inertias, curve="convex", direction="decreasing"
    ).elbow
    if elbow is None:
        raise ValueError("No elbow detected; inspect the inertia results")

    selected_k = int(elbow)
    pipeline = Pipeline([
        ("scaler", joblib.load(folder / "scaler.joblib")),
        ("kmeans", models[selected_k]),
    ])
    joblib.dump(pipeline, folder / "clustering_pipeline.joblib")

    report = {
        "run_id": metadata["run_id"],
        "selected_k": selected_k,
        "candidate_k": candidates,
        "inertia": inertias,
        "training_rows": int(train.shape[0]),
        "feature_count": int(train.shape[1]),
        "random_state": 42,
    }
    (folder / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return metadata


def load_model_predict(metadata):
    import json

    folder = Path(metadata["folder"])
    pipeline = joblib.load(folder / "clustering_pipeline.joblib")
    predict = pd.read_csv(folder / "predict.csv")
    labels = pipeline.predict(predict)
    k = pipeline.named_steps["kmeans"].n_clusters

    if len(labels) != len(predict):
        raise ValueError("Prediction count does not match input rows")
    if not np.isin(labels, np.arange(k)).all():
        raise ValueError("Predictions contain invalid cluster IDs")

    result = predict.copy()
    result["cluster"] = labels
    result.to_csv(folder / "predictions.csv", index=False)

    summary = {
        "run_id": metadata["run_id"],
        "selected_k": int(k),
        "prediction_rows": int(len(labels)),
        "cluster_counts": {
            str(i): int(np.sum(labels == i)) for i in range(k)
        },
        "artifact_folder": str(folder),
    }
    (folder / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
