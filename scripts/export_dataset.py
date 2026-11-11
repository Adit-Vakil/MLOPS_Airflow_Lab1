"""Reproduce the feature-only CSV split used by this lab."""
import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    features = load_breast_cancer(as_frame=True).data
    train, predict = train_test_split(
        features, test_size=0.2, random_state=42
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("breast_cancer_train.csv", train),
        ("breast_cancer_predict.csv", predict),
    ):
        frame.to_csv(args.output_dir / name, index=False, mode="x")
        print(f"{name}: {frame.shape[0]} rows, {frame.shape[1]} features")


if __name__ == "__main__":
    main()
