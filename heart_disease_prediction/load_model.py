import os
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Tuple, Dict
from datetime import date
from pathlib import Path

def load_model(
    paths: Dict,
    # save_dir: str = "../models/"
) -> Tuple[str, str]:
    """
    Loads the latest model and preprocessor from MLflow, and saves both in one pickle file.

    Parameters:
    ----------
    model_name : str
        Name of the registered model (e.g., "best_model_2025-07-28")
    experiment_name : str
        Name of the MLflow experiment
    save_dir : str
        Directory where combined .pkl file will be saved

    Returns:
    -------
    str
        Path to saved combined .pkl file
    """

    assert isinstance(paths["model_name"], str), "model_name must be a string"
    assert isinstance(paths["experiment_name"], str), "experiment_name must be a string"

    # Ensure tracking URI is set
    mlflow.set_tracking_uri(paths["mlflow_db_path"])
    client = MlflowClient()

    # Get latest version of registered model and preprocessor
    versions = client.get_latest_versions(name=paths["model_name"])
    latest_model = versions[0] if versions else None
    model_uri = f"models:/{paths["model_name"]}/{latest_model.version}"

    # preprocessor_versions = client.get_latest_versions(name="Preprocessor")
    # latest_preprocessor = preprocessor_versions[0] if preprocessor_versions else None
    # preprocessor_uri = f"models:/Preprocessor/{latest_preprocessor.version}"  # Adjust path if needed

    # Load artifacts using MLflow
    print(f"📦 Downloading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # print(f"📦 Downloading preprocessor from {preprocessor_uri}")
    # preprocessor = mlflow.sklearn.load_model(preprocessor_uri)

    # Ensure output directory exists
    os.makedirs(paths["final_save_dir"], exist_ok=True)

    combined_path = os.path.join(paths["final_save_dir"], "pipeline.pkl")

    with open(combined_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Combined model + preprocessor saved at: {combined_path}")

if __name__ == "__main__":
    if '__file__' in globals():
        project_root = Path(__file__).resolve().parents[1]
    paths = {
        "model_name": "best_model_2025-07-30",
        "mlflow_db_path": f"sqlite:///{project_root}/mlruns/mlflow.db",
        "artifact_loc": f"file://{project_root}/mlruns/artifacts/",
        "experiment_name": "heart-disease-experiment-pipeline",
        "final_save_dir": f"{project_root}/models/"
    }
    load_model(paths)
    # load_model(paths, save_dir="./models/")
