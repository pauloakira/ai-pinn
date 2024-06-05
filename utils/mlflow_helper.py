# Python libs
import mlflow
from datetime import datetime

def mlflowPipeline(experiment_name: str, tag: str, data: dict):
    # Set MLFlow experiment
    mlflow.set_experiment(experiment_name)

    # Start MLFlow run
    with mlflow.start_run(run_name=f"{tag}"):
        # Define a name for the run
        mlflow.set_tag("mlflow.runName", f"{tag}")
        mlflow.log_params(data)

    



