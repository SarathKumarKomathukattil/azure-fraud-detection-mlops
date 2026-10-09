from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration

credential = DefaultAzureCredential()


ml_client = MLClient.from_config(credential=credential, path="../.azureml/config.json")


training_data = Input(
    type=AssetTypes.URI_FOLDER,
    path="azureml:paysim-fraud-training-snapshot:1",
    mode=InputOutputModes.RO_MOUNT,
)


job = command(
    description="LightGBM baseline for PaySim fraud detection",
    display_name="lightgbm-baseline",
    code="./src",
    inputs={
        "training_data": training_data,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
    },
    command=(
        "python train_lightgbm.py "
        "--data ${{inputs.training_data}} "
        "--num_leaves ${{inputs.num_leaves}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--min_child_samples ${{inputs.min_child_samples}} "
        "--subsample ${{inputs.subsample}} "
        "--colsample_bytree ${{inputs.colsample_bytree}} "
        "--early_stopping_rounds ${{inputs.early_stopping_rounds}}"
    ),
    experiment_name="fraud-detection-training",
    environment="azureml:fraud-training-env:4",
    compute="cpu-fraud-training",
    identity=ManagedIdentityConfiguration(),
)


returned_job = ml_client.jobs.create_or_update(job)


print(f"LightGBM job name: {returned_job.name}")
