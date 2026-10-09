from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path="../.azureml/config.json"
)

training_data = Input(
    type=AssetTypes.URI_FOLDER,
    path="azureml:paysim-fraud-training-snapshot:1",
    mode=InputOutputModes.RO_MOUNT
)

job = command(
    display_name="xgboost-final-threshold-validation",

    code="./src",

    command=(
        "python train_xgboost_threshold.py "
        "--data ${{inputs.training_data}}"
    ),

    inputs={
        "training_data": training_data
    },

    environment="azureml:fraud-training-env:3",

    compute="cpu-fraud-training",

    experiment_name="fraud-detection-training",

    identity=ManagedIdentityConfiguration()
)

returned_job = ml_client.jobs.create_or_update(job)

print("Submitted job:", returned_job.name)