
import time

from azure.ai.ml import (
    MLClient,
    Input,
    Output,
    command
)

from azure.ai.ml.constants import (
    AssetTypes,
    InputOutputModes
)

from azure.identity import (
    DefaultAzureCredential
)


ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)


comparison_job = command(

    code="./src",

    command=(
        "python compare_champion_challenger.py "
        "--data ${{inputs.data}} "
        "--champion_model ${{inputs.champion_model}} "
        "--challenger_model ${{inputs.challenger_model}} "
        "--output_dir ${{outputs.output_dir}}"
    ),

    inputs={

        "data": Input(
            type=AssetTypes.URI_FOLDER,
            path=(
                "azureml:"
                "paysim-fraud-training-snapshot:2"
            ),
            mode=InputOutputModes.RO_MOUNT
        ),

        "champion_model": Input(
            type=AssetTypes.MLFLOW_MODEL,
            path="azureml:fraud-xgboost:2",
            mode=InputOutputModes.RO_MOUNT
        ),

        "challenger_model": Input(
            type=AssetTypes.MLFLOW_MODEL,
            path="azureml:fraud-xgboost:3",
            mode=InputOutputModes.RO_MOUNT
        )
    },

    outputs={

        "output_dir": Output(
            type=AssetTypes.URI_FOLDER
        )
    },

    # USE THE SAME ENVIRONMENT
    # AS submit_model_quality_pipeline.py
    environment=(
        "azureml:fraud-training-env:5"
    ),

    # USE THE SAME COMPUTE
    # AS YOUR EXISTING PIPELINE
    compute="cpu-fraud-training"
)


returned_job = (
    ml_client.jobs.create_or_update(
        comparison_job,
        experiment_name=(
            "fraud-champion-challenger"
        )
    )
)


print(
    "Submitted comparison job:",
    returned_job.name
)

while True:

    job = ml_client.jobs.get(
        returned_job.name
    )

    print(
        "Job status:",
        job.status
    )

    if job.status in [
        "Completed",
        "Failed",
        "Canceled"
    ]:
        break

    time.sleep(15)