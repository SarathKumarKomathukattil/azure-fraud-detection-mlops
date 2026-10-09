from azure.ai.ml.sweep import Choice
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

trial_job = command(
    description=(
        "XGBoost scale_pos_weight experiment "
        "for PaySim fraud detection"
    ),

    display_name="xgboost-weight-trial",

    code="./src",

    inputs={
        "training_data": training_data,

        # fixed best XGBoost hyperparameters
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5.0,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "early_stopping_rounds": 30,

        # only this one will vary
        "scale_pos_weight": 1.0
    },

    command=(
        "python train_xgboost_weight.py "
        "--data ${{inputs.training_data}} "
        "--max_depth ${{inputs.max_depth}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--min_child_weight ${{inputs.min_child_weight}} "
        "--subsample ${{inputs.subsample}} "
        "--colsample_bytree ${{inputs.colsample_bytree}} "
        "--early_stopping_rounds ${{inputs.early_stopping_rounds}} "
        "--scale_pos_weight ${{inputs.scale_pos_weight}}"
    ),

    experiment_name="fraud-detection-training",

    environment="azureml:fraud-training-env:3",

    compute="cpu-fraud-training",

    identity=ManagedIdentityConfiguration()
)

job_for_sweep = trial_job(
    scale_pos_weight=Choice([
        1.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1050.0
    ])
)

sweep_job = job_for_sweep.sweep(
    primary_metric="validation_pr_auc",
    goal="Maximize",
    sampling_algorithm="grid",
    max_concurrent_trials=1
)

sweep_job.display_name = (
    "xgboost-scale-pos-weight-sweep"
)

sweep_job.experiment_name = (
    "fraud-detection-training"
)

sweep_job.description = (
    "Controlled XGBoost experiment varying "
    "only scale_pos_weight"
)

returned_sweep_job = (
    ml_client.jobs.create_or_update(
        sweep_job
    )
)

print(
    f"Sweep job name: "
    f"{returned_sweep_job.name}"
)