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
    description="LightGBM tuning trial for PaySim fraud detection",
    display_name="lightgbm-tuning-trial",
    code="./src",
    inputs={
        "training_data": training_data,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30
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
    identity=ManagedIdentityConfiguration()
)

job_for_sweep = trial_job(
    num_leaves=Choice([15,31,63]),
    learning_rate=Choice([0.005,0.01,.02,0.03,0.05]),
    min_child_samples=Choice([20,50,100]),
    subsample=Choice([0.7,0.8,1.0]),
    colsample_bytree=Choice([0.7,0.8,1.0])
)

sweep_job = job_for_sweep.sweep(
    primary_metric="validation_pr_auc",
    goal="Maximize",
    sampling_algorithm="bayesian",
    max_concurrent_trials=1,
    max_total_trials=12
)

sweep_job.display_name = "lightgbm-hyperparameter-sweep"
sweep_job.experiment_name = "fraud-detection-training"
sweep_job.description = ("Bayesian LightGBM hyperparameter sweep for PaySim fraud detection")

returned_sweep_job = ml_client.jobs.create_or_update(sweep_job)

print(f"Sweep job name: {returned_sweep_job.name}")