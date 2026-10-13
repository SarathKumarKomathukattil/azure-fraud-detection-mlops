from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration, Model

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential,
                                 path="../.azureml/config.json")

evaluate_component = command(
    name="evaluate_xgboost_final",
    display_name="Evaluate Final XGBoost",
    code="./src",
    inputs={"data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={"metrics_output":Output(type=AssetTypes.URI_FOLDER,
                                      mode=InputOutputModes.UPLOAD)},
    command=("python evaluate_xgboost_test.py "
    "--data ${{inputs.data}} "
    "--metrics_output ${{outputs.metrics_output}}"),
    environment="azureml:fraud-training-env:5",
    compute="cpu-fraud-training",
    identity=ManagedIdentityConfiguration()
)

quality_gate_component = command(
    name="xgboost_quality_gate",
    display_name="XGBoost Model Quality Gate",
    code="./src",
    inputs={"metrics_input": Input(type=AssetTypes.URI_FOLDER,
                                   mode=InputOutputModes.DOWNLOAD)},
    outputs={"gate_output": Output(type=AssetTypes.URI_FOLDER,
                                   mode=InputOutputModes.UPLOAD)},
    command="python quality_gate.py "
    "--metrics_input ${{inputs.metrics_input}} "
    "--gate_output ${{outputs.gate_output}}",
    environment="azureml:fraud-training-env:5",
    compute="cpu-fraud-training",
    identity=ManagedIdentityConfiguration()
)


@dsl.pipeline(
    name="fraud-model-release-quality-gate",
    description="Evaluate final XGBoost model and enforce model quality gate"
)
def fraud_model_release_pipeline(data):
    evaluation_step = evaluate_component(data=data)
    gate_step = quality_gate_component(
        metrics_input=evaluation_step.outputs.metrics_output
    )
    return {"evaluation_metrics":evaluation_step.outputs.metrics_output,
            "quality_gate_result":gate_step.outputs.gate_output}


training_data = Input(
    type=AssetTypes.URI_FOLDER,
    path="azureml:paysim-fraud-training-snapshot:2",
    mode=InputOutputModes.RO_MOUNT
)

pipeline_job = fraud_model_release_pipeline(data=training_data)

returned_job = ml_client.jobs.create_or_update(pipeline_job,
                                               experiment_name="fraud-model-release")

print(f"Submitted pipeline: {returned_job.name}")



ml_client.jobs.stream(returned_job.name)

completed_job = ml_client.jobs.get(returned_job.name)

if completed_job.status != 'Completed':
    raise RuntimeError('Pipeline failed. Model will not be registered')

child_jobs = list(ml_client.jobs.list(parent_job_name=completed_job.name))

evaluate_job = None
for job in child_jobs:
    if job.display_name == 'evaluation_step':
        evaluate_job = job
        break

if evaluate_job is None:
    raise RuntimeError("Evaluation job could not be found.")

model_to_register = Model(
    name='fraud-xgboost',
    path=f"azureml://jobs/{evaluate_job.name}/outputs/artifacts/paths/model/",
    type=AssetTypes.MLFLOW_MODEL,
    description='XGBoost fraud detection model that passed the model quality gate.',
    tags={
        "business_domain": "fraud_detection",
        "task": "binary_classification",
        "model_family": "xgboost",
        "quality_gate": "passed",
        "data_version": "paysim-fraud-training-snapshot:2",
        "threshold_policy": "recall_70_max_precision",
        "release_status": "challenger"
    }
)

registered_model = ml_client.models.create_or_update(model=model_to_register)


print(f'Registerd model: {registered_model.name} '
      f'\nVersion: {registered_model.version}')