import time
from pathlib import Path

from azure.ai.ml import (
    Input,
    Output,
    command,
    dsl,
)

from azure.ai.ml.constants import (
    AssetTypes,
    InputOutputModes,
)

from azure.ai.ml.entities import (
    ManagedIdentityConfiguration,
    Model,
)

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


AZUREML_SRC_DIR = PROJECT_ROOT / "azureml" / "src"


COMPUTE_NAME = "cpu-fraud-training-4"


ENVIRONMENT_NAME = "azureml:fraud-training-env:5"


MODEL_NAME = "fraud-xgboost"


EXPERIMENT_NAME = "fraud-model-release"


# =========================================================
# 2. EVALUATION / TRAINING COMPONENT
# =========================================================

evaluate_component = command(
    name="evaluate_xgboost_final",
    display_name="Evaluate Final XGBoost",
    code=str(AZUREML_SRC_DIR),
    inputs={"data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={
        "metrics_output": Output(
            type=AssetTypes.URI_FOLDER, mode=InputOutputModes.UPLOAD
        )
    },
    command=(
        "python evaluate_xgboost_test.py "
        "--data ${{inputs.data}} "
        "--metrics_output "
        "${{outputs.metrics_output}}"
    ),
    environment=(ENVIRONMENT_NAME),
    compute=(COMPUTE_NAME),
    identity=(ManagedIdentityConfiguration()),
)


# =========================================================
# 3. QUALITY GATE COMPONENT
# =========================================================

quality_gate_component = command(
    name="xgboost_quality_gate",
    display_name=("XGBoost Model Quality Gate"),
    code=str(AZUREML_SRC_DIR),
    inputs={
        "metrics_input": Input(
            type=AssetTypes.URI_FOLDER, mode=InputOutputModes.DOWNLOAD
        )
    },
    outputs={
        "gate_output": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.UPLOAD)
    },
    command=(
        "python quality_gate.py "
        "--metrics_input "
        "${{inputs.metrics_input}} "
        "--gate_output "
        "${{outputs.gate_output}}"
    ),
    environment=(ENVIRONMENT_NAME),
    compute=(COMPUTE_NAME),
    identity=(ManagedIdentityConfiguration()),
)


# =========================================================
# 4. AZURE ML PIPELINE
# =========================================================


@dsl.pipeline(
    name=("fraud-model-release-quality-gate"),
    description=(
        "Train and evaluate XGBoost candidate " "and enforce the model quality gate."
    ),
)
def fraud_model_release_pipeline(data):

    # -----------------------------------------------------
    # Train/evaluate candidate model
    # -----------------------------------------------------

    evaluation_step = evaluate_component(data=data)

    # Give it a predictable display name.
    # Later we use this name to find the child job.

    evaluation_step.display_name = "evaluation_step"

    # -----------------------------------------------------
    # Run quality gate using evaluation metrics
    # -----------------------------------------------------

    gate_step = quality_gate_component(
        metrics_input=(evaluation_step.outputs.metrics_output)
    )

    gate_step.display_name = "quality_gate_step"

    # -----------------------------------------------------
    # Pipeline outputs
    # -----------------------------------------------------

    return {
        "evaluation_metrics": evaluation_step.outputs.metrics_output,
        "quality_gate_result": gate_step.outputs.gate_output,
    }


# =========================================================
# 5. WAIT FOR AZURE ML PIPELINE
# =========================================================


def wait_for_pipeline(ml_client, job_name, poll_seconds=30):

    terminal_states = {"Completed", "Failed", "Canceled", "NotResponding"}

    while True:

        job = ml_client.jobs.get(job_name)

        print("Azure ML pipeline state:", job.status)

        if job.status in terminal_states:
            return job

        time.sleep(poll_seconds)


# =========================================================
# 6. FIND EVALUATION CHILD JOB
# =========================================================


def get_evaluation_job(ml_client, parent_job_name):

    child_jobs = list(ml_client.jobs.list(parent_job_name=(parent_job_name)))

    for child_job in child_jobs:

        if child_job.display_name == "evaluation_step":

            return child_job

    # Helpful debugging if Azure changes the child name.

    print("Available child jobs:")

    for child_job in child_jobs:

        print("Name:", child_job.name, "| Display name:", child_job.display_name)

    raise RuntimeError("Evaluation child job " "could not be found.")


# =========================================================
# 7. REGISTER APPROVED CHALLENGER MODEL
# =========================================================


def register_candidate_model(ml_client, evaluation_job_name, data_version_tag):

    # evaluate_xgboost_test.py logs the MLflow model
    # under the model artifact path.
    #
    # We register that exact artifact.

    model_path = (
        f"azureml://jobs/"
        f"{evaluation_job_name}"
        f"/outputs/artifacts/"
        f"paths/model/"
    )

    model_to_register = Model(
        name=(MODEL_NAME),
        path=(model_path),
        type=(AssetTypes.MLFLOW_MODEL),
        description=(
            "XGBoost fraud detection model " "that passed the model quality gate."
        ),
        tags={
            "business_domain": "fraud_detection",
            "task": "binary_classification",
            "model_family": "xgboost",
            "quality_gate": "passed",
            "data_version": data_version_tag,
            "threshold_policy": "recall_70_max_precision",
            "release_status": "challenger",
        },
    )

    registered_model = ml_client.models.create_or_update(model=(model_to_register))

    return registered_model


# =========================================================
# 8. MAIN REUSABLE RETRAINING FUNCTION
# =========================================================


def submit_model_pipeline(ml_client, data_path, data_version_tag):
    """
    Submit the Azure ML fraud retraining pipeline.

    Parameters
    ----------
    ml_client:
        Authenticated Azure ML MLClient.

    data_path:
        Azure ML data asset reference.

        Example:
        azureml:paysim-fraud-training-snapshot:abc123

    data_version_tag:
        Human-readable data version stored
        in registered model tags.

    Returns
    -------
    dict containing:

        pipeline_job_id
        model_name
        model_version
    """

    print("Training data:", data_path)

    # -----------------------------------------------------
    # Create Azure ML pipeline input
    # -----------------------------------------------------

    training_data = Input(
        type=(AssetTypes.URI_FOLDER), path=(data_path), mode=(InputOutputModes.RO_MOUNT)
    )

    # -----------------------------------------------------
    # Build pipeline job
    # -----------------------------------------------------

    pipeline_job = fraud_model_release_pipeline(data=(training_data))

    # -----------------------------------------------------
    # Submit pipeline to Azure ML
    # -----------------------------------------------------

    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=(EXPERIMENT_NAME)
    )

    print("Submitted Azure ML pipeline:", submitted_job.name)

    # -----------------------------------------------------
    # Wait until Azure ML finishes
    # -----------------------------------------------------

    completed_job = wait_for_pipeline(
        ml_client=(ml_client), job_name=(submitted_job.name)
    )

    # -----------------------------------------------------
    # Quality gate protection
    # -----------------------------------------------------
    #
    # quality_gate.py returns failure when the candidate
    # does not satisfy the required thresholds.
    #
    # Therefore the Azure ML pipeline will not finish as
    # Completed.
    #
    # In that case we stop here.
    #
    # NO model registration happens.
    # -----------------------------------------------------

    if completed_job.status != "Completed":

        raise RuntimeError(
            "Azure ML retraining pipeline "
            "did not complete successfully. "
            "Candidate model will NOT "
            "be registered. "
            f"Final state: "
            f"{completed_job.status}"
        )

    print("Quality gate passed.")

    # -----------------------------------------------------
    # Find evaluation child job
    # -----------------------------------------------------

    evaluation_job = get_evaluation_job(
        ml_client=(ml_client), parent_job_name=(completed_job.name)
    )

    print("Evaluation job:", evaluation_job.name)

    # -----------------------------------------------------
    # Register candidate model
    # -----------------------------------------------------

    registered_model = register_candidate_model(
        ml_client=(ml_client),
        evaluation_job_name=(evaluation_job.name),
        data_version_tag=(data_version_tag),
    )

    print(
        "Registered challenger:",
        f"{registered_model.name}:" f"{registered_model.version}",
    )

    # -----------------------------------------------------
    # Return information to Databricks Task 12
    # -----------------------------------------------------

    return {
        "pipeline_job_id": str(submitted_job.name),
        "model_name": str(registered_model.name),
        "model_version": str(registered_model.version),
    }
