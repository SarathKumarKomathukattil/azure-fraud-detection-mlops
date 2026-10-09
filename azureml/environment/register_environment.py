from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

credential = DefaultAzureCredential()


ml_client = MLClient.from_config(credential=credential, path="../../.azureml/config.json")

training_env = Environment(
    name="fraud-training-env",
    version='4',
    description="Training environment for PaySim fraud models",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    conda_file="./conda.yml",
)

registered_env = ml_client.environments.create_or_update(training_env)

print(f"Environment: {registered_env}")
print(f"Version: {registered_env.version}")
