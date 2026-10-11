from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (ManagedOnlineEndpoint, 
                                  ManagedOnlineDeployment,
                                  Environment,
                                  CodeConfiguration)

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential, path="../.azureml/config.json")

endpoint_name = "fraud-identify-endpoint"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Real-time fraud detection endpoint",
    auth_mode="key",
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name}")

model = ml_client.models.get(name="fraud-xgboost", version=2)

inference_env = Environment(image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest',
                            conda_file='./online_scoring/conda.yml')

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    instance_type="Standard_DS2_v2",
    instance_count=1,
    code_configuration=CodeConfiguration(code='./online_scoring',
                                         scoring_script='score.py'),
    environment=inference_env,
    app_insights_enabled=True
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Blue deployment created")

endpoint.traffic = {"blue": 100}

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

logs = ml_client.online_deployments.get_logs(name='blue',
                                             endpoint_name=endpoint_name,
                                             lines=1)

print(logs)

response = ml_client.online_endpoints.invoke(endpoint_name=endpoint_name,
                                             request_file='./sample_request.json')
print(response)

