from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ENDPOINT_NAME = "fraud-identify-endpoint"

credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path="../.azureml/config.json"
)

response = ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    deployment_name="green",
    request_file="sample_request.json"
)

print("Green response:")
print(response)