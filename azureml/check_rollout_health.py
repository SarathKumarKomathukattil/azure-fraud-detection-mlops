from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ENDPOINT_NAME = "fraud-identify-endpoint"

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path="../.azureml/config.json"
)

blue = ml_client.online_deployments.get(
    name="blue",
    endpoint_name=ENDPOINT_NAME
)

green = ml_client.online_deployments.get(
    name="green",
    endpoint_name=ENDPOINT_NAME
)

endpoint = ml_client.online_endpoints.get(
    name=ENDPOINT_NAME
)

print("\nTraffic:", endpoint.traffic)
print("Blue state :", blue.provisioning_state)
print("Green state:", green.provisioning_state)

response = ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    request_file="sample_request.json"
)

print("Endpoint response:")
print(response)