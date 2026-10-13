from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ENDPOINT_NAME = "fraud-identify-endpoint"
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential,
                                 path='../.azureml/config.json')

endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)

endpoint.traffic = {
    'blue': 0,
    'green': 100
}

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("\nTraffic updated")
print("blue = 0%")
print("green = 100%")