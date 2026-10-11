import time
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(
    credential=credential,
    path='../.azureml/config.json'
)

for i in range(20):
    response = ml_client.online_endpoints.invoke(
        endpoint_name='fraud-identify-endpoint',
        request_file='./sample_request.json'
    )

    print(f'Request {i+1}: {response}')
    time.sleep(2)

