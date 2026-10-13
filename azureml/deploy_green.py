from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineDeployment,CodeConfiguration,Environment

ENDPOINT_NAME = 'fraud-identify-endpoint'

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential,
                                 path='../.azureml/config.json')

inference_env = Environment(image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest',
                            conda_file='./online_scoring/conda.yml')

green = ManagedOnlineDeployment(name='green',
                        endpoint_name=ENDPOINT_NAME,
                        description='Green deployment for fraud-xgboost v3 challenger.',
                        model='azureml:fraud-xgboost:3',
                        code_configuration=CodeConfiguration(code='./online_scoring',
                                                             scoring_script='score.py'),
                        environment=inference_env,
                        app_insights_enabled=True,
                        instance_type='Standard_D2as_v4',
                        instance_count=1
                        )


print("\nCreating green deployment...")

ml_client.online_deployments.begin_create_or_update(green).result()

print("\nGreen deployment created successfully.")

endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)

endpoint.traffic = {
    'blue': 100,
    'green': 0
}

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print('Traffic coonfigured')

print('blue = 100%')
print('green = 0%')