from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential


crredential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=crredential,
                                 path='../.azureml.config.json')


data_asset = Data(name='paysim-fraud-training-snapshot',
            version='2',
            description='CT retraining snapshot containing historical labeled data '
            'plus matured production data.',
            path='abfss://fraudmlops@fraudmlopsstorage.dfs.core.windows.net/training_snapshots/paysim/v3',
            type=AssetTypes.URI_FOLDER)


registered_data = ml_client.data.create_or_update(data_asset)

print(f'Registered data: {registered_data.name}:{registered_data.version}')