from azure.ai.ml.sweep import Choice
from azure.ai.ml import MLClient,command,Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes,InputOutputModes
from azure.ai.ml.entities import ManagedIdentityConfiguration


credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential,
                                 path='../.azureml/config.json')

training_data = Input(type=AssetTypes.URI_FOLDER,
                      path='azureml:paysim-fraud-training-snapshot:1',
                      mode=InputOutputModes.RO_MOUNT)

trial_job = command(
    description='XGBoost baseline for PaySim fraud detection',
    display_name='xgboost-baseline',
    code='./src',
    inputs={'training_data':training_data,
            'max_depth':6,
            'learning_rate':0.1,
            'min_child_weight':1.0,
            'subsample':0.8,
            'colsample_bytree':0.8,
            'early_stopping_rounds':30},
            
    command= (
        'python train_xgboost_tune.py ' 
        '--data ${{inputs.training_data}} '
        '--max_depth ${{inputs.max_depth}} '
        '--learning_rate ${{inputs.learning_rate}} '
        '--min_child_weight ${{inputs.min_child_weight}} '
        '--subsample ${{inputs.subsample}} '
        '--colsample_bytree ${{inputs.colsample_bytree}} '
        '--early_stopping_rounds ${{inputs.early_stopping_rounds}}'
    ),

    experiment_name='fraud-detection-training',
    environment='azureml:fraud-training-env:3',
    compute='cpu-fraud-training',
    identity=ManagedIdentityConfiguration()
)

job_for_sweep = trial_job(
    max_depth = Choice([4,6,8]),
    learning_rate = Choice([0.03,0.05,0.1,0.2]),
    min_child_weight = Choice([1.0,5.0,10.0]),
    subsample = Choice([0.7,0.8,1.0]),
    colsample_bytree = Choice([0.7,0.8,1.0])
)

sweep_job = job_for_sweep.sweep(
    primary_metric='validation_pr_auc',
    goal='Maximize',
    sampling_algorithm='bayesian',
    max_concurrent_trials=1,
    max_total_trials=12
)

sweep_job.display_name = 'xgboost-hyperparameter-sweep'
sweep_job.experiment_name = 'fraud-detection-training'
sweep_job.description = 'Bayesian hyperparameter sweep for PaySim XGBoost fraud detection'



returned_sweep_job = ml_client.jobs.create_or_update(sweep_job)

print(f'Sweep job name: {returned_sweep_job.name}')


