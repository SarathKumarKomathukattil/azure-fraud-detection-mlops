import os
import json
import pandas as pd
import mlflow.xgboost

model = None
threshold = None

def init():
    global model
    global threshold

    model_root = os.environ["AZUREML_MODEL_DIR"]

    model_dir = os.path.join(model_root,'model')
    model = mlflow.xgboost.load_model(model_dir)

    threshold_file = os.path.join(model_dir,'threshold.json')
    with open(threshold_file,'r') as file:
        threshold_data = json.load(file)

    threshold = threshold_data['operating_threshold']

def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame(data['input_data'])

    probabilities = model.predict_proba(df)[:,1]

    predictions = (probabilities>=threshold).astype(int)

    return {
        'fraud_probability':probabilities.tolist(),
        'fraud_prediction':predictions.tolist() 
    }

