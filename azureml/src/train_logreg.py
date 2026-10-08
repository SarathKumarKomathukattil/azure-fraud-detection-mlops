import mlflow
import mlflow.sklearn
import pandas as pd
from argparse import ArgumentParser
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


parser = ArgumentParser()

parser.add_argument(
    '--data',
    type=str,
    required=True)

args = parser.parse_args()

print(f'Input data path: {args.data}')

feature_columns = [
    'amount',
    'log_amount',

    'type_cash_in',
    'type_cash_out',
    'type_debit',
    'type_payment',
    'type_transfer',

    'origin_txn_count_prior',
    'origin_avg_amount_prior',
    'origin_total_amount_prior',
    'amount_vs_origin_avg',
    'origin_has_history',
]

columns_to_read = (
    feature_columns + ['is_fraud','dataset_split']
)

df = pd.read_parquet(path=args.data,
                     columns=columns_to_read)

print("Total rows:", len(df))

train_df = df[df['dataset_split']=='train'].copy()

validation_df = df[df['dataset_split']=='validation'].copy()

print(f'Train rows: {len(train_df)}')
print(f'Validation rows: {len(validation_df)}')


X_train = train_df[feature_columns]
y_train = train_df['is_fraud']

X_validation = validation_df[feature_columns]
y_validation = validation_df['is_fraud']

model = Pipeline(
    [
        ('scaler',StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000
        ))
    ]
)

mlflow.start_run() #important to start before autolog to avoid 2 jobs insted of 1
mlflow.sklearn.autolog()

model.fit(X_train,y_train)

validation_probability = model.predict_proba(
    X_validation
)[:,1]

validation_prediction = model.predict(
    X_validation
)

pr_auc = average_precision_score(y_validation,validation_probability)

roc_auc = roc_auc_score(y_validation,validation_probability)

precision = precision_score(y_validation,validation_prediction,zero_division=0)

recall = recall_score(y_validation,validation_prediction,zero_division=0)

f1 = f1_score(y_validation,validation_prediction,zero_division=0)

tn,fp,fn,tp = confusion_matrix(y_validation,validation_prediction).ravel()


mlflow.log_metrics({
    "validation_pr_auc": pr_auc,
    "validation_roc_auc": roc_auc,
    "validation_precision": precision,
    "validation_recall": recall,
    "validation_f1": f1,

    "validation_true_positive": int(tp),
    "validation_false_positive": int(fp),
    "validation_true_negative": int(tn),
    "validation_false_negative": int(fn)
})

print("\n\n----- Logistic Regression Baseline -----")

print("Validation PR-AUC    :", pr_auc)
print("Validation ROC-AUC   :", roc_auc)
print("Validation Precision :", precision)
print("Validation Recall    :", recall)
print("Validation F1        :", f1)

print("\nConfusion Matrix")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)