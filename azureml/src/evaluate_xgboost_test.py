import os
import json
import mlflow
import mlflow.xgboost
import pandas as pd
from argparse import ArgumentParser
from xgboost import XGBClassifier
from sklearn.metrics import *


parser = ArgumentParser()
parser.add_argument(
    '--data',
    type=str,
    required=True
)

parser.add_argument(
    '--metrics_output',
    type=str,
    required=True
)


args =parser.parse_args()
print(f'Input data path: {args.data}')

feature_columns = [
    "amount",
    "log_amount",
    "type_cash_in",
    "type_cash_out",
    "type_debit",
    "type_payment",
    "type_transfer",
    "origin_txn_count_prior",
    "origin_avg_amount_prior",
    "origin_total_amount_prior",
    "amount_vs_origin_avg",
    "origin_has_history",
]

columns_to_read = feature_columns + ['is_fraud','dataset_split']

df = pd.read_parquet(path=args.data,
                     columns=columns_to_read)

train_df = df[df['dataset_split']=='train'].copy()
test_df = df[df['dataset_split']=='test'].copy()
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

X_train = train_df[feature_columns]
y_train = train_df["is_fraud"]

X_test = test_df[feature_columns]
y_test = test_df['is_fraud']

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=68,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=5.0,
    subsample=0.7,
    colsample_bytree=1.0,
    scale_pos_weight=300.0,
    tree_method="hist",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

final_threshold = 0.43953168

model.fit(X_train,y_train)

test_probability = model.predict_proba(X_test)[:,1]
test_prediction = (test_probability >= final_threshold).astype(int)

test_pr_auc = average_precision_score(y_test,test_probability)
test_roc_auc = roc_auc_score(y_test,test_probability)

test_precision = precision_score(y_test,test_prediction,zero_division=0)
test_recall = recall_score(y_test,test_prediction,zero_division=0)
test_f1 = f1_score(y_test,test_prediction,zero_division=0)

test_tn,test_fp,test_fn,test_tp = confusion_matrix(y_test,test_prediction).ravel()


mlflow.log_param("final_operating_threshold",final_threshold)
mlflow.log_param("threshold_selection_policy","validation_recall_gte_70_maximize_precision")


test_fpr = test_fp/(test_fp+test_tn)
mlflow.log_metrics(
    {
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,

        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,

        "test_true_positive": int(test_tp),
        "test_false_positive": int(test_fp),
        "test_true_negative": int(test_tn),
        "test_false_negative": int(test_fn),
        "test_false_positive_rate": float(test_fpr)
    }
)

mlflow.xgboost.log_model(
    xgb_model=model,
    artifact_path='model',
    input_example=X_train.head(5)
)
mlflow.log_dict({
    'operating_threshold':final_threshold
},'model/threshold.json')

metrics = {
    "test_pr_auc": float(test_pr_auc),
    "test_roc_auc": float(test_roc_auc),

    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),

    "test_true_positive": int(test_tp),
    "test_false_positive": int(test_fp),
    "test_true_negative": int(test_tn),
    "test_false_negative": int(test_fn),

    'test_false_positive_rate': float(test_fpr)

}


os.makedirs(
    args.metrics_output,
    exist_ok=True
)

metrics_file = os.path.join(args.metrics_output,'metrics.json')

with open(metrics_file,'w') as file:
    json.dump(metrics,file,indent=4)

print(f'Metrics written to: {metrics_file}')

print("\n----- FINAL XGBOOST TEST RESULTS -----")

print("Test PR-AUC    :", test_pr_auc)
print("Test ROC-AUC   :", test_roc_auc)

print("\nLocked Threshold :", final_threshold)

print("Test Precision :", test_precision)
print("Test Recall    :", test_recall)
print("Test F1        :", test_f1)


print("\nTest Confusion Matrix")

print("TP:", test_tp)
print("FP:", test_fp)
print("TN:", test_tn)
print("FN:", test_fn)


