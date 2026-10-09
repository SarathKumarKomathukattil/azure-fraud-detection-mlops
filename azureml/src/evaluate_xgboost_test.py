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

mlflow.start_run()
mlflow.xgboost.autolog()

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
    }
)

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


mlflow.end_run()