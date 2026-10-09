import mlflow
import mlflow.xgboost
import pandas as pd
from argparse import ArgumentParser
from xgboost import XGBClassifier
from sklearn.metrics import *

parser = ArgumentParser()
parser.add_argument("--data", type=str, required=True)

parser.add_argument("--max_depth", type=int, required=True)

parser.add_argument("--learning_rate", type=float, required=True)

parser.add_argument("--min_child_weight", type=float, required=True)

parser.add_argument("--subsample", type=float, required=True)

parser.add_argument("--colsample_bytree", type=float, required=True)

parser.add_argument("--early_stopping_rounds", type=int, required=True)

parser.add_argument("--scale_pos_weight", type=float, required=True)

args = parser.parse_args()

print(f"Input data path: {args.data}")


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

columns_to_read = feature_columns + ["is_fraud", "dataset_split"]

df = pd.read_parquet(path=args.data, columns=columns_to_read)

train_df = df[df["dataset_split"] == "train"].copy()

validation_df = df[df["dataset_split"] == "validation"].copy()

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")


X_train = train_df[feature_columns]
y_train = train_df["is_fraud"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["is_fraud"]


negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

natural_imbalance_ratio = negative_count / positive_count

print("Natural imbalance ratio:", natural_imbalance_ratio)
print("Model scale_pos_weight:", args.scale_pos_weight)

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=1000,
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    min_child_weight=args.min_child_weight,
    subsample=args.subsample,
    colsample_bytree=args.colsample_bytree,
    scale_pos_weight=args.scale_pos_weight,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=args.early_stopping_rounds,
    random_state=42,
    n_jobs=-1,
)

mlflow.start_run()
mlflow.xgboost.autolog()

model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)

validation_probability = model.predict_proba(X_validation)[:, 1]

validation_prediction = model.predict(X_validation)



pr_auc = average_precision_score(y_validation, validation_probability)
roc_auc = roc_auc_score(y_validation, validation_probability)
precision = precision_score(y_validation, validation_prediction, zero_division=0)
recall = recall_score(y_validation, validation_prediction, zero_division=0)
f1 = f1_score(y_validation, validation_prediction, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_validation, validation_prediction).ravel()



mlflow.log_metrics(
    {
        "validation_pr_auc": pr_auc,
        "validation_roc_auc": roc_auc,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,
        "validation_true_positive": int(tp),
        "validation_false_positive": int(fp),
        "validation_true_negative": int(tn),
        "validation_false_negative": int(fn),
        "best_iteration": int(model.best_iteration),
    }
)

print("\n\n----- XGBoost Scale Pos Weight Trial -----")

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


mlflow.end_run()
