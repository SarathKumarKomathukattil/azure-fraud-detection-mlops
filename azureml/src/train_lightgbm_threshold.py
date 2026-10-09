import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import lightgbm as lgb

from argparse import ArgumentParser
from lightgbm import LGBMClassifier
from sklearn.metrics import *



parser = ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True
)

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

columns_to_read = feature_columns + [
    "is_fraud",
    "dataset_split"
]



df = pd.read_parquet(
    path=args.data,
    columns=columns_to_read
)

train_df = df[
    df["dataset_split"] == "train"
].copy()

validation_df = df[
    df["dataset_split"] == "validation"
].copy()

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")


X_train = train_df[feature_columns]
y_train = train_df["is_fraud"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["is_fraud"]



model = LGBMClassifier(
    objective="binary",

    n_estimators=1000,

    num_leaves=15,
    learning_rate=0.005,
    min_child_samples=20,

    subsample=1.0,
    subsample_freq=1,

    colsample_bytree=0.7,

    scale_pos_weight=1.0,

    random_state=42,
    n_jobs=-1
)




mlflow.start_run()

mlflow.lightgbm.autolog()

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_validation, y_validation)
    ],

    eval_names=[
        "validation"
    ],

    eval_metric="average_precision",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            first_metric_only=True,
            verbose=False
        )
    ]
)



validation_probability = model.predict_proba(
    X_validation
)[:, 1]



validation_prediction = model.predict(
    X_validation
)



precision_values, recall_values, thresholds = precision_recall_curve(
    y_validation,
    validation_probability
)



f1_values = (
    2
    * precision_values[:-1]
    * recall_values[:-1]
    / (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-10
    )
)

best_index = np.argmax(f1_values)

best_threshold = thresholds[
    best_index
]

optimized_prediction = (
    validation_probability >= best_threshold
).astype(int)



target_recall = 0.70

precision_for_thresholds = precision_values[:-1]
recall_for_thresholds = recall_values[:-1]


# Keep only thresholds where recall >= 70%
valid_indices = np.where(
    recall_for_thresholds >= target_recall
)[0]


# Among those thresholds,
# choose the one with highest precision
best_target_index = valid_indices[
    np.argmax(
        precision_for_thresholds[
            valid_indices
        ]
    )
]


same_recall_threshold = thresholds[
    best_target_index
]


same_recall_prediction = (
    validation_probability
    >= same_recall_threshold
).astype(int)




pr_auc = average_precision_score(
    y_validation,
    validation_probability
)

roc_auc = roc_auc_score(
    y_validation,
    validation_probability
)


precision = precision_score(
    y_validation,
    validation_prediction,
    zero_division=0
)

recall = recall_score(
    y_validation,
    validation_prediction,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    validation_prediction,
    zero_division=0
)


tn, fp, fn, tp = confusion_matrix(
    y_validation,
    validation_prediction
).ravel()



optimized_precision = precision_score(
    y_validation,
    optimized_prediction,
    zero_division=0
)

optimized_recall = recall_score(
    y_validation,
    optimized_prediction,
    zero_division=0
)

optimized_f1 = f1_score(
    y_validation,
    optimized_prediction,
    zero_division=0
)


optimized_tn, optimized_fp, optimized_fn, optimized_tp = confusion_matrix(
    y_validation,
    optimized_prediction
).ravel()



same_recall_precision = precision_score(
    y_validation,
    same_recall_prediction,
    zero_division=0
)

same_recall_recall = recall_score(
    y_validation,
    same_recall_prediction,
    zero_division=0
)

same_recall_f1 = f1_score(
    y_validation,
    same_recall_prediction,
    zero_division=0
)


same_recall_tn, same_recall_fp, same_recall_fn, same_recall_tp = confusion_matrix(
    y_validation,
    same_recall_prediction
).ravel()




mlflow.log_metrics(
    {
        # Ranking metrics
        "validation_pr_auc": pr_auc,
        "validation_roc_auc": roc_auc,


        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,

        "validation_true_positive": int(tp),
        "validation_false_positive": int(fp),
        "validation_true_negative": int(tn),
        "validation_false_negative": int(fn),


        "best_iteration": int(
            model.best_iteration_
        ),

     
        "best_validation_threshold": float(
            best_threshold
        ),

        "optimized_validation_precision":
            optimized_precision,

        "optimized_validation_recall":
            optimized_recall,

        "optimized_validation_f1":
            optimized_f1,

        "optimized_validation_true_positive":
            int(optimized_tp),

        "optimized_validation_false_positive":
            int(optimized_fp),

        "optimized_validation_true_negative":
            int(optimized_tn),

        "optimized_validation_false_negative":
            int(optimized_fn),

        "target_recall_requirement":
            target_recall,

        "same_recall_threshold":
            float(same_recall_threshold),

        "same_recall_precision":
            same_recall_precision,

        "same_recall_recall":
            same_recall_recall,

        "same_recall_f1":
            same_recall_f1,

        "same_recall_true_positive":
            int(same_recall_tp),

        "same_recall_false_positive":
            int(same_recall_fp),

        "same_recall_true_negative":
            int(same_recall_tn),

        "same_recall_false_negative":
            int(same_recall_fn),
    }
)


print(
    "\n\n----- LightGBM Final Candidate -----"
)

print(
    "Validation PR-AUC    :",
    pr_auc
)

print(
    "Validation ROC-AUC   :",
    roc_auc
)


print(
    "\n----- Default Threshold = 0.5 -----"
)

print(
    "Precision :",
    precision
)

print(
    "Recall    :",
    recall
)

print(
    "F1        :",
    f1
)


print(
    "\nDefault Threshold Confusion Matrix"
)

print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)





print(
    "\n----- Max-F1 Validation Threshold -----"
)

print(
    "Best Threshold :",
    best_threshold
)

print(
    "Precision      :",
    optimized_precision
)

print(
    "Recall         :",
    optimized_recall
)

print(
    "F1             :",
    optimized_f1
)


print(
    "\nMax-F1 Confusion Matrix"
)

print(
    "TP:",
    optimized_tp
)

print(
    "FP:",
    optimized_fp
)

print(
    "TN:",
    optimized_tn
)

print(
    "FN:",
    optimized_fn
)




print(
    "\n----- Same Recall Comparison: "
    "Recall >= 70% -----"
)

print(
    "Threshold :",
    same_recall_threshold
)

print(
    "Precision :",
    same_recall_precision
)

print(
    "Recall    :",
    same_recall_recall
)

print(
    "F1        :",
    same_recall_f1
)


print(
    "\nSame-Recall Confusion Matrix"
)

print(
    "TP:",
    same_recall_tp
)

print(
    "FP:",
    same_recall_fp
)

print(
    "TN:",
    same_recall_tn
)

print(
    "FN:",
    same_recall_fn
)


print(
    "\nBest Iteration :",
    model.best_iteration_
)



mlflow.end_run()