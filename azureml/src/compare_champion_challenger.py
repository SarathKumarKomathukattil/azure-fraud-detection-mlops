import os
import json
from argparse import ArgumentParser

import mlflow.xgboost
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


parser = ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True
)

parser.add_argument(
    "--champion_model",
    type=str,
    required=True
)

parser.add_argument(
    "--challenger_model",
    type=str,
    required=True
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True
)

args = parser.parse_args()


FEATURES = [
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


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

columns = (
    FEATURES
    + [
        "is_fraud",
        "dataset_split",
        "data_source"
    ]
)

df = pd.read_parquet(
    args.data,
    columns=columns
)


historical_test = df[
    (df["dataset_split"] == "test")
    &
    (df["data_source"] == "historical")
].copy()


production_test = df[
    (df["dataset_split"] == "test")
    &
    (df["data_source"] == "production")
].copy()


print(
    "Historical test rows:",
    len(historical_test)
)

print(
    "Production test rows:",
    len(production_test)
)


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

champion_model = mlflow.xgboost.load_model(
    args.champion_model
)

challenger_model = mlflow.xgboost.load_model(
    args.challenger_model
)


# ------------------------------------------------------------
# LOAD EACH MODEL'S OWN THRESHOLD
# ------------------------------------------------------------

def load_threshold(model_path):

    threshold_file = os.path.join(
        model_path,
        "threshold.json"
    )

    with open(
        threshold_file,
        "r"
    ) as file:

        threshold_data = json.load(file)

    return float(
        threshold_data[
            "operating_threshold"
        ]
    )


champion_threshold = load_threshold(
    args.champion_model
)

challenger_threshold = load_threshold(
    args.challenger_model
)


print(
    "Champion threshold:",
    champion_threshold
)

print(
    "Challenger threshold:",
    challenger_threshold
)


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(
    model,
    threshold,
    evaluation_df
):

    X = evaluation_df[FEATURES]

    y = evaluation_df[
        "is_fraud"
    ]


    probability = (
        model.predict_proba(X)[:, 1]
    )


    prediction = (
        probability >= threshold
    ).astype(int)


    pr_auc = average_precision_score(
        y,
        probability
    )

    roc_auc = roc_auc_score(
        y,
        probability
    )

    precision = precision_score(
        y,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y,
        prediction,
        zero_division=0
    )


    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction
    ).ravel()


    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    return {
        "rows": int(len(evaluation_df)),
        "fraud_rows": int(y.sum()),

        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),

        "threshold": float(threshold)
    }


# ------------------------------------------------------------
# HISTORICAL HOLDOUT
# ------------------------------------------------------------

champion_historical = evaluate_model(
    champion_model,
    champion_threshold,
    historical_test
)

challenger_historical = evaluate_model(
    challenger_model,
    challenger_threshold,
    historical_test
)


# ------------------------------------------------------------
# NEW PRODUCTION HOLDOUT
# ------------------------------------------------------------

champion_production = evaluate_model(
    champion_model,
    champion_threshold,
    production_test
)

challenger_production = evaluate_model(
    challenger_model,
    challenger_threshold,
    production_test
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

comparison = {

    "champion": {
        "model_version": "2",
        "historical_test": champion_historical,
        "production_test": champion_production
    },

    "challenger": {
        "model_version": "3",
        "historical_test": challenger_historical,
        "production_test": challenger_production
    }
}


print(
    "\n========== HISTORICAL TEST =========="
)

print(
    "Champion v2:",
    champion_historical
)

print(
    "Challenger v3:",
    challenger_historical
)


print(
    "\n========== PRODUCTION TEST =========="
)

print(
    "Champion v2:",
    champion_production
)

print(
    "Challenger v3:",
    challenger_production
)


# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

os.makedirs(
    args.output_dir,
    exist_ok=True
)


output_file = os.path.join(
    args.output_dir,
    "champion_challenger_comparison.json"
)


with open(
    output_file,
    "w"
) as file:

    json.dump(
        comparison,
        file,
        indent=4
    )


print(
    "\nComparison saved to:",
    output_file
)