import os
import sys
import json
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--metrics_input", type=str, required=True)

parser.add_argument("--gate_output", type=str, required=True)

args = parser.parse_args()

metrics_file = os.path.join(args.metrics_input, "metrics.json")

with open(metrics_file, "r") as file:
    metrics = json.load(file)

test_recall = metrics["test_recall"]
test_pr_auc = metrics["test_pr_auc"]
test_roc_auc = metrics["test_roc_auc"]

final_operating_threshold = metrics["final_operating_threshold"]
validation_pr_auc = 0.26736296521775227
validation_roc_auc = 0.9212790418374809

minimum_recall = 0.70

minimum_pr_auc = validation_pr_auc * 0.90
minimum_roc_auc = validation_roc_auc - 0.03

gates_passed = True


if test_recall >= minimum_recall:
    print(f"Recall check: PASS " f"({test_recall:.4f} >= {minimum_recall:.4f})")
else:
    print(f"Recall check: FAILED " f"({test_recall:.4f} < {minimum_recall:.4f})")
    gates_passed = False


if test_pr_auc >= minimum_pr_auc:
    print(f"PR-AUC check: PASS " f"({test_pr_auc:.4f} >= {minimum_pr_auc:.4f})")
else:
    print(f"PR-AUC check: FAILED " f"({test_pr_auc:.4f} < {minimum_pr_auc:.4f})")
    gates_passed = False


if test_roc_auc >= minimum_roc_auc:
    print(f"ROC-AUC check: PASS " f"({test_roc_auc:.4f} >= {minimum_roc_auc:.4f})")
else:
    print(f"ROC-AUC check: FAILED " f"({test_roc_auc:.4f} < {minimum_roc_auc:.4f})")
    gates_passed = False


if gates_passed:
    status = "PASS"
else:
    status = "FAIL"


gate_result = {
    "status": status,
    "test_recall": test_recall,
    "minimum_recall": minimum_recall,
    "test_pr_auc": test_pr_auc,
    "minimum_pr_auc": minimum_pr_auc,
    "test_roc_auc": test_roc_auc,
    "minimum_roc_auc": minimum_roc_auc,
    "operating_threshold": final_operating_threshold
}

os.makedirs(args.gate_output,exist_ok=True)

gate_file = os.path.join(args.gate_output,'gate_result.json')

with open(gate_file,'w') as file:
    json.dump(gate_result,file,indent=4)

print("Gate result saved to:", gate_file)

if gates_passed:
    print("\nMODEL QUALITY GATE PASSED")
else:
    print("\nMODEL QUALITY GATE FAILED")
    sys.exit(1)
