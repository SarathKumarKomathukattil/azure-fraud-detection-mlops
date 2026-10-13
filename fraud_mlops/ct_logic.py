VALIDATION_PR_AUC = 0.26736296521775227
VALIDATION_ROC_AUC = 0.9212790418374809

MINIMUM_RECALL = 0.70
MINIMUM_PR_AUC = VALIDATION_PR_AUC * 0.90
MINIMUM_ROC_AUC = VALIDATION_ROC_AUC - 0.03


def evaluate_ct_trigger(
    overall_drift,
    recall,
    pr_auc,
    roc_auc,
):
    recall_failed = recall < MINIMUM_RECALL
    pr_auc_failed = pr_auc < MINIMUM_PR_AUC
    roc_auc_failed = roc_auc < MINIMUM_ROC_AUC

    should_retrain = (
        overall_drift
        or recall_failed
        or pr_auc_failed
        or roc_auc_failed
    )

    trigger_reasons = []

    if overall_drift:
        trigger_reasons.append("DATA_DRIFT")

    if recall_failed:
        trigger_reasons.append("LOW_RECALL")

    if pr_auc_failed:
        trigger_reasons.append("LOW_PR_AUC")

    if roc_auc_failed:
        trigger_reasons.append("LOW_ROC_AUC")

    if not trigger_reasons:
        trigger_reasons.append("NO_TRIGGER")

    return {
        "recall_failed": recall_failed,
        "pr_auc_failed": pr_auc_failed,
        "roc_auc_failed": roc_auc_failed,
        "should_retrain": should_retrain,
        "trigger_reason": ",".join(trigger_reasons),
    }