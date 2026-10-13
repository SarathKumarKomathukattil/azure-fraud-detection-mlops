from fraud_mlops.ct_logic import (
    MINIMUM_RECALL,
    MINIMUM_PR_AUC,
    MINIMUM_ROC_AUC,
    evaluate_ct_trigger,
)


def test_ct_thresholds():
    assert MINIMUM_RECALL == 0.70
    assert MINIMUM_PR_AUC == 0.26736296521775227 * 0.90
    assert MINIMUM_ROC_AUC == 0.9212790418374809 - 0.03


def test_healthy_model_does_not_trigger_retraining():
    result = evaluate_ct_trigger(
        overall_drift=False,
        recall=0.75,
        pr_auc=0.30,
        roc_auc=0.92,
    )

    assert result["should_retrain"] is False
    assert result["trigger_reason"] == "NO_TRIGGER"


def test_drift_triggers_retraining():
    result = evaluate_ct_trigger(
        overall_drift=True,
        recall=0.75,
        pr_auc=0.30,
        roc_auc=0.92,
    )

    assert result["should_retrain"] is True
    assert result["trigger_reason"] == "DATA_DRIFT"


def test_low_recall_triggers_retraining():
    result = evaluate_ct_trigger(
        overall_drift=False,
        recall=0.60,
        pr_auc=0.30,
        roc_auc=0.92,
    )

    assert result["should_retrain"] is True
    assert result["recall_failed"] is True
    assert result["trigger_reason"] == "LOW_RECALL"


def test_multiple_failures():
    result = evaluate_ct_trigger(
        overall_drift=True,
        recall=0.48,
        pr_auc=0.004,
        roc_auc=0.59,
    )

    assert result["should_retrain"] is True

    assert result["trigger_reason"] == (
        "DATA_DRIFT," "LOW_RECALL," "LOW_PR_AUC," "LOW_ROC_AUC"
    )
