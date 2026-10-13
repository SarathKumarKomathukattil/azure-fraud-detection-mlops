from fraud_mlops.drift_logic import (
    calculate_psi,
    get_drift_status,
    is_drift_detected,
)


def test_identical_distributions_have_zero_psi():
    psi = calculate_psi(
        reference_counts=[50, 50],
        current_counts=[50, 50],
        reference_total=100,
        current_total=100,
    )

    assert abs(psi) < 1e-10


def test_large_distribution_change_has_high_psi():
    psi = calculate_psi(
        reference_counts=[90, 10],
        current_counts=[10, 90],
        reference_total=100,
        current_total=100,
    )

    assert psi > 0.20


def test_zero_count_is_handled_with_epsilon():
    psi = calculate_psi(
        reference_counts=[100, 0],
        current_counts=[50, 50],
        reference_total=100,
        current_total=100,
    )

    assert psi > 0


def test_drift_status_thresholds():
    assert get_drift_status(0.05) == "LOW"
    assert get_drift_status(0.10) == "MODERATE"
    assert get_drift_status(0.20) == "MODERATE"
    assert get_drift_status(0.25) == "HIGH"


def test_drift_trigger_threshold():
    assert is_drift_detected(0.19) is False
    assert is_drift_detected(0.20) is True
    assert is_drift_detected(0.47) is True