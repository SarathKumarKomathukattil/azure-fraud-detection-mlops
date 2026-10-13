import math


PSI_LOW_THRESHOLD = 0.10
PSI_HIGH_THRESHOLD = 0.25
DRIFT_THRESHOLD = 0.20

EPSILON = 1e-6


def calculate_psi(
    reference_counts,
    current_counts,
    reference_total,
    current_total,
    epsilon=EPSILON,
):
    if len(reference_counts) != len(current_counts):
        raise ValueError(
            "Reference and current counts must have the same number of bins."
        )

    if reference_total <= 0 or current_total <= 0:
        raise ValueError(
            "Reference and current totals must be greater than zero."
        )

    psi = 0.0

    for reference_count, current_count in zip(
        reference_counts,
        current_counts,
    ):
        reference_pct = max(
            reference_count / reference_total,
            epsilon,
        )

        current_pct = max(
            current_count / current_total,
            epsilon,
        )

        psi += (
            (current_pct - reference_pct)
            * math.log(current_pct / reference_pct)
        )

    return float(psi)


def get_drift_status(psi):
    if psi < PSI_LOW_THRESHOLD:
        return "LOW"

    if psi < PSI_HIGH_THRESHOLD:
        return "MODERATE"

    return "HIGH"


def is_drift_detected(psi):
    return psi >= DRIFT_THRESHOLD