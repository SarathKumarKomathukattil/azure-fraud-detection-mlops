import pytest

from fraud_mlops.split_logic import (
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    calculate_split_boundaries,
)


def test_default_ratios():
    assert TRAIN_RATIO == 0.70
    assert VALIDATION_RATIO == 0.15
    assert TEST_RATIO == 0.15


def test_10000_rows_produce_expected_boundaries():
    train_end, validation_end = (
        calculate_split_boundaries(10000)
    )

    assert train_end == 7000
    assert validation_end == 8500


def test_1500_rows_produce_expected_boundaries():
    train_end, validation_end = (
        calculate_split_boundaries(1500)
    )

    assert train_end == 1050
    assert validation_end == 1275


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError):
        calculate_split_boundaries(0)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError):
        calculate_split_boundaries(
            total_rows=10000,
            train_ratio=0.80,
            validation_ratio=0.15,
            test_ratio=0.15,
        )