TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def calculate_split_boundaries(
    total_rows,
    train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO,
    test_ratio=TEST_RATIO,
):
    if total_rows <= 0:
        raise ValueError("Total rows must be greater than zero.")

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation and test ratios must sum to 1.0."
        )

    train_end = int(
        total_rows * train_ratio
    )

    validation_end = int(
        total_rows
        * (train_ratio + validation_ratio)
    )

    return train_end, validation_end