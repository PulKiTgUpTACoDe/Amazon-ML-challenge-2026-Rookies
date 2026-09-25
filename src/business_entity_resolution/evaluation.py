"""
Evaluation utilities — per-entity macro-average F0.5.

The competition metric:
    F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
computed per Source 1 entity, then macro-averaged.
"""


def f05_single(predicted: set, actual: set) -> float:
    """Compute F0.5 for a single Source 1 entity.

    Both ``predicted`` and ``actual`` are sets of matched entity IDs.

    Special cases:
    - actual empty, predicted empty  → 1.0  (correct singleton)
    - actual empty, predicted non-empty → 0.0  (false merge on singleton)
    - actual non-empty, predicted empty → 0.0  (missed all matches)
    """
    if len(actual) == 0 and len(predicted) == 0:
        return 1.0
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0

    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual)

    if precision + recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
) -> float:
    """Compute macro-averaged F0.5 over all Source 1 entities in ``ground_truth``.

    ``predictions`` and ``ground_truth`` are dicts mapping S1 entity_id → set of
    matched S2/S3 entity IDs.

    Every key in ``ground_truth`` must be scored. If a key is missing from
    ``predictions``, it is treated as predicting no matches.
    """
    scores = []
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        scores.append(f05_single(predicted, actual))
    return sum(scores) / len(scores) if scores else 0.0
