"""Tests for data loading and evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.data import (
    TRAIN_DIR, TEST_DIR, EXPECTED_SOURCE_COLS, EXPECTED_GT_COLS,
    load_source, load_ground_truth, parse_matched_ids,
)
from business_entity_resolution.evaluation import f05_single, macro_f05


def test_parse_matched_ids():
    assert parse_matched_ids("") == []
    assert parse_matched_ids("  ") == []
    assert parse_matched_ids("S2-1") == ["S2-1"]
    assert parse_matched_ids("S2-1,S3-2") == ["S2-1", "S3-2"]
    assert parse_matched_ids("S2-1, S3-2 ") == ["S2-1", "S3-2"]
    print("  PASS: parse_matched_ids")


def test_f05_single():
    # Both empty → 1.0
    assert f05_single(set(), set()) == 1.0
    # Actual empty, predicted non-empty → 0.0
    assert f05_single({"S2-1"}, set()) == 0.0
    # Predicted empty, actual non-empty → 0.0
    assert f05_single(set(), {"S2-1"}) == 0.0
    # Perfect match
    assert f05_single({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
    # Partial: pred={S2-1, S2-2, S3-1} actual={S2-1, S3-1}
    # P=2/3, R=1.0, F0.5 = 1.25*0.667*1.0 / (0.25*0.667+1.0) ≈ 0.714
    score = f05_single({"S2-1", "S2-2", "S3-1"}, {"S2-1", "S3-1"})
    assert abs(score - 0.7143) < 0.01, f"Expected ~0.714, got {score}"
    print("  PASS: f05_single")


def test_macro_f05():
    gt = {"A": {"S2-1"}, "B": set(), "C": {"S2-2", "S3-1"}}
    preds = {"A": {"S2-1"}, "B": set(), "C": {"S2-2", "S3-1"}}
    assert macro_f05(preds, gt) == 1.0
    # Missing prediction → treat as empty set
    preds2 = {"A": {"S2-1"}, "B": set()}  # C missing → 0.0 for C
    score = macro_f05(preds2, gt)
    expected = (1.0 + 1.0 + 0.0) / 3
    assert abs(score - expected) < 0.001
    print("  PASS: macro_f05")


def test_load_train_s1_schema():
    """Verify train_source1.tsv loads with correct schema and string IDs."""
    import pandas as pd
    df = load_source(TRAIN_DIR / "train_source1.tsv")
    for col in EXPECTED_SOURCE_COLS:
        assert col in df.columns, f"Missing column: {col}"
    # entity_id should be string (pandas 3.x uses StringDtype, not object)
    import pandas as pd
    assert pd.api.types.is_string_dtype(df["entity_id"]), "entity_id should be string"
    # All IDs should start with S1-
    assert df["entity_id"].str.startswith("S1-").all(), "Not all IDs start with S1-"
    print(f"  PASS: train_source1 schema ({len(df):,} rows)")


def test_load_ground_truth_schema():
    """Verify ground truth loads and parses."""
    gt = load_ground_truth()
    for col in EXPECTED_GT_COLS:
        assert col in gt.columns, f"Missing column: {col}"
    assert gt["source1_entity_id"].str.startswith("S1-").all()
    # Parse a few
    for _, row in gt.head(10).iterrows():
        ids = parse_matched_ids(row["matched_entity_ids"])
        for mid in ids:
            assert mid.startswith(("S2-", "S3-")), f"Bad prefix: {mid}"
    print(f"  PASS: ground_truth schema ({len(gt):,} rows)")


if __name__ == "__main__":
    print("Running tests...")
    test_parse_matched_ids()
    test_f05_single()
    test_macro_f05()
    test_load_train_s1_schema()
    test_load_ground_truth_schema()
    print("\nAll tests passed!")
